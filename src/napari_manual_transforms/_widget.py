from __future__ import annotations

import contextlib
import datetime
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np
import pytransform3d.rotations as pr
from cachetools import cached
from cachetools.keys import hashkey
from loguru import logger
from napari.layers import Image
from ndimreg.fusion import MergeFusion
from ndimreg.registration import BaseRegistration, Keller3DRegistration
from ndimreg.transform import transform, transform_matrix
from ndimreg.utils import to_device_arrays
from qtpy.QtWidgets import QCheckBox, QLabel, QPushButton, QWidget
from vispy.util.keys import ALT

from ._model import MINIMUM_SCALE
from ._tform_widget import RegistrationConfig, TransformationView
from ._util import _Quaternion, transform_array_3d

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer
    from napari.utils.events import Event
    from ndimreg.image import Device
    from numpy.typing import NDArray

__AVAILABLE_DEVICES: Final[set[Device]] = {"cpu", "gpu"}
__DEVICE_ENV: Final = os.getenv("DEVICE", "cpu").lower().strip()
__DEVICE: Final = __DEVICE_ENV if __DEVICE_ENV in __AVAILABLE_DEVICES else "cpu"


def __strtobool(val: str) -> bool:
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return True

    if val in {"n", "no", "f", "false", "off", "0"}:
        return False

    msg = f"Invalid truth value: {val}"
    raise ValueError(msg)


def norm_arr(image: NDArray, minx: float | None = None) -> NDArray:
    minx = minx if minx is not None else np.min(image)
    return (image - minx) / (np.max(image) - minx)


@dataclass(frozen=True, slots=True)
class HashableArray:
    """Wrapper to allow for hashing of NumPy arrays."""

    shape: tuple
    dtype: np.dtype
    data: bytes

    @classmethod
    def from_ndarray(cls, arr: NDArray) -> Self:
        """Create a HashableArray from a NumPy array."""
        return cls(shape=arr.shape, dtype=arr.dtype, data=arr.tobytes())

    def to_ndarray(self) -> NDArray:
        """Convert the HashableArray back to a NumPy array."""
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)


@cached(
    cache={},
    key=lambda _registration, config, tform, _origin, _fixed, _moving: hashkey(
        tform, config
    ),
)
def update_images_cached(
    registration: BaseRegistration,
    config: RegistrationConfig,
    transformation: HashableArray,
    origin: NDArray,
    fixed: NDArray,
    moving: NDArray,
) -> tuple[NDArray, list[Image]]:
    fixed = (
        transform(fixed, rotation=pr.random_quaternion(), dim=3)
        if __strtobool(os.getenv("RANDOM", "False"))
        else fixed.copy()
    )
    moving = moving.copy()

    fixed[fixed < config.threshold] = 0
    moving[moving < config.threshold] = 0

    tform_matrix = transformation.to_ndarray()
    logger.info(f"Transformation matrix:\n{tform_matrix}")

    moving_trans = transform(moving, matrix=tform_matrix, dim=3, inverse=True)
    moving_trans[moving_trans < config.threshold] = 0

    arrays_on_device = to_device_arrays(fixed, moving_trans, device=__DEVICE)
    result = registration.register(*arrays_on_device)

    logger.info(f"Registration result: {result.transformation}")
    logger.info(f"Registration duration: {result.total_duration:.2f}s")

    if result.transformation is None:
        msg = "Registration failed"
        raise ValueError(msg)

    recovery_matrix = transform_matrix(
        dim=3,
        translation=result.transformation.translation,
        rotation=result.transformation.rotation,
        scale=result.transformation.scale,
        origin=origin,
    )
    logger.info(f"Recovery matrix:\n{recovery_matrix}")

    recovered = transform(moving_trans, matrix=recovery_matrix, dim=3, inverse=True)
    fused = MergeFusion().fuse(fixed, recovered)

    full_matrix = recovery_matrix @ tform_matrix
    logger.info(f"Final matrix:\n{full_matrix}")

    if registration.debug:
        # FIX: Must be updated.
        debug_images_2d = []
        debug_images_3d = []
        scale = (1,) * 3

        # debug_images_2d = [
        #     Image(np.expand_dims(im.data, axis=0), name=im.name)
        #     for im in registration.debug_images["registration"]
        # ]
        # scale = (max(debug_images_2d[0].data.shape) / max(fixed.shape),) * 3
        # debug_images_3d = [
        #     Image(im.data, name=im.name, scale=scale)
        #     for im in registration.debug_images_3d["registration"]
        # ]
    else:
        debug_images_2d = []
        debug_images_3d = []
        scale = (1,) * 3

    images = [
        Image(fixed, name="fixed", scale=scale),
        Image(moving_trans, name="moving", scale=scale),
        Image(recovered, name="recovered", scale=scale),
        Image(fused, name="fused", scale=scale),
        *debug_images_3d,
        *debug_images_2d,
    ]

    return full_matrix, images


class LayerFollower(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer, parent=None) -> None:
        assert viewer
        self._viewer: napari.Viewer | None = None
        self._active: napari.layers.Layer | None = None
        super().__init__(parent)
        self._connect_viewer(viewer)

    # viewer connections

    def _connect_viewer(self, viewer: napari.Viewer):
        self._viewer = viewer
        sel_events = viewer.layers.selection.events
        sel_events.active.connect(self._on_active_layer_change)
        if active := viewer.layers.selection.active:
            self._connect_layer(active)

    def _disconnect_viewer(self):
        if self._viewer is not None:
            sel_events = self._viewer.layers.selection.events
            sel_events.changed.disconnect(self._on_active_layer_change)
        self._viewer = None

    def _on_active_layer_change(self, event: Event):
        if self._active is not None:
            self._disconnect_layer()
        if event.value is not None:
            self._connect_layer(event.value)

    # layer connections

    def _connect_layer(self, layer: napari.layers.Layer):
        self._active = layer
        layer.events.connect(self._on_layer_event)

    def _disconnect_layer(self):
        if self._active is not None:
            self._active.events.disconnect(self._on_layer_event)
        self._active = None

    def _on_layer_event(self, event):
        # layer = event.source
        # attr = event.type
        # value = getattr(layer, attr, None)
        ...

    # cleanup

    def __del__(self):
        with contextlib.suppress(Exception):
            self._disconnect_viewer()
            self._disconnect_layer()


class TransformationWidget(LayerFollower, TransformationView):
    def __init__(
        self,
        viewer: napari.viewer.Viewer | None = None,
        parent=None,
        registration: BaseRegistration | None = None,
        output_dir: Path | str = "output",
        *,
        auto_registration: bool = True,
        spacing: tuple[float, float, float] | None = None,
    ):
        self._spacing: tuple[float, float, float] | None = spacing
        self._registration: BaseRegistration = registration or Keller3DRegistration()
        self._tform_matrix: NDArray | None = None
        self._output_dir: Path = Path(output_dir)

        self._help = QLabel("(hold alt while dragging canvas to edit)")
        self._help.setStyleSheet("font-size: 9pt; color: #AAA; text-align: center;")
        super().__init__(viewer, parent)

        self.layout().insertWidget(0, self._help)

        self._center_orig = QPushButton("center origin")
        self._center_orig.clicked.connect(self._center_origin)
        self.layout().addWidget(self._center_orig)

        self._resample_btn = QPushButton("resample")
        self._resample_btn.clicked.connect(self._resample)
        self.layout().addWidget(self._resample_btn)

        self._export_result_btn = QPushButton("export results")
        self._export_result_btn.clicked.connect(self._export_results)
        self.layout().addWidget(self._export_result_btn)

        self._auto_registration_checkbox = QCheckBox("Auto-registration")
        self._auto_registration_checkbox.setChecked(auto_registration)
        self.layout().addWidget(self._auto_registration_checkbox)

        # try:
        #     self._layer = viewer.layers["rotation axis"]
        # except (AttributeError, KeyError):
        #     self._layer = viewer.add_vectors(
        #         self._model.rotation_vector, name="rotation axis"
        #     )

        self._model.valueChanged.connect(self._on_model_changed)
        self._on_model_changed()

    def _connect_viewer(self, viewer: napari.Viewer):
        super()._connect_viewer(viewer)
        viewer.mouse_drag_callbacks.append(self._on_mouse_drag)

    def _disconnect_viewer(self):
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        super()._disconnect_viewer()

    def _on_model_changed(self):
        self._update_active()

    def _center_origin(self):
        if self._active is not None:
            self._model.origin = np.asarray(self._active.data.shape) // 2

    def _update_active(self) -> None:
        # TODO... add support for 2d
        if not isinstance(self._active, Image) or self._active.data.ndim < 3:
            self.setEnabled(False)
            return

        self.setEnabled(True)
        with self._model.valueChanged.blocked():
            self._active.affine = self._model.transform
            if self._auto_registration_checkbox.isChecked() and self._viewer:
                tform_matrix, updated_layers = update_images_cached(
                    self._registration,
                    RegistrationConfig(self._model.config_threshold),
                    HashableArray.from_ndarray(self._model.transform),
                    self._model.origin,
                    *(x.data for x in self._viewer.layers[1::-1]),
                )

                self._tform_matrix = tform_matrix
                idx = 2

                if len(self._viewer.layers) == idx:
                    self._viewer.layers.extend(updated_layers)
                else:
                    self._viewer.layers[idx:] = updated_layers

                for layer in self._viewer.layers[:idx]:
                    layer.visible = False

                # We set the current's view to the middle of the Z axis
                # as convenience.
                fixed_im = self._viewer.layers[idx]
                z_mid_layer_index = fixed_im.data.shape[0] * fixed_im.scale[0] // 2
                self._viewer.dims.set_point(0, z_mid_layer_index)

                self._viewer.layers.selection.active = self._viewer.layers[0]

    def _on_mouse_drag(self, viewer, event):
        """Update layer affine when alt-dragging."""
        if self._active is None or ALT not in event.modifiers:
            return

        q = _Quaternion(*self._model.quaternion)
        p2 = None
        wh = event.source.size
        yield

        while event.type == "mouse_move":
            p1, p2 = p2, event.pos
            if p1 is None:
                p1 = p2

            qp2 = _Quaternion.from_arcball(p1, wh)
            qp1 = _Quaternion.from_arcball(p2, wh)
            q = qp2 * qp1 * q

            _q = np.array([q.w, q.x, q.y, q.z])

            if np.allclose(self._model.quaternion, -_q, atol=0.2):
                _q *= -1

            self._model.quaternion = _q
            yield

    def _on_layer_event(self, event):
        if event.type == "affine":
            self._update_from_layer()

    def _update_from_layer(self):
        if self._active is None:
            return

        with self._model.valueChanged.blocked():
            # Affine matrix decomposition based on
            # https://math.stackexchange.com/a/1463487
            affine_matrix = self._active.affine.affine_matrix

            # Extract scaling only from first row of the rotation
            # matrix as only single scaling factor is allowed as input.
            scale = np.linalg.norm(affine_matrix[0, :3]).item()
            self._model.scale = scale

            # Adjust translation based on origin offset.
            origin = self._model.origin
            T = np.eye(4)
            T[:3, -1] = origin
            self._model.translation = (affine_matrix @ T)[:3, 3] - origin

            # Prevent division by zero by using at least the minimum
            # scaling value that is allowed as scaling input.
            self._model.matrix = affine_matrix[:3, :3] / max(scale, MINIMUM_SCALE)

        self._model.valueChanged.emit()

    def _connect_layer(self, layer: napari.layers.Layer):
        super()._connect_layer(layer)
        self._update_from_layer()
        self._help.show()

    def _disconnect_layer(self):
        super()._disconnect_layer()
        self._help.hide()

    def _resample(self):
        if self._active is not None:
            data, translate = transform_array_3d(
                self._active.data, self._model.matrix, self._model.origin
            )
            new_layer = type(self._active)(
                data,
                translate=translate,
                blending="additive",
                name=f"resampled {self._active.name}",
            )
            self._viewer.add_layer(new_layer)

    def _export_results(self):
        if self._tform_matrix is not None and self._viewer:
            now = datetime.datetime.now(tz=datetime.UTC)

            out_dir = self._output_dir / now.strftime("%Y%m%d-%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            tform_matrix_path = out_dir / Path("tform_matrix.npy")
            np.save(tform_matrix_path, self._tform_matrix)

            rotation_matrix_path = out_dir / Path("rotation_matrix.npy")
            np.save(rotation_matrix_path, self._tform_matrix[:3, :3])

            tform_text_path = out_dir / Path("tform_matrix.txt")
            tform_text_path.write_text(str(self._tform_matrix))

            screenshot_full_path = out_dir / "screenshot-full.png"
            screenshot_canvas_path = out_dir / "screenshot-canvas.png"
            self._viewer.screenshot(str(screenshot_full_path), canvas_only=False)
            self._viewer.screenshot(str(screenshot_canvas_path), canvas_only=True)

            image_names = (im.name for im in self._viewer.layers[1::-1])

            size = len(self._viewer.layers[0].data)
            spacing = ":".join(str(v) for v in self._spacing) if self._spacing else "/"
            time = now.strftime("%Y-%m-%d %H:%M:%S")
            method = self._registration.__class__.__name__
            command = " ".join(sys.argv)

            values = (*image_names, size, spacing, time, method, command)
            text = ("Fixed", "Moving", "Size", "Spacing", "Time", "Method", "Command")
            meta = "\n".join(f"{x}: {y}" for x, y in zip(text, values, strict=True))

            metadata_txt_path = out_dir / Path("metadata.txt")
            metadata_txt_path.write_text(meta)


if __name__ == "__main__":
    import napari

    v = napari.Viewer()
    v.open_sample("napari", "cells3d")
    v.dims.ndisplay = 3
    wdg = TransformationWidget(v)
    v.window.add_dock_widget(wdg)
    napari.run()
