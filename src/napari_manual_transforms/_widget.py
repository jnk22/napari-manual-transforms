from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from imreg3d.fusion import MergeFusion
from imreg3d.registration import BaseRegistration, Keller3DRegistration
from imreg3d.transform import transform_nd
from loguru import logger
from napari.layers import Image
from qtpy.QtWidgets import QLabel, QPushButton, QWidget
from vispy.util.keys import ALT

from napari_manual_transforms._model import MINIMUM_SCALE
from napari_manual_transforms._tform_widget import TransformationView
from napari_manual_transforms._util import _Quaternion, transform_array_3d

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer
    from napari.utils.events import Event
    from numpy.typing import NDArray


@dataclass(frozen=True)
class HashableArray:
    """Wrapper to allow for hashing of NumPy arrays."""

    shape: tuple
    dtype: np.dtype
    data: bytes

    @classmethod
    def from_ndarray(cls, arr: NDArray) -> Self:
        """Creates a HashableArray from a NumPy array."""
        return cls(shape=arr.shape, dtype=arr.dtype, data=arr.tobytes())

    def to_ndarray(self) -> NDArray:
        """Converts the HashableArray back to a NumPy array."""
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)


# TODO: Only hash based on transformation matrix for better performance.
@functools.cache
def update_images_cached(
    registration: BaseRegistration, tform: HashableArray, data: HashableArray
) -> list[Image]:
    fixed = data.to_ndarray()
    moving = transform_nd(fixed, matrix=tform.to_ndarray(), dim=3, inverse=True)
    result = registration.register(fixed, moving)

    recovered = transform_nd(
        moving,
        dim=3,
        translation=result.translation,
        rotation=result.rotation,
        inverse=True,
    )
    fused = MergeFusion().fuse(fixed, recovered)

    if registration.debug:
        debug_images_2d = [
            Image(im.data, name=im.name)
            for im in registration.debug_images["registration"]
        ]
        scale = (max(debug_images_2d[0].data.shape) / max(fixed.shape),) * 3
        debug_images_3d = [
            Image(im.data, name=im.name, scale=scale)
            for im in registration.debug_images_3d["registration"]
        ]
    else:
        debug_images_2d = []
        debug_images_3d = []
        scale = (1,) * 3

    return [
        Image(fixed, name="fixed", scale=scale),
        Image(moving, name="moving", scale=scale),
        Image(recovered, name="recovered", scale=scale),
        Image(fused, name="fused", scale=scale),
        *debug_images_3d,
        *debug_images_2d,
    ]


def update_images_cached_register(
    registration: BaseRegistration,
    tform: HashableArray,
    fixed: HashableArray,
    moving: HashableArray,
) -> list[Image]:
    fixed_ = fixed.to_ndarray()
    moving_ = moving.to_ndarray()
    moving_transformed = transform_nd(
        moving_, matrix=tform.to_ndarray(), dim=3, inverse=True
    )
    result = registration.register(fixed_, moving_transformed)
    logger.info(result)

    recovered = transform_nd(
        moving_transformed,
        dim=3,
        translation=result.translation,
        rotation=result.rotation,
        inverse=True,
    )
    fused = MergeFusion().fuse(fixed_, recovered)

    if registration.debug:
        debug_images_2d = [
            Image(im.data, name=im.name)
            for im in registration.debug_images["registration"]
        ]
        scale = (max(debug_images_2d[0].data.shape) / max(fixed.shape),) * 3
        debug_images_3d = [
            Image(im.data, name=im.name, scale=scale)
            for im in registration.debug_images_3d["registration"]
        ]
    else:
        debug_images_2d = []
        debug_images_3d = []
        scale = (1,) * 3

    return [
        Image(fixed_, name="fixed", scale=scale),
        Image(moving_transformed, name="moving", scale=scale),
        Image(recovered, name="recovered", scale=scale),
        Image(fused, name="fused", scale=scale),
        *debug_images_3d,
        *debug_images_2d,
    ]


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
        mode: Literal["transform", "register"] | None = None,
        viewer: napari.viewer.Viewer | None = None,
        parent=None,
        registration: BaseRegistration | None = None,
    ):
        self.mode: Literal["transform", "register"] | None = mode
        self.registration = registration or Keller3DRegistration()

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
            match self.mode:
                case "transform":
                    self.__imreg3d_code_tform()
                case "register":
                    self.__imreg3d_code_register()

    def __imreg3d_code_register(self):
        selected = self._active
        tform = self._model.transform
        self._active.affine = tform

        moving = self._viewer.layers[0].data
        fixed = self._viewer.layers[1].data

        updated_layers = update_images_cached_register(
            self.registration,
            HashableArray.from_ndarray(tform),
            HashableArray.from_ndarray(fixed),
            HashableArray.from_ndarray(moving),
        )

        if len(self._viewer.layers) == 1:
            self._viewer.layers.extend(updated_layers)
        else:
            self._viewer.layers[2:] = updated_layers

        fixed_im = self._viewer.layers[2]
        z_mid = fixed_im.data.shape[0] * fixed_im.scale[0] // 2
        self._viewer.dims.set_point(0, z_mid)
        self._viewer.layers.selection.active = selected

    def __imreg3d_code_tform(self):
        selected = self._active
        tform = self._model.transform
        self._active.affine = tform

        updated_layers = update_images_cached(
            self.registration,
            HashableArray.from_ndarray(tform),
            HashableArray.from_ndarray(selected.data),
        )

        if len(self._viewer.layers) == 1:
            self._viewer.layers.extend(updated_layers)
        else:
            self._viewer.layers[1:] = updated_layers

        fixed_im = self._viewer.layers[1]
        z_mid = fixed_im.data.shape[0] * fixed_im.scale[0] // 2
        self._viewer.dims.set_point(0, z_mid)
        self._viewer.layers.selection.active = selected

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
        if self._active is not None and self._active == self._viewer.layers[0]:
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


if __name__ == "__main__":
    import napari

    v = napari.Viewer()
    v.open_sample("napari", "cells3d")
    v.dims.ndisplay = 3
    wdg = TransformationWidget(v)
    v.window.add_dock_widget(wdg)
    napari.run()
