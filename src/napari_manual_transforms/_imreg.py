"""TODO."""

from collections.abc import Sequence
from typing import Final, Literal

import napari
import numpy as np
from cyclopts import App
from cyclopts.types import File, PositiveInt
from imreg3d.image import Image
from imreg3d.registration import (
    BaseRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    TranslationFFT3DRegistration,
)
from napari.layers import Image as NapariImage

from ._widget import TransformationWidget

RegistrationMethod3D = Literal["keller", "rotationaxis", "translation"]
RotationAxis3D = Literal["x", "y", "z"]

REGISTRATION_METHODS: Final[dict[str, type[BaseRegistration]]] = {
    "keller": Keller3DRegistration,
    "translation": TranslationFFT3DRegistration,
    "rotationaxis": RotationAxis3DRegistration,
}
REGISTRATION_CHOICES = list(REGISTRATION_METHODS.keys())

app = App()


@app.command
def register(
    image_paths: tuple[File, File],
    method: RegistrationMethod3D = "keller",
    resize: int | None = 64,
    spacing: tuple[float, float, float] = (1, 1, 1),
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    max_pad: bool = False,
    safe_pad: bool = False,
    debug: bool = False,
):
    """Manually transform the 'moving' image and auto-register with 'fixed' image.

    This mode can be used to test any 3D registration method by
    registering two input images. Both input will images will be loaded
    as 'fixed' and 'moving' while the moving image can be freely
    transformed to be registered with the fixed image.

    The first input is the 'fixed' image. The second image is the
    'moving' image.
    """
    from imreg3d.image import Paths3DImageLoader

    images = list(Paths3DImageLoader(image_paths))
    _prepare_images(images, resize, spacing, max_pad=max_pad, safe_pad=safe_pad)
    _start_napari(
        images[::-1],
        method,
        "register",
        spacing,
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


@app.command
def transform(
    image_path: File,
    method: RegistrationMethod3D = "keller",
    resize: int | None = 64,
    spacing: tuple[float, float, float] = (1, 1, 1),
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    max_pad: bool = False,
    safe_pad: bool = False,
    debug: bool = False,
):
    """Manually transform and auto-register the original image and a copy of it.

    This mode can be used to test any 3D registration method by
    transformed to be registered with the fixed image.
    """
    from imreg3d.image import Image3D

    images = [Image3D.from_path(image_path)]
    _prepare_images(images, resize, spacing, max_pad=max_pad, safe_pad=safe_pad)
    _start_napari(
        images,
        method,
        "transform",
        spacing,
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


def _prepare_images(
    images: Sequence[Image],
    resize: int | None,
    spacing: tuple[float, ...] | None = None,
    *,
    max_pad: bool = False,
    safe_pad: bool = False,
) -> None:
    target_shape = np.array(images[0].resolution, dtype=float) * spacing
    target_shape /= max(target_shape) / (resize or 1)

    for im in images:
        im.resize_to_shape(tuple(target_shape.round().astype(int)))

    if max_pad:
        for im in images:
            im.pad_equal_sides()

    if safe_pad:
        for im in images:
            im.pad_safe_rotation(keep_shape=True)


def _start_napari(
    images: Sequence[Image],
    method: RegistrationMethod3D,
    mode: Literal["transform", "register"],
    spacing: tuple[float, float, float] | None,
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    debug: bool = False,
) -> None:
    registration = REGISTRATION_METHODS[method](
        upsample_factor=upsample_factor, axis=rotation_axis, debug=debug
    )

    v = napari.Viewer()
    for im in images:
        v.add_layer(NapariImage(im.data, name=im.name))

    v.dims.ndisplay = 3
    v.window.add_dock_widget(
        TransformationWidget(
            viewer=v, registration=registration, mode=mode, spacing=spacing
        )
    )
    napari.run()
