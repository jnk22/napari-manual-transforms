"""TODO."""

import os
import sys
from collections.abc import Sequence
from typing import Final, Literal

import napari
from cyclopts import App
from cyclopts.types import File, PositiveInt
from imreg3d.image import Image
from imreg3d.registration import (
    BaseRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    TranslationFFT3DRegistration,
)
from loguru import logger
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

logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))


@app.command
def register(
    image_paths: tuple[File, File],
    method: RegistrationMethod3D = "keller",
    resize: int | None = 64,
    spacing: tuple[float, float, float] | None = None,
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    normalize: bool = True,
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
    from imreg3d.utils.image import prepare_benchmark_image

    images = list(Paths3DImageLoader(image_paths))

    for im in images:
        prepare_benchmark_image(
            im,
            normalize=normalize,
            resize=resize,
            spacing=spacing,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

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
    spacing: tuple[float, float, float] | None = None,
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    normalize: bool = True,
    max_pad: bool = False,
    safe_pad: bool = False,
    debug: bool = False,
):
    """Manually transform and auto-register the original image and a copy of it.

    This mode can be used to test any 3D registration method by
    transformed to be registered with the fixed image.
    """
    from imreg3d.image import Image3D
    from imreg3d.utils.image import prepare_benchmark_image

    images = [Image3D.from_path(image_path)]

    for im in images:
        prepare_benchmark_image(
            im,
            normalize=normalize,
            resize=resize,
            spacing=spacing,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

    _start_napari(
        images,
        method,
        "transform",
        spacing,
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


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
