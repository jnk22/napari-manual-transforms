"""TODO."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final, Literal

import napari
from cyclopts import App
from cyclopts.types import File, PositiveInt  # noqa: TC002
from loguru import logger
from ndimreg.registration import (
    BaseRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    TranslationFFT3DRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndimreg.image import Image

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
def register(  # noqa: PLR0913
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
) -> None:
    """Manually transform the 'moving' image and auto-register with 'fixed' image.

    This mode can be used to test any 3D registration method by
    registering two input images. Both input will images will be loaded
    as 'fixed' and 'moving' while the moving image can be freely
    transformed to be registered with the fixed image.

    The first input is the 'fixed' image. The second image is the
    'moving' image.
    """
    from ndimreg.image import Paths3DImageLoader
    from ndimreg.utils.image import prepare_benchmark_image

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
def transform(  # noqa: PLR0913
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
) -> None:
    """Manually transform and auto-register the original image and a copy of it.

    This mode can be used to test any 3D registration method by
    transformed to be registered with the fixed image.
    """
    from ndimreg.image import Image3D
    from ndimreg.utils.image import prepare_benchmark_image

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


def _start_napari(  # noqa: PLR0913
    images: Sequence[Image],
    method: RegistrationMethod3D,
    mode: Literal["transform", "register"],
    spacing: tuple[float, float, float] | None,
    upsample_factor: PositiveInt = 1,
    rotation_axis: RotationAxis3D = "z",
    *,
    debug: bool = False,
) -> None:
    from napari.layers import Image as NapariImage

    from ._widget import TransformationWidget

    registration = REGISTRATION_METHODS[method](
        upsample_factor=upsample_factor,
        axis=rotation_axis,
        debug=debug,
        rotation_axis_normalization=False,
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
