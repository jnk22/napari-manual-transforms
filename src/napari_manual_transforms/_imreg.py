from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Optional, TypeAlias

import napari
import numpy as np
from click import Choice
from imreg3d.image import Image, Image3D
from imreg3d.registration import (
    BaseRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    TranslationFFT3DRegistration,
)
from napari.layers import Image as NapariImage
from typer import Option, Typer

from ._widget import TransformationWidget

FloatOrNone: TypeAlias = Optional[float]  # noqa: UP007
IntOrNone: TypeAlias = Optional[int]  # noqa: UP007

REGISTRATION_METHODS: Final[dict[str, type[BaseRegistration]]] = {
    "keller": Keller3DRegistration,
    "translation": TranslationFFT3DRegistration,
    "rotationaxis": RotationAxis3DRegistration,
}
REGISTRATION_CHOICES = list(REGISTRATION_METHODS.keys())

RegistrationMethod3D = Annotated[str, Option(click_type=Choice(REGISTRATION_CHOICES))]
RotationInput = Annotated[FloatOrNone, Option()]
ScaleInput = Annotated[FloatOrNone, Option(min=0)]
UpsampleFactorInput = Annotated[int, Option(min=1)]
ResizeInput = Annotated[IntOrNone, Option(min=1)]
SafePadInput = Annotated[bool, Option()]
MaxPadInput = Annotated[bool, Option()]
RotationAxisRecovery = Annotated[str, Option(click_type=Choice(["x", "y", "z"]))]

cli = Typer()


@cli.command()
def register(
    image_paths: tuple[Path, Path],
    method: RegistrationMethod3D = "keller",
    resize: ResizeInput = 64,
    spacing: tuple[float, float, float] | None = None,
    upsample_factor: UpsampleFactorInput = 1,
    rotation_axis: RotationAxisRecovery = "z",
    *,
    max_pad: MaxPadInput = False,
    safe_pad: SafePadInput = False,
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
    images = [Image3D.from_path(im_path) for im_path in image_paths]
    __prepare_images(images, resize, spacing, max_pad=max_pad, safe_pad=safe_pad)
    __start_napari(
        list(reversed(images)),
        method,
        "register",
        spacing,
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


@cli.command()
def transform(
    image_path: Path,
    method: RegistrationMethod3D = "keller",
    resize: ResizeInput = 64,
    spacing: tuple[float, float, float] | None = None,
    upsample_factor: UpsampleFactorInput = 1,
    rotation_axis: RotationAxisRecovery = "z",
    *,
    max_pad: MaxPadInput = False,
    safe_pad: SafePadInput = False,
    debug: bool = False,
):
    """Manually transform and auto-register the original image and a copy of it.

    This mode can be used to test any 3D registration method by
    transformed to be registered with the fixed image.
    """
    images = [Image3D.from_path(image_path)]
    __prepare_images(images, resize, spacing, max_pad=max_pad, safe_pad=safe_pad)
    __start_napari(
        images,
        method,
        "transform",
        spacing,
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


def __prepare_images(
    images: Sequence[Image],
    resize: ResizeInput,
    spacing: tuple[float, ...] | None = None,
    *,
    max_pad: MaxPadInput = False,
    safe_pad: SafePadInput = False,
) -> None:
    target_shape = np.array(images[0].resolution) * (spacing or 1.0)
    target_shape /= max(target_shape) / (resize or 1.0)
    for im in images:
        im.resize_to_shape(tuple(target_shape.round().astype(int)))

    if max_pad:
        for im in images:
            im.pad_to_max()

    if safe_pad:
        for im in images:
            im.pad_safe_rotation()


def __start_napari(
    images: Sequence[Image],
    method: RegistrationMethod3D,
    mode: Literal["transform", "register"],
    spacing: tuple[float, float, float] | None,
    upsample_factor: UpsampleFactorInput = 1,
    rotation_axis: RotationAxisRecovery = "z",
    *,
    debug: bool = False,
):
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


if __name__ == "__main__":
    cli()
