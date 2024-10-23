from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Optional, TypeAlias

import napari
import numpy as np
from click import Choice
from imreg3d.image import Image2D, Image3D
from imreg3d.registration import (
    BaseRegistration,
    ImregDft2DRegistration,
    Keller2DRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    Scikit2DRegistration,
    TranslationFFT2DRegistration,
    TranslationFFT3DRegistration,
)
from loguru import logger
from napari.layers import Image
from typer import Argument, Option, Typer

from ._widget import TransformationWidget

FloatOrNone: TypeAlias = Optional[float]  # noqa: UP007
IntOrNone: TypeAlias = Optional[int]  # noqa: UP007
PathsOrNone: TypeAlias = Optional[list[Path]]  # noqa: UP007
PathOrNone: TypeAlias = Optional[Path]  # noqa: UP007

REGISTRATION_METHODS_2D: Final[dict[str, type[BaseRegistration]]] = {
    "keller": Keller2DRegistration,
    "scikit": Scikit2DRegistration,
    "imregdft": ImregDft2DRegistration,
    "translation": TranslationFFT2DRegistration,
}
REGISTRATION_METHODS_3D: Final[dict[str, type[BaseRegistration]]] = {
    "keller": Keller3DRegistration,
    "translation": TranslationFFT3DRegistration,
    "rotationaxis": RotationAxis3DRegistration,
}

HELP_SHIFT_PERCENTAGE = (
    "Shift input in percentage. The actual shift is calculated based in image shape."
)
HELP_SCALE = (
    "Scale parameters. Use <1 for downscaling, >1 for upscaling. "
    "The scale is applied to both x and y dimensions."
)
HELP_ROTATION = "Rotation angle in degrees."
HELP_RESIZE = "Resize all dimensions of the input image to the given size."
HELP_MAX_PAD = "Pad all axes lengths to match the size of the largest axis."
HELP_SAFE_PAD = "Add padding to image to prevent data loss during rotation."
HELP_UPSAMPLE_FACTOR = (
    "Upsample factor for phase correlation. Higher values might increase precision."
)
HELP_DATASET_NAME = "scikit-image dataset to use."
HELP_PRE_ZOOM_OUT_PERCENTAGE = (
    "Zoom in or out of image before registration. Zooming out can improve the "
    "precision of rotation recovery with high shifts (> 50%). Zoom into images if "
    "the image has a border of zeros that shall be removed before registration. "
    "The image content will be downsampled and there image size stays the same."
)
HELP_ENABLE_FILTER = "Enable filter."
HELP_REGISTRATION_FUNCTION_CHOICE = "Registration function to use for registration."

# Image registration parameter types.
ScikitDatasetName = Annotated[str, Option(help=HELP_DATASET_NAME)]
RegistrationMethod2D = Annotated[
    str,
    Option(
        click_type=Choice(list(REGISTRATION_METHODS_2D.keys())),
        help=HELP_REGISTRATION_FUNCTION_CHOICE,
    ),
]
RegistrationMethod3D = Annotated[
    str,
    Option(
        click_type=Choice(list(REGISTRATION_METHODS_3D.keys())),
        help=HELP_REGISTRATION_FUNCTION_CHOICE,
    ),
]
InterpolationOrderInput = Annotated[int, Option(min=0, max=5)]
RotationInput = Annotated[FloatOrNone, Option(help=HELP_ROTATION)]
ScaleInput = Annotated[FloatOrNone, Option(min=0, help=HELP_SCALE)]
UpsampleFactorInput = Annotated[int, Option(min=1, help=HELP_UPSAMPLE_FACTOR)]
DimensionInput = Annotated[int, Option(min=2, max=3, help="Output dimension")]
ResizeInput = Annotated[IntOrNone, Option(min=1, help=HELP_RESIZE)]
ShiftPercentageInput = Annotated[
    float, Option(min=-100, max=100, help=HELP_SHIFT_PERCENTAGE)
]
PreZoomInput = Annotated[FloatOrNone, Option(min=0, help=HELP_PRE_ZOOM_OUT_PERCENTAGE)]
PreBandpassFilter = Annotated[bool, Option(help=HELP_ENABLE_FILTER)]
PreWindowFilter = Annotated[bool, Option(help=HELP_ENABLE_FILTER)]
SafePadInput = Annotated[bool, Option(help=HELP_SAFE_PAD)]
MaxPadInput = Annotated[bool, Option(help=HELP_MAX_PAD)]

# Pipeline parameter types.
# Arguments
ImagePathsOptional = Annotated[PathsOrNone, Argument(exists=True, readable=True)]
ImagePathOptional = Annotated[PathOrNone, Argument(exists=True, readable=True)]
ImagePathsTwo = Annotated[tuple[Path, Path], Argument(exists=True, readable=True)]

# Options
DatasetNames = Annotated[list[str], Option()]
DatasetName = Annotated[str, Option()]
OutputDir = Annotated[Path, Option(dir_okay=True, writable=True)]
ResultsFile = Annotated[Path, Option(file_okay=True, writable=True)]
ScalesList = Annotated[list[float], Option(min=0.0, max=2.0)]
RotationsList = Annotated[list[float], Option()]
ScikitDatasetNamesList = Annotated[list[str], Option()]
ShiftsXList = Annotated[list[int], Option()]
ShiftsYList = Annotated[list[int], Option()]
PreZoomOutPercentageList = Annotated[list[int], Option(min=1, max=100)]
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

    if spacing is not None:
        logger.debug(f"Spacing '{spacing}' will be applied both input images")
        corrected_shape = np.array(images[0].resolution) * spacing

        if resize is not None:
            # If the image will be resized to a fixed size, we can
            # already apply the target size here. Otherwise, we would
            # potentially have a huge size here and make it smaller afterwards.
            corrected_shape = corrected_shape / (max(corrected_shape) / resize)

        # We explicitly convert to built-in integers to prevent showing
        # 'np.int64(...)' within debug logs.
        target_shape = tuple(int(x) for x in corrected_shape.round().astype(int))
        logger.debug(f"Target shape with applied spacing correction: {target_shape}")

        for im in images:
            im.resize_to_shape(target_shape)

    if max_pad:
        for im in images:
            im.pad_to_max()

    if resize is not None:
        for im in images:
            im.resize_to_shape(resize)

    if safe_pad:
        for im in images:
            im.pad_safe_rotation()

    __start_napari(
        list(reversed(images)),
        method,
        "register",
        upsample_factor,
        rotation_axis,
        debug=debug,
    )


@cli.command()
def transform(
    image_path: Path,
    method: RegistrationMethod3D = "keller",
    resize: ResizeInput = 64,
    upsample_factor: UpsampleFactorInput = 1,
    rotation_axis: RotationAxisRecovery = "z",
    *,
    safe_pad: SafePadInput = False,
    debug: bool = False,
):
    """Manually transform and auto-register the original image and a copy of it.

    This mode can be used to test any 3D registration method by
    transformed to be registered with the fixed image.
    """
    image = Image3D.from_path(image_path).resize_to_shape(resize)

    if safe_pad:
        image.pad_safe_rotation().resize_to_shape(resize)

    __start_napari(
        [image], method, "transform", upsample_factor, rotation_axis, debug=debug
    )


def __start_napari(
    images: Sequence[Image3D | Image2D],
    method: RegistrationMethod3D,
    mode: Literal["transform", "register"],
    upsample_factor: UpsampleFactorInput = 1,
    rotation_axis: RotationAxisRecovery = "z",
    *,
    debug: bool = False,
):
    registration = REGISTRATION_METHODS_3D[method](
        upsample_factor=upsample_factor,
        rotation_normalization=False,
        axis=rotation_axis,
        debug=debug,
    )

    v = napari.Viewer()
    # TODO: Make first image invisible.
    for im in images:
        v.add_layer(Image(im.data, name=im.name))

    v.dims.ndisplay = 3
    v.window.add_dock_widget(
        TransformationWidget(viewer=v, registration=registration, mode=mode)
    )
    napari.run()


if __name__ == "__main__":
    cli()
