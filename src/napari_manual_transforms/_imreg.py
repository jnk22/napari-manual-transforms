"""CLI module for image registration using 'ndimreg'."""

from __future__ import annotations

from typing import Final, Literal

from cyclopts import App
from cyclopts.types import File, Json  # noqa: TC002

RegistrationMethod3D = Literal["keller", "rotationaxis", "translation"]
RotationAxis3D = Literal["x", "y", "z"]
LogLevel = Literal["info", "warning", "error", "trace", "debug", "success", "critical"]

EMPTY_OPTIONS: Final[dict] = {}

app = App()


@app.default
def register(  # noqa: PLR0913
    image_paths: list[File],
    *,
    method: RegistrationMethod3D = "keller",
    axis: RotationAxis3D = "z",
    options: Json | dict = EMPTY_OPTIONS,
    spacing: tuple[float, float, float] | None = None,
    normalize: bool = True,
    max_pad: bool = False,
    safe_pad: bool = False,
    resize: int | None = None,
    debug: bool = False,
    log_level: LogLevel = "info",
) -> None:
    """Manually transform the 'moving' image and auto-register with 'fixed' image.

    Arguments
    ---------
    image_paths
        One or two image. The first input is the 'fixed' image. The
        second image is the 'moving' image. If only one image is
        provided, its copy is used as moving image.
    """
    import os
    import sys

    import napari
    from loguru import logger
    from napari.layers import Image as NapariImage
    from ndimreg.image import Paths3DImageLoader
    from ndimreg.registration import (
        Keller3DRegistration,
        RotationAxis3DRegistration,
        TranslationFFT3DRegistration,
    )
    from ndimreg.utils.image import prepare_benchmark_image

    from ._widget import TransformationWidget

    logger.remove()
    logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", log_level.upper()))

    if not 1 <= len(image_paths) <= 2:  # noqa: PLR2004
        print("Only one or two images allowed")
        sys.exit(1)

    images = [
        prepare_benchmark_image(
            im,
            spacing=spacing,
            normalize=normalize,
            max_pad=max_pad,
            safe_pad=safe_pad,
            resize=resize,
        )
        for im in Paths3DImageLoader(image_paths)
    ]

    # If only one image is provided, we use this image as 'fixed' and
    # 'moving' image instead.
    if len(images) == 1:
        images = [images[0], images[0].copy()]

    registration_methods = {
        "keller": Keller3DRegistration,
        "translation": TranslationFFT3DRegistration,
        "rotationaxis": RotationAxis3DRegistration,
    }

    registration = registration_methods[method](axis=axis, debug=debug, **options)

    v = napari.Viewer()
    for im in images:
        v.add_layer(NapariImage(im.data, name=im.name))

    v.dims.ndisplay = 3
    v.window.add_dock_widget(
        TransformationWidget(viewer=v, registration=registration, spacing=spacing)
    )
    napari.run()
