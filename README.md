# napari-manual-transforms

[![License](https://img.shields.io/pypi/l/napari-manual-transforms.svg?color=green)](https://github.com/tlambert03/napari-manual-transforms/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-manual-transforms.svg?color=green)](https://pypi.org/project/napari-manual-transforms)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-manual-transforms.svg?color=green)](https://python.org)
[![tests](https://github.com/tlambert03/napari-manual-transforms/workflows/tests/badge.svg)](https://github.com/tlambert03/napari-manual-transforms/actions)
[![codecov](https://codecov.io/gh/tlambert03/napari-manual-transforms/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/napari-manual-transforms)

Interface to manually edit layer affine transforms.

- express rotations as quaternion, euler angle, or axis + angle.
- allows rotation around arbitrary origin
- currently, focusing on rigid rotations
- Alt-Drag to rotate a layer independently of the rest.
- image resampling (i.e. "apply" the transformation to create new dataset that can be saved)

![Plugin Preview](/preview.jpeg)

caveats:

- only works on 3D Image layers for now, open a feature request for other dims/layers.
- will likely result in "Non-orthogonal slicing is being requested" warnings in 2D view.

## Try it out

```python
import napari

v = napari.Viewer()
v.dims.ndisplay = 3
v.open_sample('napari', 'cells3d')
v.window.add_plugin_dock_widget('napari-manual-transforms')

napari.run()
```

---

## Installation

You can install `napari-manual-transforms` via [pip]:

```sh
pip install napari-manual-transforms
```

To install latest development version :

```sh
pip install git+https://github.com/tlambert03/napari-manual-transforms.git
```

## License

Distributed under the terms of the [BSD-3] license,
"napari-manual-transforms" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/tlambert03/napari-manual-transforms/issues
[pip]: https://pypi.org/project/pip/
