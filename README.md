# General deep learning tools with PyTorch

[![Travis CI Build Status][travis-badge]][travis-link]
[![PyPI][pypi-badge]][pypi-link]

General deep learing utility library.  It contains a utilities I used in my own
research that have some intersection with:
* Convolution layer dimensionality calculation
* [PyTorch] convolution factory
* Unrelated (to deep learning) plotting utilities.

Much of the code provides convenience functionality to [PyTorch].  However,
there is functionality that could be used for other deep learning APIs.


## Obtaining

The easist way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.dltools
```

Binaries are also available on [pypi].


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

Copyright (c) 2019 Paul Landes

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


<!-- links -->
[PyTorch]: https://pytorch.org

[travis-link]: https://travis-ci.org/plandes/dltools
[travis-badge]: https://travis-ci.org/plandes/dltools.svg?branch=master
[pypi]: https://pypi.org/project/zensols.dltools/
[pypi-link]: https://pypi.python.org/pypi/zensols.dltools
[pypi-badge]: https://img.shields.io/pypi/v/zensols.dltools.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370
