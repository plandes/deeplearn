# Deep learning framework to provide consistent and reproducible results

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]

General deep learing utility library.  It contains a utilities I used in my own
research that have some intersection with:
* Convolution layer dimensionality calculation
* [PyTorch] convolution factory
* Unrelated (to deep learning) plotting utilities.

Much of the code provides convenience functionality to [PyTorch].  However,
there is functionality that could be used for other deep learning APIs.


## Documentation

See the [full documentation](https://plandes.github.io/deeplearn/index.html).


## Obtaining

The easist way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.deeplearn
```

Binaries are also available on [pypi].


## Reprehensibility

Being able to reproduce the results is one of the major goals of this
framework.  While it provides an API (`TorchConfig`) to set the random seed of
[PyTorch], numpy, and the Python environment, there is still some variance in
some cases in results.

According to this [GitHub issue](https://github.com/pytorch/pytorch/issues/18412):
> This is expected, some of our kernels are not deterministic (specially during backward).
> Might be good to refer to [#1535](https://github.com/pytorch/pytorch/issues/15359).


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 Paul Landes


<!-- links -->
[PyTorch]: https://pytorch.org

[pypi]: https://pypi.org/project/zensols.deeplearn/
[pypi-link]: https://pypi.python.org/pypi/zensols.deeplearn
[pypi-badge]: https://img.shields.io/pypi/v/zensols.deeplearn.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370

[PyTorch]: https://pytorch.org
