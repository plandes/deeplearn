# Zensols Deep Learning Framework

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]

This deep learning library was designed to provide consistent and reproducible
results (see the [full documentation]).

Features:
* Easy to configure and framework to allow for programmatic [debugging] of
  neural networks.
* [Reproducibility] of results
  * All [random seed state] is persisted in the trained model files.
  * Persisting of keys and key order across train, validation and test sets.
* Analysis of results with complete metrics available.
* A [vectorization] framework that allows for pickling tensors.
* Additional layers
  * Convolution layer dimensionality calculation.
  * Recurrent layers that abstracts RNN, GRU and LSTM.
  * Conditional random field layer.
* Pandas interface to easily create and vectorize features.
* Multi-process for time consuming CPU feature [vectorization] requiring little
  to no coding.
* Resource and tensor deallocation with memory management.
* Plotting utilities.

Much of the code provides convenience functionality to [PyTorch].  However,
there is functionality that could be used for other deep learning APIs.


## Documentation

See the [full documentation](https://plandes.github.io/deeplearn/index.html).


## Workflow

This package provides a workflow for processing features, training and then
testing a model.  A high level outline of this process follows:
1. Container objects are used to represent and access data as features.
1. Instances of *data points* wrap the container objects.
1. Vectorize the features of each data point in to tensors.
1. Store the vectorized tensor features to disk so they can be retrieved
   quickly and frequently.
1. At train time, load the vectorized features in to memory and train.
1. Test the model and store the results to disk.

To jump right in, see the [examples](#examples) section.  However, it is better
to peruse the in depth explanation with the [Iris example] code follows:
* The initial [data processing](doc/preprocess.md), which includes data
  representation to batch creation.
* Creating and configuring the [model](doc/model.md).
* Using a [facade](doc/facade.md) to train, validate and test the model.
* Analysis of [results](doc/results.md), including training/validation loss
  graphs and performance metrics.


## Examples

The [Iris example] (also see the [Iris example configuration]) is the most
basic example of how to use this framework.  This example is detailed in the
[workflow](#workflow) documentation in detail.

There are also examples in the form of [Juypter] notebooks as well, which
include the:
* [Iris notebook] data set, which is a small data set of flower dimensions as a
  three label classification,
* [MNIST notebook] for the handwritten digit data set,
* [debugging notebook].


## Obtaining

The easist way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.deeplearn
```

Binaries are also available on [pypi].


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.deeplearn/
[pypi-link]: https://pypi.python.org/pypi/zensols.deeplearn
[pypi-badge]: https://img.shields.io/pypi/v/zensols.deeplearn.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370

[PyTorch]: https://pytorch.org
[Juypter]: https://jupyter.org

[full documentation]: https://plandes.github.io/deeplearn/index.html
[Iris notebook]: https://github.com/plandes/deeplearn/tree/master/notebook/iris.ipynb
[MNIST notebook]: https://github.com/plandes/deeplearn/tree/master/notebook/mnist.ipynb
[debugging notebook]: https://github.com/plandes/deeplearn/tree/master/notebook/debug.ipynb

[reproducibility]: doc/results.html#reproducibility
[debugging]: doc/facade.html#debugging-the-model
[random seed state]: api/zensols.deeplearn.html#zensols.deeplearn.torchconfig.TorchConfig.set_random_seed
[vectorization]: doc/preprocess.html#vectorizers
[Iris example]: https://github.com/plandes/deeplearn/blob/master/test/python/iris/model.py
[Iris example configuration]: https://github.com/plandes/deeplearn/blob/master/test-resources/iris
