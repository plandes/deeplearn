# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [0.0.6] - 2020-05-11
### Changed
- Upgrade to [zensols.util] 1.2.1 and torch 1.4.0.

### Added
- Type conversion from torch to numpy and sparse types.
- Implementation of sparse matrix encoding and decoding to compensate for
  PyTorch bug with hanging child processing in `multiprocessing` API.
- Key split stash implementation for data set splits.
- Vectorization framework
- Model training and testing framework.
- Notebook example usage.
- Unit tests, for which, now give very good coverage.

### Removed
- Travis build: too time consuming to make CUDA, python and xenial etc. all
  play together nicely.


## [0.0.5] - 2020-05-04

### Added
- Create sparse tensors.


## [0.0.4] - 2020-04-27

### Change
- Travis tests.
- Upgraded to Python 3.7.


## [0.0.3] - 2019-12-14
Data classes are now used so Python 3.7 is now a requirement.

### Added
- Arbitrarily deep linear layer using constant or decay parameter counts.
- Conv2d interface to PyTorch.

### Changed
- Better handling of tensor to memory mapping and device handling in general.


### Removed
- Moved `test.py` from this repository to [zensols.actioncli].


## [0.0.2] - 2019-07-07
### Added
- Tests.


## [0.0.1] - 2019-07-07
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/deeplearn/compare/v0.0.6...HEAD
[0.0.5]: https://github.com/plandes/deeplearn/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/plandes/deeplearn/compare/v0.0.3...v0.0.6
[0.0.3]: https://github.com/plandes/deeplearn/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/plandes/deeplearn/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/deeplearn/compare/v0.0.0...v0.0.1

[zensols.actioncli]: https://github.com/plandes/actioncli
[zensols.util]: https://github.com/plandes/util
