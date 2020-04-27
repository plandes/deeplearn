# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


### Changed
- Upgrade to [zensols.util] 1.2.1 and torch 1.4.0.

### Removed
- Travis build: too time consuming to make CUDA, python and xenial etc. all
  play together nicely.


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
[Unreleased]: https://github.com/plandes/dltools/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/plandes/dltools/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/plandes/dltools/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/dltools/compare/v0.0.0...v0.0.1

[zensols.actioncli]: https://github.com/plandes/actioncli
[zensols.util]: https://github.com/plandes/util
