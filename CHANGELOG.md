## [Unreleased]
### Added
- run a tournament
### Changed
- command line arguments:
    - e.g. add ```--passive``` mode
    - for further changes, refer to ```--help```
- negative score for killing oneself
### Fixed
- prevent PyGame window from rendering for render modes other than ```"human"```
- action format: environment only accepts action inputs from proper action space (i.e. numbers)
- enable custom avatars

## [v1.0.1]
### Fixed
- ```Coins``` observation was missing in ```README.md``` (credit to student for pointing out)