Automatically generated (and slightly modified) using::

  git log --pretty=format:"- %ad, %aN%x09%s" --date=short v1.4.3..HEAD > changelog/CHANGELOG_v143_v144.rst

  Release 1.4.3 -> 1.4.4
  =======================================

- 2019-08-24, jgliss	Fixed some style errors and weakened PIL import (only relevant for special case of Heidelberg camera)
- 2019-08-12, jgliss	Autocorrected mutable input args in methods
- 2019-08-12, jgliss	(REFACTOR) Renamed module CameraBaseInfo.py to camera_base_info.py
- 2019-08-11, jgliss	Fixed failing flake8 tests
- 2019-08-11, jgliss	Removed invocation of example scripts in .travis.yml (cf. #27 and #29)
- 2019-08-10, Johann Jacobsohn	remove unneeded and unused requirements.txt
- 2019-08-06, Johann Jacobsohn	fixup: import mpl_toolkits.mplot3d and ignore
- 2019-08-06, Johann Jacobsohn	add flake8 to travis ci
- 2019-08-06, Johann Jacobsohn	remove tox config
- 2019-08-10, jgliss	Removed python 3.5 from travis.yml due broken conda installation (basemap / matplotlib version conflict)
- 2019-02-06, Johann Jacobsohn	add regexp filename parser, refactor filename parsing code for testability
- 2019-08-06, Johann Jacobsohn	revert b523bbf scripts: doas calib: significantly raise tolerances in tests to make them pass, needs to be investigated
- 2019-08-05, Johann Jacobsohn	replace pandas from_csv with read_csv
- 2019-06-01, jgliss	Made method _load_cam_info in inout.py more robust
- 2019-06-01, jgliss	Added __repr__ in utils.Filter class
- 2019-06-01, jgliss	BUGFIX: Filter.__str__ returned None, now a str
- 2019-06-01, jgliss	Moved lib init methods from __init__.py into new module _init_utils.py
- 2019-05-26, Johann Jacobsohn	scripts: doas calib: significantly raise tolerances in tests to make them pass, needs to be investigated
- 2019-05-26, Johann Jacobsohn	reset requirements for tox build
- 2019-05-26, Johann Jacobsohn	scripts: return proper exit code on failed tests
- 2019-05-26, Johann Jacobsohn	avoid attempts to discover tests in 3rd party code
- 2019-01-27, Johann Jacobsohn	improve wording
- 2019-01-27, Johann Jacobsohn	satisfy lint
- 2019-01-18, Johann Jacobsohn	Revert "WIP to pass tox scripts - needs to be reverted and fixed!"
- 2019-05-24, jgliss	Added methods in __init__.py to update loggers
- 2019-05-24, jgliss	Changed some warnings from logger to print_log; removed some unnecessary checks and some try / except blocks, that would cause silent or unwanted errors
- 2019-05-24, jgliss	Imglists throw IndexError now in methods _first_file and _last_file; changed some output from pyplis.logger instance to pyplis.print_log
- 2019-05-24, jgliss	Went through all print statements (logger.info(..)) and, where applicable, updated to logger.warning or print_log.info, respectively
- 2019-05-24, jgliss	API refactor: removed all print() and warn() statements and replaced with logger.info() or logger.warning(), respectively (pyplis loggers are now imported in each module)
- 2019-05-24, jgliss	Changed default input files=[] to files=None (to avoid silent errors since list is mutable) in all image list classes and updated how this is handled in __init__; minor changes in docs
- 2019-05-24, jgliss	Updated how PIL availability is checked / handled in custom_image_import.py, now using new PILAVAILABLE flag set in __init__.py
- 2019-05-24, jgliss	Added 2 default loggers "logger" and "print_log" in __init__.py and cleaned up imports, etc.
- 2019-05-23, jgliss	New test module test_setupclasses.py, so far only containing one test for class FilterSetup
- 2019-05-23, jgliss	Removed old code and updated default filter access due to updated FilterSetup class (prev. commit)
- 2019-05-23, jgliss	Simplified FilterSetup class and made filter assignment more flexible, now also supporting dict-like assignment and access of filters via __getitem__ and __setitem__ methods
- 2019-05-23, jgliss	Updated __str__ method of Filter class
- 2019-05-23, jgliss	Added DeprecationError in exceptions.py
- 2019-05-23, jgliss	Removed reg_shift_off entry (8, -7) for ECII camera in cam_info.txt
- 2019-05-23, jgliss	Improved docstring of Img class in image.py
- 2019-05-23, jgliss	Fixed bug in data_search_dirs() method in inout.py and bumped version
- 2019-05-22, jgliss	Added first test version of .travis.yml
