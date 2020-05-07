## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python

include ./zenbuild/main.mk

.PHONY:		testtorchconf
testtorchconf:
	make PY_SRC_TEST_PAT=test_torchconf.py test

.PHONY:		testsplitkey
testsplitkey:
	make PY_SRC_TEST_PAT=test_splitkey.py test
