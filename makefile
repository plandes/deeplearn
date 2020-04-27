## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python

include ./zenbuild/main.mk

.PHONY:		testtorchconf
testtorchconf:
	make PY_SRC_TEST_PKGS=test_torchconf.TestTorchConfig test
