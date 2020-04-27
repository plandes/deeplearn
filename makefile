## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python

include ./zenbuild/main.mk

.PHONY:		testcuda
testcuda:
	make PY_SRC_TEST_PKGS=test_cuda.TestCuda test
