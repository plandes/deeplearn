## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=		python
PROJ_MODULES=		git python-resources python-doc python-doc-deploy
# mnist -> torchvision makes numpy warnings; warnings.filterwarnings doesn't work
# https://discuss.pytorch.org/t/userwarning-the-given-numpy-array-is-not-writeable/78748/13
PYTHON_TEST_ARGS=	-W ignore::UserWarning 
ADD_CLEAN_ALL +=	$(wildcard *.log) datasets
CLEAN_DEPS +=		pycleancache

#PY_SRC_TEST_PAT ?=	'test_mnist_data*'

include ./zenbuild/main.mk

.PHONY:			testiris
testiris:		clean
			PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
				$(PYTHON_BIN) $(PY_SRC_TEST)/iris/proto.py

.PHONY:			testadult
testadult:		clean
			PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
				$(PYTHON_BIN) $(PY_SRC_TEST)/adult/proto.py

.PHONY:			testmnist
testmnist:		clean
			PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
				$(PYTHON_BIN) $(PY_SRC_TEST)/mnist/proto.py

.PHONY:			testall
testall:		clean test testiris testadult testmnist
