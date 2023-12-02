## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE=		python
PROJ_MODULES=		git python-resources python-doc python-doc-deploy
ADD_CLEAN_ALL +=	$(wildcard *.log) datasets
CLEAN_DEPS +=		pycleancache


## Includes
#
include ./zenbuild/main.mk


## Targets
#
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
