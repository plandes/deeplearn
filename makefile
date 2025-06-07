## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE=		python
PROJ_MODULES=		python/doc
ADD_CLEAN_ALL +=	$(wildcard *.log) datasets


## Includes
#
include ./zenbuild/main.mk


## Targets
#
.PHONY:			testmodel
testmodel:
			make clean $(PY_PYPROJECT_FILE)
			@echo "testing model $(MODEL)"
			@PYTHONPATH=$(PY_TEST_DIR) $(PY_PX_BIN) run \
				--environment testcur python \
				$(PY_TEST_DIR)/$(MODEL)/proto.py

.PHONY:			testiris
testiris:
			make MODEL=iris testmodel

.PHONY:			testadult
testadult:
			make MODEL=adult testmodel

.PHONY:			testmnist
testmnist:
			make MODEL=mnist testmodel

.PHONY:			testall
testall:		test testiris testadult testmnist
