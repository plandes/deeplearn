#@meta {author: "Paul Landes"}
#@meta {desc: "deep learning library build and deployment", date: "2025-06-14"}


## Build system
#
PROJ_TYPE=		python
PROJ_MODULES=		python/doc python/deploy
ADD_CLEAN_ALL +=	$(wildcard *.log) datasets


## Includes
#
include ./zenbuild/main.mk


## Targets
#
.PHONY:			testmodel
testmodel:
			@echo "removing any old test results..."
			@make --no-print-directory clean $(PY_PYPROJECT_FILE)
			@echo "testing model $(MODEL)"
			@PYTHONPATH=$(PY_TEST_DIR) make \
				--no-print-directory pyharn \
				PY_HARNESS_BIN='' \
				PY_INVOKE_ARG="-e testcur" \
				ARG=$(PY_TEST_DIR)/$(MODEL)/proto.py

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
