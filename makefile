## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python
PROJ_MODULES=	git python-doc
ADD_CLEAN_ALL =	$(wildcard *.log) datasets

include ./zenbuild/main.mk

.PHONY:		testtorchconf
testtorchconf:
		make PY_SRC_TEST_PAT=test_torchconf.py test

.PHONY:		testsplitkey
testsplitkey:
		make PY_SRC_TEST_PAT=test_splitkey.py test

.PHONY:		testvectorize
testvectorize:
		make PY_SRC_TEST_PAT=test_vec.py test

.PHONY:		testsparse
testsparse:
		make PY_SRC_TEST_PAT=test_sparse.py test

.PHONY:		testbatchstash
testbatchstash:
		make PY_SRC_TEST_PAT=test_batch_stash.py test

.PHONY:		testbatchdata
testbatchdata:
		make PY_SRC_TEST_PAT=test_batch_data.py test

.PHONY:		testmnistdata
testmnistdata:
		make PY_SRC_TEST_PAT=test_mnist_data.py test

.PHONY:		testmodel
testmodel:
		make PY_SRC_TEST_PAT=test_model.py test

.PHONY:		testfacade
testfacade:
		make PY_SRC_TEST_PAT=test_facade.py test

.PHONY:		testiris
testiris:	clean
		PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
			$(PYTHON_BIN) $(PY_SRC_TEST)/iris/proto.py

.PHONY:		testadult
testadult:	clean
		PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
			$(PYTHON_BIN) $(PY_SRC_TEST)/adult/proto.py

.PHONY:		testmnist
testmnist:	clean
		PYTHONPATH=$(PY_SRC):$(PY_SRC_TEST) \
			$(PYTHON_BIN) $(PY_SRC_TEST)/mnist/proto.py

.PHONY:		testall
testall:	clean test testiris testadult testmnist

.PHONY:		notebook
notebook:
		jupyter notebook
