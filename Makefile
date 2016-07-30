init:
    conda env create -f environment.yml
test:
	nosetests tests
