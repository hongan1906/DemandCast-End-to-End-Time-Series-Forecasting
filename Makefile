.PHONY: install test data all
install:
	pip install -e ".[all]"
test:
	pytest -q
data:
	python -m demandcast data
all:
	python -m demandcast all
