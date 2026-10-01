PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: install test synthetic docker-build docker-test docker-synthetic clean

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: $(VENV)/bin/activate
	$(PY) -m pip install -e '.[test,bop]'

test: install
	$(PY) -m pytest -q

synthetic: install
	$(PY) -m zutik_geometry synthetic

docker-build:
	docker build -t zutik-geometry:local .

docker-test: docker-build
	docker run --rm --entrypoint python zutik-geometry:local -m pytest -q

docker-synthetic: docker-build
	docker run --rm zutik-geometry:local synthetic

clean:
	rm -rf $(VENV)

