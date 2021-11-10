.PHONY: activate requirements

SHELL := /bin/bash

activate:
	source .venv/bin/activate

requirements:
	pip install -r requirements.txt