#!/usr/bin/env sh
set -eu
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --disable-pip-version-check --no-input -r requirements.txt
.venv/bin/python -m streamlit run app.py

