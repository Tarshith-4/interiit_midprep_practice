@echo off
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-input -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m streamlit run app.py

