python -m venv .venv
python -m venv --upgrade .venv
rem pip freeze > requirements.txt
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt