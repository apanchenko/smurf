@echo off

echo create environment..
call python -m venv .venv
call python -m venv --upgrade .venv
rem pip freeze > requirements.txt

echo activate env..
call .venv\Scripts\activate

echo upgrade pip..
call python -m pip install --upgrade pip

echo update from requirements..
call pip install -r requirements.txt