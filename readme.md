# About this project

To learn python, machine learning and adjacent topics.

1. Select ML task
2. Write data retrieving code
3. Solve
4. Improve
5. Repeat

## 1. setup

```bat
rem upgrade pip
python -m pip install --upgrade pip
rem create environment
python -m venv .venv
python -m venv --upgrade .venv
pip freeze > requirements.txt
rem activate env
.venv\Scripts\activate
rem update from requirements
pip install -r requirements.txt
```

## 2. [titanic](https://www.kaggle.com/c/titanic)