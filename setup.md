# Environment setup instructions

1. Install python

2. Create virtual environment for this project

      ```python -m venv .venv```

      ```python -m venv --upgrade .venv```

      ```python -m pip install --upgrade pip```

3. Activate virtual environment

      ```.venv\Scripts\activate```

4. Either install saved requirements

      ```pip install -r requirements.txt```

5. Or freeze current packages

      ```pip freeze > requirements.txt```

6. List outdated packages

      ```pip list -o```

7. Upgrade package

      ```pip install -U package_name```
