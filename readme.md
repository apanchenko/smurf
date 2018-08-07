# setup
create environment:
  py -m venv ./.venv
  py -m venv --upgrade venv
  pip freeze > requirements.txt
activate env:
  "venv/Scripts/activate"