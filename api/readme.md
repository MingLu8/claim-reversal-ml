#Environment Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    fastapi \
    uvicorn \
    pydantic

python -m pip freeze > requirements.txt

#Reinstall later with
python -m pip install -r requirements.txt