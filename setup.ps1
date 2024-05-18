pyenv install 3.9.5
pyenv local 3.9.5
pip install virtualenv
virtualenv INM706_cw_env
INM706_cw_env/Scripts/activate
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118