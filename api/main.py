from fastapi import FastAPI, Depends

from api.auth import verify_token

app = FastAPI()

@app.get('/')
def read_root(payload: dict = Depends(verify_token)):
    """
    Exibe uma mensagem de boas vindas para o usuário autenticado.
    """
    return {'message': 'Seja bem vindo ao Transaction Classifier API!'}
