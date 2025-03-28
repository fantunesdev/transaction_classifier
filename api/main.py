import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl='http://localhost:8000/api/token/')

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv("ALGORITHM", "HS256")

def verify_token(token: str = Depends(OAUTH2_SCHEME)):
    """
    Verifica se a requisição possui o token de autentificação.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token inválido ou expirado',
            headers={'WWW-Authenticate': 'Bearer'}
        ) from exc

@app.get('/')
def read_root(payload: dict = Depends(verify_token)):
    """
    Exibe uma mensagem de boas vindas para o usuário autenticado.
    """
    print(payload)
    return {'message': 'Seja bem vindo ao Transaction Classifier API!'}
