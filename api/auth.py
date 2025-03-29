import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

load_dotenv()

TOKEN_URL = os.getenv('TOKEN_URL')
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl=TOKEN_URL)

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM', 'HS256')


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
            headers={'WWW-Authenticate': 'Bearer'},
        ) from exc
