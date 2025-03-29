import os

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError

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
    except ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Token expirado") from e
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Token inválido") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

def get_token_from_header(authorization: str = Header(None)):
    """
    Extrai o token do header.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Token não fornecido")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Token mal formado")
    return authorization[7:]
