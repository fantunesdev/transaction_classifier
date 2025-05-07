import os

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

from training.data_fetcher import get_data

load_dotenv()

SERVER_URL = os.getenv('SERVER_URL')
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl=f'{SERVER_URL}/api/token')


def verify_token(token: str = Depends(OAUTH2_SCHEME)):
    """
    Verifica se a requisição possui um token válido (consultando o backend Django)
    e retorna o payload, sem verificar a assinatura localmente.

    :param token: str - Token JWT.
    :return: dict - Payload decodificado do token.
    """
    token_is_valid = get_data('validate-token', token=token)['valid']

    if token_is_valid:
        payload = jwt.decode(token, key='', options={"verify_signature": False})
        return payload

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token inválido ou expirado')


def get_token_from_header(authorization: str = Header(None)):
    """
    Extrai o token do header.

    :authorization: str - o Header da requisição.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail='Token não fornecido')
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=400, detail='Token mal formado')
    return authorization[7:]
