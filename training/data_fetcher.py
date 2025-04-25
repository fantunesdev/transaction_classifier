import os
from typing import Optional

import requests

from api.oauth2_client import OAuth2Client

SERVER_URL = os.getenv('SERVER_URL')


def get_data(resource: str, token: Optional[str] = None):
    """
    Obtém os dados para treinamento.

    :param resource: str - resource do recurso (ex: 'transactions/')
    :param token: str (opcional) - token JWT manual (ex: vindo do Insomnia).
    :return: dict|None
    """
    if not token:
        print('[Data Fetcher] Nenhum token fornecido. Gerando token via OAuth2...')
        oauth_client = OAuth2Client()
        token = oauth_client.get_token()

    url = f'{SERVER_URL}/api/{resource}'
    headers = {'Authorization': f'Bearer {token}'}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f'[Data Fetcher] Erro ao acessar {url}: {e}')
        return None
