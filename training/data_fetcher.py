import os

import requests

API_URL = os.getenv('URL')

def get_data(endpoint: str, token: str):
    """
    Obtém os dados para treinamento

    :endpoint: str - o endpoint do recurso. Ex: transactions
    :token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
    :API_URL: str - A URL base da api que está setada no .env. Ex: http://localhost:8000/api/
    """
    url = API_URL + endpoint
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers, timeout=5)

    if response.status_code == 200:
        return response.json()

    print(f'Erro ao obter dados de {endpoint}. Status code: {response.status_code}')
    return None
