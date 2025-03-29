import os

import requests

API_URL = os.getenv('URL')

def get_data(endpoint: str, token: str):
    """
    Obtém os dados para treinamento
    """
    url = API_URL + endpoint
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers, timeout=5)
    print('oi')

    if response.status_code == 200:
        return response.json()

    print(f'Erro ao obter dados de {endpoint}. Status code: {response.status_code}')
    return None
