import os
import requests
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_KEY = os.getenv('REDIS_KEY')

OAUTH2_TOKEN_URL = os.getenv('OAUTH2_TOKEN_URL')
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')


class OAuth2Client:
    """Classe que gerencia a autenticação da aplicação."""

    def __init__(self):
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self.redis.ping()
            self.cache_available = True
        except redis.exceptions.ConnectionError:
            print('[OAuth2] Redis não disponível. Cache será ignorado.')
            self.redis = None
            self.cache_available = False

    def get_token(self):
        """Obtém o token cacheado ou gera um novo."""
        if self.cache_available:
            try:
                cached_token = self.redis.get(REDIS_KEY)
                if cached_token:
                    return cached_token.decode('utf-8')
            except redis.exceptions.RedisError as e:
                print(f'[OAuth2] Erro ao acessar o Redis: {e}')
        return self._request_token()

    def _request_token(self):
        """Obtém token do MyFinance e, se possível, armazena no Redis."""
        data = {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        }

        try:
            response = requests.post(OAUTH2_TOKEN_URL, data=data, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f'[OAuth2] Erro ao requisitar token: {e}')
            raise

        token_info = response.json()
        access_token = token_info['access_token']
        expires_in = token_info.get('expires_in', 3600)

        if self.cache_available:
            try:
                self.redis.setex(REDIS_KEY, expires_in - 60, access_token)
                print('[OAuth2] Novo token armazenado no Redis.')
            except redis.exceptions.RedisError as e:
                print(f'[OAuth2] Erro ao salvar token no Redis: {e}')

        return access_token
