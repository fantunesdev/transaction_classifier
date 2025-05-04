import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

from training.pipelines.subcategory import build_pipeline


class TransactionClassifier(ABC):
    """
    Classe responsável por fazer o treinamento ou a predição
    """

    debug = True
    type = None

    def __init__(self, user_id):
        self.pipeline = build_pipeline()
        self.user_id = user_id
        self.model_dir = os.path.join('training', 'model')
        self.extra_state = {}
        self.actions = {
            'subcategory_predictor': {
                'train': True,
                'feedback': True,
            },
            'description_predictor': {
                'train': False,
                'feedback': True,
            }
        }

    def status(self):
        """Obtém o status de treinamento dos modelos."""
        predictors_info = []
        predictor_names = ['subcategory', 'description']

        for name in predictor_names:
            is_trained = self.is_trained(name)
            date = self.get_file_modification_date(name)
            predictors_info.append(
                {
                    'ai': 'Transaction Classifier',
                    'name': name,
                    'description': f'{name.capitalize()} Predictor',
                    'status': 'Treinado' if is_trained else 'Não treinado',
                    'date': date,
                    'actions': self.actions[f'{name}_predictor']
                }
            )

        return {
            'success': True,
            'message': 'Dados obtidos com sucesso',
            'data': predictors_info,
        }

    def is_trained(self, type):
        """Verifica se um modelo salvo existe.

        :param type: Tipo do modelo ('subcategory' ou 'description').
        :return: True se o modelo existe, False caso contrário.
        """
        filepath = os.path.join(self.model_dir, f'{type}_model_user_{self.user_id}.pkl')
        return os.path.exists(filepath)

    def get_file_modification_date(self, type):
        """Obtém a data de modificação do arquivo do modelo.

        :param type: Tipo do modelo ('subcategory' ou 'description').
        :return: Data de modificação no formato 'YYYY-MM-DD' ou None se o arquivo não existir.
        """
        try:
            filepath = os.path.join(self.model_dir, f'{type}_model_user_{self.user_id}.pkl')
            modification_time = os.path.getmtime(filepath)
            return datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d')
        except FileNotFoundError:
            return None

    def save_model(self):
        """
        Salva o modelo treinado para o usuário atual como um arquivo pickle.
        O modelo inclui o pipeline treinado e o mapeamento de subcategoria para categoria.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, f'{self.type}_model_user_{self.user_id}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({'pipeline': self.pipeline, 'extra_state': self.extra_state}, f)

        if self.debug:
            print(f'Modelo salvo em {filepath}')

    def load_model(self):
        """
        Carrega o modelo treinado de um arquivo, se existir.
        """
        filepath = os.path.join(self.model_dir, f'{self.type}_model_user_{self.user_id}.pkl')

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.pipeline = data['pipeline']
                self.extra_state = data.get('extra_state', {})
            if self.debug:
                print(f'Modelo carregado de {filepath}')
        else:
            if self.debug:
                print(f'Nenhum modelo salvo encontrado em {filepath}')

    @abstractmethod
    def train(self, token: str):
        """Processa dados e treina o modelo para o usuário.

        :param token: Token JWT para autenticação e acesso aos dados do usuário.
        :raises NotImplementedError: Se não implementado na subclasse.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, description: str, category: str = None) -> dict:
        """Prevê categoria e subcategoria para uma descrição.

        :param description: Descrição da transação.
        :param category: (Opcional) Categoria informada pelo usuário.
        :return: Dicionário com IDs previstos de categoria e subcategoria.
        :raises NotImplementedError: Se não implementado na subclasse.
        """
        raise NotImplementedError

    def retrain_from_feedback(self, feedbacks: list[dict], token: str):
        """Re-treina o modelo com base no feedback do usuário.

        Utiliza pesos inteligentes baseados no histórico de correções.

        :param feedbacks: Lista de dicionários contendo o feedback do usuário.
        :param token: Token JWT para autenticação e acesso aos dados do usuário.
        :raises NotImplementedError: Se não implementado na subclasse.
        """
        raise NotImplementedError
