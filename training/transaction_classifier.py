import os
import pickle

from training.pipeline import build_pipeline
from training.data_fetcher import get_data


class TransactionClassifier:
    """
    Classe responsável por fazer o treinamento ou a predição
    """
    debug = True
    model_dir = 'training/model/'

    def __init__(self, user_id):
        self.pipeline = build_pipeline()
        self.subcategories = []
        self.subcategory_id_to_cat_id = {}
        self.user_id = user_id

    def train(self, token):
        """
        Função para processar dados e treinar o modelo para o usuário

        :param token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
        """
        categories = get_data('categories', token)
        subcategories = get_data('subcategories', token)
        transactions = get_data('transactions', token)

        if not categories or not subcategories:
            raise ValueError('Não foi possível obter categorias ou subcategorias.')

        if not transactions:
            raise ValueError('Não há lançamentos suficientes para treinar o modelo.')

        self.subcategories = subcategories
        self.subcategory_id_to_cat_id = {
            sub['id']: sub['category'] for sub in subcategories
        }

        category_id_to_description = {
            cat['id']: cat['description'] for cat in categories
        }

        # Primeiro: treino com base nas descrições das subcategorias (ajuda como referência)
        for subcategory in subcategories:
            example = {
                'description': subcategory['description'],
                'category': category_id_to_description.get(subcategory['category'], '')
            }
            target = subcategory['id']
            self.pipeline.learn_one(example, target)

        # Segundo: treino com base nos lançamentos reais cadastrados na aplicação
        for transaction in transactions:
            example = {
                'description': transaction['description'],
                'category': category_id_to_description.get(transaction['category'], '')
            }
            target = transaction['subcategory']
            self.pipeline.learn_one(example, target)

        self.save_model()

        return {
            'success': True,
            'message': f'Modelo do usuário {self.user_id} treinado com sucesso! ' \
                       f'com {len(subcategories)} subcategorias ' \
                       f'e {len(transactions)} lançamentos.'
        }

    def predict(self, description: str, category: str = ''):
        """
        Faz uma previsão de categoria e subcategoria para uma descrição dada.

        :param description: Descrição da transação.
        :param category: (Opcional) Categoria informada pelo usuário.
        :return: dicionário com IDs previstos de categoria e subcategoria.
        """
        self.load_model()

        example = {
            'description': description,
            'category': category
        }
        predicted_subcategory_id = self.pipeline.predict_one(example)
        predicted_category_id = self.subcategory_id_to_cat_id.get(predicted_subcategory_id)

        return {
            'subcategory_id': predicted_subcategory_id,
            'category_id': predicted_category_id
        }

    def save_model(self):
        """
        Salva o modelo treinado em um arquivo.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, f'model_user_{self.user_id}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'subcategory_map': self.subcategory_id_to_cat_id
            }, f)

        if self.debug:
            print(f'Modelo salvo em {filepath}')

    def load_model(self):
        """
        Carrega o modelo treinado de um arquivo, se existir.
        """
        filepath = os.path.join(self.model_dir, f'model_user_{self.user_id}.pkl')

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.pipeline = data['pipeline']
                self.subcategory_id_to_cat_id = data['subcategory_map']
            if self.debug:
                print(f'Modelo carregado de {filepath}')
        else:
            if self.debug:
                print(f'Nenhum modelo salvo encontrado em {filepath}')
