import os
import logging
import pickle

from training.data_fetcher import get_data
from training.pipeline import build_pipeline


class TransactionClassifier:
    """
    Classe responsável por fazer o treinamento ou a predição
    """

    debug = True
    model_dir = 'training/model/'

    def __init__(self, user_id):
        self.pipeline = build_pipeline()
        self.subcategories = []
        self.subcategory_id_to_category_id = {}
        self.user_id = user_id

    def train(self, token: str):
        """
        Função para processar dados e treinar o modelo para o usuário

        :param token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
        """
        categories = get_data('categories', token)
        subcategories = get_data('subcategories', token)
        transactions = get_data('transactions', token)

        if not categories:
            raise ValueError('Não foi possível obter as categorias para treinar o modelo. ' \
            'Verifique se o token é válido')
        if not subcategories:
            raise ValueError('Não foi possível obter as subcategorias para treinar o modelo. ' \
            'Verifique se o token é válido')
        if not transactions:
            raise ValueError('Não foi possível obter os lançamentos para treinar o modelo. ' \
            'Verifique se o token é válido')

        self.subcategories = subcategories
        self.subcategory_id_to_category_id = {subcategory['id']: subcategory['category']
                                              for subcategory in subcategories}

        category_id_to_description = {category['id']: category['description'] for category in categories}

        # Primeiro: treino com base nas descrições das subcategorias
        for subcategory in subcategories:
            example = {
                'description': subcategory['description'],
                'category': category_id_to_description.get(subcategory['category'], ''),
            }
            target = subcategory['id']
            self.pipeline.learn_one(example, target)

        # Segundo: treino com base nos lançamentos reais cadastrados na aplicação
        for transaction in transactions:
            example = {
                'description': transaction['description'],
                'category': category_id_to_description.get(transaction['category'], ''),
            }
            target = transaction['subcategory']
            self.pipeline.learn_one(example, target)

        self.save_model()

        return {
            'success': True,
            'message': f'Modelo do usuário {self.user_id} treinado com sucesso! '
            f'com {len(subcategories)} subcategorias '
            f'e {len(transactions)} lançamentos.',
        }

    def predict(self, description: str, category: str = ''):
        """
        Faz uma previsão de categoria e subcategoria para uma descrição dada.

        :param description: Descrição da transação.
        :param category: (Opcional) Categoria informada pelo usuário.
        :return: dicionário com IDs previstos de categoria e subcategoria.
        """
        self.load_model()

        example = {'description': description, 'category': category}
        predicted_subcategory_id = self.pipeline.predict_one(example)
        predicted_category_id = self.subcategory_id_to_category_id.get(predicted_subcategory_id)

        return {'subcategory_id': predicted_subcategory_id, 'category_id': predicted_category_id}

    def save_model(self):
        """
        Salva o modelo treinado para o usuário atual como um arquivo pickle.
        O modelo inclui o pipeline treinado e o mapeamento de subcategoria para categoria.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, f'model_user_{self.user_id}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({'pipeline': self.pipeline, 'subcategory_map': self.subcategory_id_to_category_id}, f)

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
                self.subcategory_id_to_category_id = data['subcategory_map']
            if self.debug:
                print(f'Modelo carregado de {filepath}')
        else:
            if self.debug:
                print(f'Nenhum modelo salvo encontrado em {filepath}')

    def is_trained(self):
        """
        Verifica se já existe um modelo salvo para o usuário.
        """
        filepath = os.path.join(self.model_dir, f'model_user_{self.user_id}.pk')
        return os.path.exists(filepath)

    def retrain_from_feedback(self, feedbacks: list, token: str):
        """
        Re-treina o modelo com base nas correções feitas pelo usuário,
        usando pesos inteligentes baseados no histórico de correções.
        """
        self.load_model()

        categories = get_data('categories', token)
        category_id_to_description = {category['id']: category['description'] for category in categories}

        # Ids dos feedbacks que serão marcados como já utilizados no banco de dados
        feedback_ids = []

        for feedback in feedbacks:
            description = feedback['description']
            predicted_subcategory = feedback.get('predicted_subcategory_id')
            corrected_category = feedback.get('corrected_category_id')
            corrected_subcategory = feedback.get('corrected_subcategory_id')

            # Verifica se é uma correção real
            is_correction = predicted_subcategory != corrected_subcategory

            if description and corrected_category and corrected_subcategory:
                if is_correction:
                    # É uma correção, então o peso é 50 vezes maior
                    weight = 50
                else:
                    # Se não é uma correção, não precisa retreinar.
                    weight = 0

                if weight > 0:
                    category_description = category_id_to_description.get(corrected_category, '')
                    example = {
                        'description': description,
                        'category': category_description
                    }

                    logging.info('Treinando exemplo %s com peso %d', description, weight)
                    for _ in range(weight):
                        self.pipeline.learn_one(example, corrected_subcategory)
                feedback_ids.append(feedback.get('id'))
            else:
                logging.warning('Dados incompletos no feedback %d: ignorado', feedback['id'])

        self.save_model()
        return {
            'success': True,
            'message': f'Modelo treinado com sistema de pesos inteligente para o usuário {self.user_id}!',
            'feedback_ids': feedback_ids
        }
