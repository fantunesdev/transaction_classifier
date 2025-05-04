import logging

from training.data_fetcher import get_data
from training.pipelines.subcategory import build_pipeline
from training.transaction_classifier import TransactionClassifier


class SubcategoryPredictor(TransactionClassifier):
    """
    Classe responsável por fazer o treinamento ou a predição
    """
    type = 'subcategory'
    subcategories = []

    def train(self, token: str):
        """
        Função para processar dados e treinar o modelo para o usuário

        :param token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
        """
        categories = get_data('categories', token)
        subcategories = get_data('subcategories', token)
        transactions = get_data('transactions', token)

        if not categories:
            raise ValueError(
                'Não foi possível obter as categorias para treinar o modelo. Verifique se o token é válido'
            )
        if not subcategories:
            raise ValueError(
                'Não foi possível obter as subcategorias para treinar o modelo. Verifique se o token é válido'
            )
        if not transactions:
            raise ValueError(
                'Não foi possível obter os lançamentos para treinar o modelo. Verifique se o token é válido'
            )

        self.subcategories = subcategories
        self.extra_state = {
            subcategory['id']: subcategory['category'] for subcategory in subcategories
        }

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
        predicted_category_id = self.extra_state.get(predicted_subcategory_id)

        return {'subcategory_id': predicted_subcategory_id, 'category_id': predicted_category_id}

    def retrain_from_feedback(self, feedbacks: list, token: str):
        """
        Re-treina o modelo com base nas correções feitas pelo usuário,
        usando pesos inteligentes baseados no histórico de correções.
        """
        self.load_model()

        categories = get_data('categories', token)
        category_id_to_description = {category['id']: category['description'] for category in categories}

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
                    example = {'description': description, 'category': category_description}

                    logging.info('Treinando exemplo %s com peso %d', description, weight)
                    for _ in range(weight):
                        self.pipeline.learn_one(example, corrected_subcategory)
            else:
                logging.warning('Dados incompletos no feedback %d: ignorado', feedback['id'])

        self.save_model()
        return {
            'success': True,
            'message': f'Modelo treinado com sistema de pesos inteligente para o usuário {self.user_id}!',
        }
