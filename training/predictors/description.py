import os
import pickle
import traceback
import unicodedata

from river import compose, feature_extraction, naive_bayes, preprocessing

from training.data_fetcher import get_data
from training.pipelines.description import build_pipeline
from training.transaction_classifier import TransactionClassifier


class DescriptionPredictor(TransactionClassifier):
    """
    Classe responsável pelo treinamento e previsão do modelo de descrição
    """
    type = 'description'

    def __init__(self, user_id):
        super().__init__(user_id)
        self.min_samples = 5  # Mínimo de amostras para treinar um modelo confiável
        self.min_confidence = 0.3  # Confiança mínima para fazer uma previsão
        self.model = naive_bayes.MultinomialNB()
        self.vectorizer = {}  # Dicionário para armazenar vocabulário
        self.preprocessing_enabled = True  # Habilita ou desabilita o pré-processamento

    def ensure_serializable(self, obj):
        """
        Garante que um objeto seja serializável para JSON.
        Converte tipos não serializáveis em tipos serializáveis.
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.ensure_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self.ensure_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return self.ensure_serializable(obj.__dict__)
        else:
            # Fallback para str
            try:
                return str(obj)
            except:
                return "Objeto não serializável"

    def preprocess_text(self, text):
        """
        Pré-processa o texto para melhorar a qualidade do modelo
        """
        if not self.preprocessing_enabled:
            return text

        if not isinstance(text, str):
            print(
                f"AVISO: Tipo de entrada inválido para preprocess_text: {type(text)}")
            if isinstance(text, dict) and 'description' in text:
                text = text['description']
            else:
                return str(text)

        # Converter para minúsculas
        text = text.lower()

        # Remover acentos (forma simples)
        try:

            text = unicodedata.normalize('NFKD', text).encode(
                'ASCII', 'ignore').decode('ASCII')
        except Exception as e:
            print(f"Erro ao remover acentos: {e}")

        return text

    def vectorize_text(self, text):
        """
        Converte o texto em um vetor de características usando contagem de palavras
        """
        if not isinstance(text, str):
            print(
                f"AVISO: Tipo de entrada inválido para vectorize_text: {type(text)}")
            if isinstance(text, dict) and 'description' in text:
                text = text['description']
            else:
                return {}

        text = self.preprocess_text(text)
        words = text.split()
        vector = {}

        for word in words:
            if word not in vector:
                vector[word] = 0
            vector[word] += 1

            # Atualizar o vocabulário global
            if word not in self.vectorizer:
                self.vectorizer[word] = 0
            self.vectorizer[word] += 1

        return vector

    def train(self, token: str):
        try:
            feedbacks = get_data('categorization-feedback', token)

            if not feedbacks:
                raise ValueError('Não foi possível obter feedbacks.')

            self.delete_model()

            # Limpar modelo e vetorizador
            self.model = naive_bayes.MultinomialNB()
            self.vectorizer = {}

            used_feedbacks = 0
            training_data = []

            for feedback in feedbacks:
                # Verificar se a entrada é válida
                if not isinstance(feedback, dict):
                    print(
                        f"AVISO: Formato de feedback inválido: {type(feedback)}")
                    continue

                if 'description' not in feedback or 'corrected_description' not in feedback:
                    print("AVISO: Feedback não contém os campos necessários")
                    continue

                # Ignora feedbacks sem correção de descrição
                if feedback['description'] != feedback['corrected_description']:
                    try:
                        # Extrair texto e vetorizar
                        description = feedback['description']
                        vector = self.vectorize_text(description)
                        target = feedback['corrected_description']

                        # Treinar o modelo
                        self.model.learn_one(vector, target)

                        training_data.append({
                            'description': description,
                            'corrected_description': target
                        })
                        used_feedbacks += 1
                    except Exception as e:
                        print(f"Erro ao processar feedback: {e}")

            if used_feedbacks < self.min_samples:
                print(f"Aviso: Apenas {used_feedbacks} exemplos foram utilizados para treinamento. "
                      f"Recomendamos pelo menos {self.min_samples} para um modelo confiável.")

            # Validação simples
            if used_feedbacks > 0:
                correct = 0
                for item in training_data:
                    vector = self.vectorize_text(item['description'])
                    prediction = self.model.predict_one(vector)
                    if prediction == item['corrected_description']:
                        correct += 1
                accuracy = correct / len(training_data) if training_data else 0
                print(
                    f"Acurácia do modelo nos dados de treinamento: {accuracy:.2f}")

            # Salvar modelo
            self.save_model()

            return {
                'success': True,
                'message': f'Modelo do usuário {self.user_id} treinado com sucesso! '
                f'com {used_feedbacks} feedback(s) utilizados.'
            }

        except Exception as e:
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Erro ao treinar modelo: {str(e)}'
            }

    def predict(self, description: str, category: str = None):
        """
        Faz uma previsão da descrição corrigida para uma descrição dada.
        """
        try:
            print("Carregando modelo...")
            self.load_model()
            print(f"Modelo carregado para {self.user_id}")

            print(f"Fazendo previsão para: {description}")

            # Verificar se o vocabulário da descrição está presente no vetor treinado
            tokens = self.preprocess_text(description).split()
            known_tokens = [token for token in tokens if token in self.vectorizer]

            if not known_tokens:
                print("Aviso: Nenhuma palavra da descrição foi vista no treinamento")
                return {
                    'success': True,
                    'prediction': description,
                    'message': 'Não foi possível prever: a descrição parece inédita para o modelo.'
                }

            # Vetorizar o texto e fazer a previsão
            try:
                vector = self.vectorize_text(description)
                print(f"Texto vetorizado: {len(vector)} características")

                prediction = self.model.predict_one(vector)
                print(f"Previsão bruta: {prediction}")

                # Garantir que a previsão seja serializável
                if prediction is None:
                    prediction_str = None
                else:
                    prediction_str = str(prediction)

                # Calcular confiança
                confidence = 0.0
                if hasattr(self.model, 'predict_proba_one'):
                    probas = self.model.predict_proba_one(vector)
                    print(f"Probabilidades: {probas}")

                    if prediction_str in probas:
                        confidence = probas[prediction_str]
                    elif prediction in probas:
                        confidence = probas[prediction]

                    print(f"Confiança da previsão: {confidence:.2f}")

                    if confidence < self.min_confidence:
                        print("Aviso: Confiança abaixo do limite")
                        return {
                            'success': True,
                            'prediction': None,
                            'message': f'Baixa confiança na previsão para esta descrição (confiança: {confidence:.2f}).'
                        }

            except Exception as predict_error:
                print(f"Erro na previsão: {str(predict_error)}")
                traceback.print_exc()
                return {
                    'success': True,
                    'prediction': None,
                    'message': f'Não foi possível fazer uma previsão para esta descrição: {str(predict_error)}'
                }

            print(f"Previsão realizada: {prediction_str}")

            # Se a previsão for muito próxima da entrada ou None, retornamos None
            if prediction_str is None or self.preprocess_text(prediction_str) == self.preprocess_text(description):
                print("Aviso: Previsão igual à entrada")
                return {
                    'success': True,
                    'prediction': None,
                    'message': 'Nenhuma correção sugerida para esta descrição.'
                }

            # Adiciona informações de debug à resposta
            response = {
                'success': True,
                'prediction': prediction_str,
                'confidence': float(confidence) if isinstance(confidence, (int, float)) else 0.0
            }

            # Garantir que a resposta seja completamente serializável
            response = self.ensure_serializable(response)

            print(f"Retornando resposta: {response}")
            return response

        except Exception as e:
            traceback.print_exc()
            print(f"Erro geral: {str(e)}")
            # Garantir que a resposta de erro seja completamente serializável
            return {
                'success': False,
                'prediction': None,
                'message': f'Erro ao realizar predição: {str(e)}'
            }

    def retrain_from_feedback(self, feedbacks: list, token: str):
        """
        Re-treina o modelo com base nas correções feitas pelo usuário, usando pesos inteligentes baseados no
        histórico de correções.

        :param feedbacks: uma lista de feedbacks.
        :param token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
        """
        try:
            if not feedbacks:
                return {
                    'success': False,
                    'message': 'Nenhum feedback fornecido para re-treinamento.'
                }

            self.load_model()

            correction_counts = {}  # Para contar frequência das correções

            used_feedbacks = 0

            for feedback in feedbacks:
                if not isinstance(feedback, dict):
                    print(f"AVISO: Formato inválido de feedback: {type(feedback)}")
                    continue

                if 'description' not in feedback or 'corrected_description' not in feedback:
                    print("AVISO: Feedback incompleto")
                    continue

                if feedback['description'] != feedback['corrected_description']:
                    try:
                        description = feedback['description']
                        corrected = feedback['corrected_description']
                        vector = self.vectorize_text(description)

                        # Contar correções para aplicar peso
                        correction_counts[corrected] = correction_counts.get(corrected, 0) + 1

                        used_feedbacks += 1
                    except Exception as e:
                        print(f"Erro ao processar feedback: {e}")

            if used_feedbacks == 0:
                return {
                    'success': False,
                    'message': 'Nenhum feedback válido para re-treinamento.'
                }

            # Reaplicar os feedbacks com pesos (ex: feedbacks mais frequentes reforçam mais)
            for feedback in feedbacks:
                if feedback['description'] != feedback['corrected_description']:
                    description = feedback['description']
                    corrected = feedback['corrected_description']
                    vector = self.vectorize_text(description)
                    weight = correction_counts[corrected]
                    for _ in range(weight):
                        self.model.learn_one(vector, corrected)

            self.save_model()

            return {
                'success': True,
                'message': f'{used_feedbacks} feedback(s) aplicado(s) com sucesso no modelo do usuário {self.user_id}.'
            }

        except Exception as e:
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Erro ao re-treinar modelo: {str(e)}'
            }

    def save_model(self):
        """
        Salva o modelo e o vetorizador em um arquivo
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'preprocessing_enabled': self.preprocessing_enabled
        }

        model_path = f"training/model/{self.type}_model_user_{self.user_id}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Modelo salvo em {model_path}")

    def load_model(self):
        """
        Carrega o modelo e o vetorizador de um arquivo
        """
        model_path = f"training/model/{self.type}_model_user_{self.user_id}.pkl"

        if not os.path.exists(model_path):
            print(f"Modelo não encontrado em {model_path}")
            # Criar modelo vazio
            self.model = naive_bayes.MultinomialNB()
            self.vectorizer = {}
            self.preprocessing_enabled = True
            return

        print(f"Carregando modelo de {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data.get('model', naive_bayes.MultinomialNB())
        self.vectorizer = model_data.get('vectorizer', {})
        self.preprocessing_enabled = model_data.get(
            'preprocessing_enabled', True)

        print(f"Modelo carregado de {model_path}")

    def delete_model(self):
        """
        Remove o arquivo do modelo
        """
        model_path = f"training/model/{self.type}_model_user_{self.user_id}.pkl"

        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Modelo excluído de {model_path}")
