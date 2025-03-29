import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from training.data_fetcher import get_data


class TransactionClassifier:
    """
    Classe responsável por fazer o treinamento ou a predição
    """
    debug = True
    model_dir = 'training/model/'

    def __init__(self, user_id):
        self.model = None
        self.model_path = f'{self.model_dir}{user_id}_model.pkl'
        self.user_id = user_id
        self.vectorizer = None
        self.vectorizer_path = f'{self.model_dir}{user_id}_vectorizer.pkl'

    def train_model(self, token):
        """
        Função para processar dados e treinar o modelo para o usuário
        """
        transactions = get_data('transactions', token)
        transactions_df = pd.DataFrame(transactions)

        # Transformar a coluna 'description' em uma representação numérica usando TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)
        description_matrix = vectorizer.fit_transform(transactions_df['description'])

        # Criar um DataFrame com os dados transformados
        description_df = pd.DataFrame(description_matrix.toarray())

        # Adicionar a coluna 'value' e garantir que todos os nomes das colunas sejam strings
        x = pd.concat([description_df, transactions_df[['value']].reset_index(drop=True)], axis=1)
        x.columns = x.columns.astype(str)

        y = transactions_df[['category', 'subcategory']]

        # Divisão dos dados de treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Treinando o modelo
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        model.fit(x_train, y_train)

        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)

        if self.debug:
            # Fazer previsões
            y_pred = model.predict(x_test)

            # Avaliação
            category_accuracy = accuracy_score(y_test['category'], y_pred[:, 0])
            subcategory_accuracy = accuracy_score(y_test['subcategory'], y_pred[:, 1])
            print(f'[{self.user_id}] Acurácia para "category": {category_accuracy * 100:.2f}%')
            print(f'[{self.user_id}] Acurácia para "subcategory": {subcategory_accuracy * 100:.2f}%')
            print(f'Modelo do usuário {self.user_id} treinado e salvo.')

    def load(self):
        """
        Carrega o modelo treinado e o vetorizer.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            self.model_path = 'training/model/model.pkl'
            self.vectorizer_path = 'training/model/vectorizer.pkl'
            if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
                message = 'O modelo ou o vectorizer não foram encontrados. Treine o modelo primeiro.'
                raise FileNotFoundError(message)

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        if self.debug:
            print('Modelo carregado com sucesso')

    def predict(self, description, value):
        """
        Faz uma previsão de categoria e subcategoria para uma descrição dada.

        :param description: str - Descrição da transação
        :param value: float - Valor da transação
        :return: tuple - Categoria e Subcategoria previstos
        """
        if not self.model or not self.vectorizer:
            self.load()

        description_matrix = self.vectorizer.transform([description])
        x = pd.concat([pd.DataFrame(description_matrix.toarray()), pd.DataFrame([[value]], columns=['value'])], axis=1)
        x.columns = x.columns.astype(str)

        prediction = self.model.predict(x)
        category, subcategory = prediction[0]

        return category, subcategory
