from fastapi import Body, Depends, FastAPI, HTTPException

from api.auth import get_token_from_header, verify_token
from schemas.transaction import Transaction
from training.transaction_classifier import TransactionClassifier

app = FastAPI()

@app.get('/status')
async def get_status(payload: dict = Depends(verify_token)):
    """
    Obtém os dados do status de treinamento do modelo do usuário.
    """
    classifier = TransactionClassifier(payload['user_id'])
    return classifier.status()


@app.post('/train')
async def train_model(payload: dict = Depends(verify_token), token: str = Depends(get_token_from_header)):
    """
    Processa os dados e treina um modelo para o usuário

    :payload: dict - Usado para autenticação.
    :token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
    """
    try:
        classifier = TransactionClassifier(payload['user_id'])
        return classifier.train(token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/feedback')
async def feedback(feedbacks: list = Body(...), payload: dict = Depends(verify_token), token: str = Depends(get_token_from_header)):
    """
    Processa os dados para dar feedback para o modelo

    :categorization_feedbacks - Lista de feedbacks
    :token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
    """
    try:
        classifier = TransactionClassifier(payload['user_id'])
        return classifier.retrain_from_feedback(feedbacks, token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/predict')
async def predict(transaction: Transaction, payload: dict = Depends(verify_token)):
    """
    Prediz a categoria e a subcategoria com base na descrição do lançamento

    :transaction: Transaction - Um objeto do tipo Transaction que contenha a descrição
    """
    try:
        classifier = TransactionClassifier(payload['user_id'])
        result = classifier.predict(transaction.description, transaction.category or '')
        return {'category_id': result['category_id'], 'subcategory_id': result['subcategory_id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/predict-batch')
async def predict_batch(transactions: list = Body(...), payload: dict = Depends(verify_token)):
    """
    Prediz as categorias e subcategorias com base nas descrições do lançamento.

    :transactions (list): Uma lista de objetos do tipo Transaction
    """
    try:
        classifier = TransactionClassifier(payload['user_id'])
        results = []

        for transaction_data in transactions:
            transaction = Transaction(**transaction_data)
            prediction = classifier.predict(transaction.description, transaction.category or '')
            results.append(prediction)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
