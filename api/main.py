from fastapi import Body, Depends, FastAPI, HTTPException

from api.auth import get_token_from_header, verify_token
from schemas.transaction import Transaction
from training.transaction_classifier import TransactionClassifier

app = FastAPI()


@app.post('/train/{user_id}')
async def train_model(user_id: int, payload: dict = Depends(verify_token), token: str = Depends(get_token_from_header)):
    """
    Processa os dados e treina um modelo para o usuário

    :user_id: int - Id do usuário logado.
    :payload: dict - Usado para autenticação.
    :token: str - Um token JWT criado pela aplicação Django que será usado na autentificação.
    """
    try:
        classifier = TransactionClassifier(user_id)
        return classifier.train(token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/predict/{user_id}')
async def predict(user_id: int, transaction: Transaction, payload: dict = Depends(verify_token)):
    """
    Prediz a categoria e a subcategoria com base na descrição do lançamento

    :user_id: int - Id do usuário logado.
    :transaction: Transaction - Um objeto do tipo Transaction que contenha a descrição
    """
    try:
        classifier = TransactionClassifier(user_id)
        result = classifier.predict(transaction.description, transaction.category or '')
        return {'category_id': result['category_id'], 'subcategory_id': result['subcategory_id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
