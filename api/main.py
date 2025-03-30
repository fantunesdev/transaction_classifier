from fastapi import Body, Depends, FastAPI, HTTPException

from api.auth import verify_token
from schemas.transaction import Transaction
from training.model import TransactionClassifier
from api.auth import get_token_from_header

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
        classifier.train_model(token)
        return {'message': f'Modelo do usuário {user_id} treinado com sucesso!'}
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
        category, subcategory = classifier.predict(transaction.description, transaction.value)
        return {'category': str(category), 'subcategory': str(subcategory)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
