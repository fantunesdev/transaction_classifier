from pydantic import BaseModel
from typing import Optional


class Transaction(BaseModel):
    """
    Classe para validação dos lançamentos
    """

    description: str
    category: Optional[str] = None
