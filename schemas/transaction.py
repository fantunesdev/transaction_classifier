from typing import Optional

from pydantic import BaseModel


class Transaction(BaseModel):
    """
    Classe para validação dos lançamentos
    """

    description: str
    category: Optional[str] = None
