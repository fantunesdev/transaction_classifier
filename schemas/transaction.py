from typing import Optional

from pydantic import BaseModel


class Transaction(BaseModel):
    """
    Classe para validação dos lançamentos

    TODO: Atualmente o treinamento está sendo feito com base nos lançamentos do usuário, mas talvez o mais adequado
    seria criar um model no django que represente as edições feitas pelos usuários nos campos categoria, subcategoria
    e descrição. Adicionando o cartão ou a conta, a validação pode ficar ainda mais assertiva.
    """

    description: str
    category: Optional[str] = None
