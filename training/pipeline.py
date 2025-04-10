"""
Este módulo define um pipeline de aprendizado de máquina para classificação de transações.

O pipeline utiliza:
- TF-IDF para processar descrições de transações.
- OneHotEncoder para codificação de categorias.
- Regressão Logística para classificação incremental.
"""

from river import compose, feature_extraction, naive_bayes, preprocessing

def build_pipeline():
    """
    Cria um pipeline de aprendizado de máquina para classificação de transações
    """
    return (
        compose.Select('description', 'category') |
        feature_extraction.TFIDF(on='description') +
        preprocessing.OneHotEncoder() |
        naive_bayes.MultinomialNB()
    )
