from river import compose, feature_extraction, naive_bayes, preprocessing


def build_pipeline():
    """
    Cria um pipeline de aprendizado de máquina para classificação de transações
    Usando combinação de BagOfWords e TFIDF para melhor representação do texto
    """
    return compose.Pipeline(
        ('tokenizer', feature_extraction.BagOfWords(lowercase=True, strip_accents=True)),
        ('tfidf', feature_extraction.TFIDF()),
        ('normalizer', preprocessing.StandardScaler()),
        ('model', naive_bayes.MultinomialNB()),
    )
