from river import compose, feature_extraction, naive_bayes


def build_pipeline():
    """
    Método que constroi o Pipeline
    """
    return compose.Pipeline(
        feature_extraction.BagOfWords(on='description'),
        naive_bayes.MultinomialNB()
    )
