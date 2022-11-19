import itertools
from nltk.stem import WordNetLemmatizer, PorterStemmer



class Normalizer:
    def __init__(self, sentences = None, word_frequency = None):
        self.sentences = sentences
        self.word_frequency = word_frequency
        self.tokens = list(itertools.chain.from_iterable(sentences))

    def stem(self):
        stemmer = PorterStemmer()
        stems = list()
        for token in self.tokens:
            stem = stemmer.stem(token, pos ="a")
            stems.append(stem)
        self.stem_root = set(stems)
        return self.stem_root


    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        lemmas = list()
        for token in self.tokens:
            lemma = lemmatizer.lemmatize(token, pos ="a")
            lemmas.append(lemma)
        self.lemma_root = set(lemmas)
        return self.lemma_root

    def ultra_stem(self,fix=None):
        text = ""
