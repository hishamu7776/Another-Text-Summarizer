import nltk

from nltk.corpus import stopwords

from preprocess import Preprocess
from vectorizer import Vectorizer
from evaluator import Evaluator
from plots import Plots

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    file_path = "../dailymail/dataset/test.csv"
    stopwords = set(stopwords.words('english'))
    preprocess = Preprocess(path=file_path, columns=['article', 'highlights'])
    vectorizer = Vectorizer()
    evaluator = Evaluator()
    plots = Plots()

    preprocess.select_a_document()
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=2)

    
    
    stem = vectorizer.normalize(sentences=preprocess.sentence_vectors ,type="stemming")
    stem_vector, stem_matrix = vectorizer.vectorize(stem)

    lemma = vectorizer.normalize(sentences=preprocess.sentence_vectors ,type="lemmatizer")
    lemma_vector, lemma_matrix = vectorizer.vectorize(stem)

    ultra_stem_fix_1 = vectorizer.normalize(sentences=preprocess.sentence_vectors ,type="ultra-stemming",fix=1)
    ultra_stem_fix_1_vector, ultra_stem_fix_1_matrix = vectorizer.vectorize(ultra_stem_fix_1)

    raw = preprocess.sentence_vectors
    raw_vector, raw_matrxi = vectorizer.vectorize(raw)

    density_matrix_fix1 = evaluator.matrix_density(ultra_stem_fix_1_matrix)
    plots.plot_density(density_matrix_fix1)
    
