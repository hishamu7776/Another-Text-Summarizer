import nltk

from nltk.corpus import stopwords

from preprocess import Preprocess
from normalizer import Normalizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    file_path = "../dailymail/dataset/test.csv"
    stopwords = set(stopwords.words('english'))
    preprocess = Preprocess(path=file_path, columns=['article', 'highlights'])
    preprocess.select_a_document()
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=2)

    normalizer = Normalizer()
    
    stem = normalizer.normalize(sentences=preprocess.sentence_vectors ,type="stemming")
    stem_vector, stem_matrix = normalizer.vectorize(stem)

    lemma = normalizer.normalize(sentences=preprocess.sentence_vectors ,type="lemmatizer")
    lemma_vector, lemma_matrix = normalizer.vectorize(stem)

    ultra_stem_fix_1 = normalizer.normalize(sentences=preprocess.sentence_vectors ,type="ultra-stemming",fix=1)
    ultra_stem_fix_1_vector, ultra_stem_fix_1_matrix = normalizer.vectorize(ultra_stem_fix_1)

    print(lemma_vector)
    
