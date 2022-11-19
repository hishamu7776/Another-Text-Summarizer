import nltk

from nltk.corpus import stopwords

from preprocess import Preprocess
from normalizer import Normalizer

nltk.download('stopwords')
nltk.download('punkt')

if __name__ == "__main__":
    file_path = "../dailymail/dataset/test.csv"
    stopwords = set(stopwords.words('english'))
    preprocess = Preprocess(path=file_path, columns=['article', 'highlights'])
    preprocess.select_a_document()
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=2)

    normalizer = Normalizer(sentences=preprocess.sentence_vectors,
                            word_frequency=preprocess.word_freq)
    
    normalizer.stem()
