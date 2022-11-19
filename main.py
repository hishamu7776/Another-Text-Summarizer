
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from preprocess import Preprocess

nltk.download('stopwords')
nltk.download('punkt')

if __name__ == "__main__":
    file_path = "../dailymail/dataset/test.csv"
    stopwords = set(stopwords.words('english'))
    preprocess = Preprocess(path=file_path,columns=['article','highlights'])
    preprocess.select_a_document()
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords)
    

