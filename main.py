
import nltk

from nltk.corpus import stopwords 

from preprocessing import Preprocessing

nltk.download('stopwords')
nltk.download('punkt')

file_path = "../dailymail/dataset/test.csv"
stopwords = set(stopwords.words('english'))
preprocessing = Preprocessing()
corpus =  preprocessing.read_data(file_path)
document = preprocessing.get_one_item(column = 'article', corpus = corpus)
highlight = preprocessing.get_one_item(column = 'highlights', corpus = corpus)
texts = preprocessing.prepare_text(document,stopwords)

