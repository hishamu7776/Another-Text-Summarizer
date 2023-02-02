import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

from preprocess import Preprocess
from summarizer import Summarizer
from helper import Helper

def text_summarizer(document=None,
                    stopwords=None,
                    min_freq=2,
                    summarizer='artex',
                    num_sentences=5,
                    word_normalizer='ultra_1'):
    preprocess = Preprocess(document=document)
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, min_freq=min_freq)
    summarizer = Summarizer(summarizer=summarizer,
                            preprocess=preprocess,
                            num_sentences=num_sentences)
    summary = summarizer.summarize(word_normalizer=word_normalizer)
    return summary


if __name__ == "__main__":
    print(123)
    min_freq = 2
    num_sentences = 3
    summarizer = 'artex'
    normalizer = 'raw'
    file_path = "../summary_dataset.csv"
    stopwords = set(stopwords.words('english'))
    #corpus = pd.read_csv(file_path, encoding='ISO-8859-1')
    #document = corpus.iloc[:, 0][22]
    #corpus = Helper.read_data("D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary")
    #document = corpus[0]
    document, summary = Helper.read_file('D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary\\News Articles\\entertainment\\253.txt','D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary\\Summaries\\entertainment\\253.txt')
    print(document)

    '''
    corpus = open(../summary_dataset.txt, "r", encoding='utf-8')
    document = corpus.read()
    corpus.close()
    '''

    #for normalizer in ['stem','lemma','ultra_1','ultra_2','ultra_3']:
    summary = text_summarizer(document=document,
                stopwords=stopwords,
                min_freq=min_freq,
                summarizer=summarizer,
                num_sentences=num_sentences,
                word_normalizer=normalizer)

