import sys
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
                    occurance=2,
                    summarizer='artex',
                    num_sentences=5,
                    word_normalizer='ultra_1'):
    preprocess = Preprocess(document=document)
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=occurance)
    summarizer = Summarizer(summarizer=method,
                            preprocess=preprocess,
                            num_sentences=num_sentences)
    summary = summarizer.summarize(word_normalizer=word_normalizer)
    return summary


if __name__ == "__main__":
    arg_size = len(sys.argv)
    if arg_size < 5:
        print("Error**-Arguments required for this application.")
        print(
            "Eg:- python main.py './dataset/football_article.txt' 'text' 7 'freq'"
        )
    else:

        file_path, file_type, num_sentences, method, article_col, row_index, word_normalizer, occurance = Helper.get_arguments(
            sys.argv)
        stopwords = set(stopwords.words('english'))
        try:

            if file_type == 'csv':
                corpus = pd.read_csv(file_path, encoding='utf-8')
                document = corpus[article_col][row_index]
            else:
                corpus = open(file_path, "r", encoding='utf-8')
                document = corpus.read()
                corpus.close()
        except:
            print("Error**-Please check your file path or file type.")
        summary = text_summarizer(document=document,
                        stopwords=stopwords,
                        occurance=occurance,
                        summarizer=method,
                        num_sentences=num_sentences,
                        word_normalizer=word_normalizer)
        print(summary)
