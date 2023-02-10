
import random
import numpy as np
import pandas as pd
from plots import Plots
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

from preprocess import Preprocess
from summarizer import Summarizer
from helper import Helper
from evaluator import Evaluator

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
    
    return summary, summarizer.time_elapsed


if __name__ == "__main__":
    min_freq = 2
    num_sentences = 3
    summarizer = 'artex'
    normalizer = 'raw'
    file_path = "../summary_dataset.csv"
    stopwords = set(stopwords.words('english'))
    #corpus = pd.read_csv(file_path, encoding='ISO-8859-1')
    #document = corpus.iloc[:, 0][22]
    (texts, summaries) = Helper.read_data("D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary")
    #document = corpus[0]
    document, summary = Helper.read_file('D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary\\News Articles\\entertainment\\253.txt','D:\\Courses\\MSC DS & E\\Natural Language Processing\\Project\\BBC News Summary\\Summaries\\entertainment\\253.txt')

    '''
    corpus = open(../summary_dataset.txt, "r", encoding='utf-8')
    document = corpus.read()
    corpus.close()
    '''
    
    normalizers = ['raw','stem','lemma','ultra_1','ultra_2']
    document_indexes = random.sample(range(len(texts)), 1500)
    results = []
    for normalizer in normalizers:
            document = texts[0]
            generated_summary, time_elapsed = text_summarizer(document=document,
                        stopwords=stopwords,
                        min_freq=min_freq,
                        summarizer=summarizer,
                        num_sentences=num_sentences,
                        word_normalizer=normalizer)
                #print(len(time_elapsed),time_elapsed)
            
            results.append(generated_summary)
    r1_matrix = np.zeros([5,5])
    r2_matrix = np.zeros([5,5])
    rl_matrix = np.zeros([5,5])
    for idx, normalizer1 in enumerate(normalizers):
        for jdx, normalizer1 in enumerate(normalizers):
            r1_matrix[idx,jdx], r2_matrix[idx,jdx], rl_matrix[idx,jdx] = Evaluator.compute_rouge(results[idx], results[jdx])

    Plots.plot_mantel(matrix=r1_matrix,x_tick=normalizers, y_tick= normalizers, title="Similarity between summaries(ROGUE-1)", labels=['Normalizer', 'Normalizer'], annot=True)
    Plots.plot_mantel(matrix=r2_matrix,x_tick=normalizers, y_tick= normalizers, title="Correlation between summaries(ROGUE-2)", labels=['Normalizer', 'Normalizer'], annot=True)
    Plots.plot_mantel(matrix=rl_matrix,x_tick=normalizers, y_tick= normalizers, title="Correlation between summaries(ROGUE-L)", labels=['Normalizer', 'Normalizer'], annot=True)


