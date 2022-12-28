import nltk

import numpy as np
import sys


from nltk.corpus import stopwords
from functools import reduce
from preprocess import Preprocess
from vectorizer import Vectorizer
from evaluator import Evaluator
from plots import Plots
#from artex import Artex

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Require parameters")
    else:
        file_path = sys.argv[1] #../dailymail/dataset/test.csv
        dataset_type = sys.argv[2] #single-page or dataframe
        article_col = None
        if len(sys.argv) == 4:
            article_col = sys.argv[3] #article
        
        
    
    stopwords = set(stopwords.words('english'))
    
    
    preprocess = Preprocess(path=file_path, dataset_type=dataset_type)
    preprocess.read_article(column = article_col)
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=2)
    #print('---------------------------------------------------')
    #print(preprocess.sentence_vectors)
    #print('---------------------------------------------------')
    

    '''
    vectorizer = Vectorizer()
    evaluator = Evaluator()
    plots = Plots()
    #artex = Artex()
    print(preprocess.word_tokens)
    
    raw = preprocess.sentence_vectors
    raw_vector, raw_matrix = vectorizer.vectorize(raw)

    stem = vectorizer.normalize(sentences=preprocess.sentence_vectors,
                                type="stemming")
    stem_vector, stem_matrix = vectorizer.vectorize(stem)

    lemma = vectorizer.normalize(sentences=preprocess.sentence_vectors,
                                 type="lemmatizer")
    lemma_vector, lemma_matrix = vectorizer.vectorize(lemma)

    ultra_stem_fix_1 = vectorizer.normalize(
        sentences=preprocess.sentence_vectors, type="ultra-stemming", fix=1)
    ultra_stem_fix_1_vector, ultra_stem_fix_1_matrix = vectorizer.vectorize(
        ultra_stem_fix_1)

    ultra_stem_fix_2 = vectorizer.normalize(
        sentences=preprocess.sentence_vectors, type="ultra-stemming", fix=2)
    ultra_stem_fix_2_vector, ultra_stem_fix_2_matrix = vectorizer.vectorize(
        ultra_stem_fix_2)
    
    print(lemma_matrix)
    #artex.summeraize(matrix=ultra_stem_fix_1_matrix)
        
    density_matrix_raw = evaluator.matrix_density(raw_matrix)
    avg_density_raw = evaluator.compute_average_density(density_matrix_raw)
    volume_raw = evaluator.compute_volume(A=density_matrix_raw,
                                          B=density_matrix_raw)

    density_matrix_stem = evaluator.matrix_density(stem_matrix)
    avg_density_stem = evaluator.compute_average_density(density_matrix_stem)
    volume_stem = evaluator.compute_volume(A=density_matrix_raw,
                                          B=density_matrix_stem)

    density_matrix_lemma = evaluator.matrix_density(lemma_matrix)
    avg_density_lemma = evaluator.compute_average_density(density_matrix_lemma)
    volume_lemma = evaluator.compute_volume(A=density_matrix_raw,
                                          B=density_matrix_lemma)
    density_matrix_fix1 = evaluator.matrix_density(ultra_stem_fix_1_matrix)
    avg_density_fix1 = evaluator.compute_average_density(density_matrix_fix1)
    volume_fix1 = evaluator.compute_volume(A=density_matrix_raw,
                                          B=density_matrix_fix1)

    density_matrix_fix2 = evaluator.matrix_density(ultra_stem_fix_2_matrix)
    avg_density_fix2 = evaluator.compute_average_density(density_matrix_fix2)
    volume_fix2 = evaluator.compute_volume(A=density_matrix_raw,
                                          B=density_matrix_fix2)
                                      
    word_length_distribution = evaluator.word_length_distribution(raw)
    '''
    '''
    plots.plot_density(matrix = density_matrix_raw,x_tick = raw_vector, title="Raw Data")
    plots.plot_density(matrix = density_matrix_stem,x_tick = stem_vector, title="Stemming")
    plots.plot_density(matrix = density_matrix_lemma,x_tick = lemma_vector, title="Lemmatization")
    plots.plot_density(matrix = density_matrix_fix1,x_tick = ultra_stem_fix_1_vector, title="Fix 1")
    plots.plot_density(matrix = density_matrix_fix2,x_tick = ultra_stem_fix_2_vector, title="Fix 2")
    plots.bar_plot(y_val=[avg_density_raw,avg_density_stem,avg_density_lemma,avg_density_fix1,avg_density_fix2],x_val=["Raw","Stem","Lemma","Fix1","Fix2"],title="Average Density")
    plots.bar_plot(y_val=[volume_raw,volume_stem,volume_lemma,volume_fix1,volume_fix2],x_val=["Raw","Stem","Lemma","Fix1","Fix2"],title="Volume")
    plots.line_plot(data=word_length_distribution, x_label="Word Lengths", y_label="Frequency", title="Average distribution of letters")
    #evaluator.cross_correlate_all(matrices = [raw_matrix, stem_matrix, lemma_matrix, ultra_stem_fix_1_matrix, ultra_stem_fix_2_matrix], names=['Raw', 'Stemming', 'Lemmatizer', 'Fix1', 'Fix2'])
    '''
