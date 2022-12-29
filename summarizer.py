import heapq
import numpy as np
from helper import Helper

class Summarizer:
    def __init__(self, summarizer="freq", preprocess = None, num_sentences = 5):
        self.summarizer = summarizer
        self.preprocess = preprocess
        self.num_sentences = num_sentences
        self.weighted_word_freq = dict()
        self.sentence_scores = dict((sentence,0) for sentence in self.preprocess.sentences)
        self.matrix = None #Store PxN matrix for VSM

    def summarize(self, word_normalizer = 'stem'):
        if self.summarizer == 'freq':
            maximum_frequency = max(self.preprocess.word_freq.values())
            for word in self.preprocess.word_freq:
                self.weighted_word_freq[word] = self.preprocess.word_freq[word]/maximum_frequency
            for index, sentence in enumerate(self.preprocess.sentence_tokens):
                score = 0
                for word in sentence.split():
                    if word in self.weighted_word_freq:
                        score+=self.weighted_word_freq[word]
                self.sentence_scores[self.preprocess.sentences[index]] = round(score,3)
        elif self.summarizer == 'artex':
            #self.preprocess.sentence_vectors
            #self.preprocess.sentence_vectors_indexes
            wn_arr = word_normalizer.split('_')
            word_normalizer = wn_arr[0]
            try:
                fix = int(wn_arr[1])
            except:
                fix = 0
            _, self.matrix = Helper.vectorize_word(sentences=self.preprocess.sentence_vectors, word_normalizer = word_normalizer, fix=fix)
            self.artex()        
        return self.find_summary(num_sentences=self.num_sentences, scores=self.sentence_scores)

    
    def artex(self):
        P,N = self.matrix.shape #P and N
        
        a = np.mean(self.matrix, axis=1) #Lexical weight - the number of words in the sentence
        b = np.mean(self.matrix, axis=0) #Global Topic - Average pseudo sentence vector

        #print(len(b),len(self.matrix[0]))
        sxb_angle = np.array([self.compute_angle(j,b) for j in self.matrix])/np.multiply(P,N)
        #sxb_dot = [np.dot(j,b) for j in self.matrix]
        scores = np.multiply(sxb_angle,a)
        for local_idx, global_idx in enumerate(self.preprocess.sentence_vectors_indexes):
            self.sentence_scores[self.preprocess.sentences[global_idx]] = scores[local_idx]
        
        
    @staticmethod
    def compute_angle(vec1,vec2):
        num = np.dot(vec1, vec2)
        denom = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        cross = num/denom if denom else 0
        return np.arccos(cross)
    
    @staticmethod
    def find_summary(num_sentences = 5, scores = None):
        sentences = heapq.nlargest(num_sentences, scores, key = scores.get)
        indices = np.argsort([list(scores.keys()).index(sentence) for sentence in sentences])
        sentences = [sentences[i] for i in indices]
        summary = ' '.join(sentences)
        return summary

