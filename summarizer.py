import heapq
import numpy as np
import helper as Helper

class Summarizer:
    def __init__(self, summarizer="freq", preprocess = None, num_sentences = 5):
        self.summarizer = summarizer
        self.preprocess = preprocess
        self.num_sentences = num_sentences
        self.weighted_word_freq = dict()
        self.sentence_scores = dict()
        self.matrix = []

    def summarize(self):
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
            return self.find_summary(num_sentences=self.num_sentences, scores=self.sentence_scores)
        else:
            self.artex(self.artex(self.matrix))

    @staticmethod
    def artex(matrix = None):
        P,N = matrix.shape
        a = list() #Average pseudo vector
        b= list() #Average pseudo sentence vector
        sentence_weight = list()
        for i in range(P):
            a.append(np.sum(matrix[i])/len(matrix[i]))
        for j in range(N):
            word_count = 0
            for i in range(P):
                word_count += matrix[i][j]
            b.append(word_count/P)
        
        score = list()
        for i,sentence in enumerate(matrix):
            sb = 0
            for j, sj in enumerate(sentence) :
                sb += sj*b[j]
            sb = sb*a[i]
            score.append(sb/(P*N))
        
        a_mag = N*np.sqrt(P)
        b_mag = np.sqrt(N)*P
        s_mag = N
        print(a_mag, b_mag, s_mag)
        
                

    @staticmethod
    def compute_cross(vec1,vec2):
        num = np.dot(vec1, vec2)
        denom = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        cross = num/denom if denom else 0
        return cross

        #prod = np.dot(prod,np.array(a))
        #print(prod)
        
        return
    @staticmethod
    def randbin(P,N,prob=1/2):  
        return np.random.choice([0, 1], size=(P,N), p=[prob, 1-prob])
    
    @staticmethod
    def find_summary(num_sentences = 5, scores = None):
        
        sentences = heapq.nlargest(num_sentences, scores, key = scores.get)
        summary = ' '.join(sentences)
        return summary

