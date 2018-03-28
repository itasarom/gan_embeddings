
import numpy as np
import torch
import string

class Vocab:
    def __init__(self, embeddings_path, max_sent_length=100):
        self.pad = '<pad>'
        self.eos = '</s>'
        
        f = open(embeddings_path, "r")
        self.embeddings  = []
        self.words = [self.pad]
#         self.embedding_dim = None
        n_words, self.embedding_dim = map(int, f.readline().strip().split())
        bad_words = 0

        word_set = set()
        for line in f:
            line = line.strip().split(" ")
            word = line[0]
            vec = np.array(list(map(float, line[1:]))).reshape(1, -1)

            word_set.add(word)
            self.words.append(word)
            self.embeddings.append(vec)
            assert self.embedding_dim == vec.shape[1], (self.embedding_dim, vec.shape[1], vec)
                
        self.embeddings = [np.zeros(shape=(1, self.embedding_dim))] + self.embeddings
        self.max_sent_length = max_sent_length
        
        # print(self.embeddings[0])
        self.transformation = dict(zip(self.words, range(len(self.words))))
        self.embeddings  = np.vstack(self.embeddings)
        
        assert n_words + 1 == len(self.embeddings) + bad_words, (n_words + 1, len(self.embeddings) + bad_words)
        assert n_words + 1 == len(self.transformation) + bad_words, (n_words + 1, len(self.transformation) + bad_words)

    def embed_word_list(self, id_list):
        return self.embeddings[id_list]

    def get_word_embeddings(self, words):
        return self.embed_word_list(self.transform_one(words))

    
    def transform_one(self, sent):
        # print(sent)
        
        result = np.zeros(shape=(len(sent) + 1, ), dtype=np.int32)
        for word_id, word in enumerate(sent):
            result[word_id] = self.transformation[word]
        
        result[word_id + 1] = self.transformation[self.eos]
        
        return result
        
    def transform(self, sents):

        
        batch_max_length = 0
        for sent in sents:
            # print("sent length", len(sent))
            batch_max_length = min(max(batch_max_length, len(sent)), self.max_sent_length) 
        
        # print("BML", batch_max_length)
        result = np.zeros(shape=(len(sents), batch_max_length + 1))
        mask = np.zeros(shape=(len(sents), batch_max_length + 1), dtype=np.int64)
        
        for sent_id, sent in enumerate(sents):

            sent = sent[:batch_max_length]
            current = self.transform_one(sent)
            result[sent_id, :len(current)] = current
            mask[sent_id, :len(current)] = 1
            
        return result, mask










# def read_all_labels(file_path):
#     f = open(file_path, "r")
#     all_labels = {}
#     all_labels_inverse = {}
#     for line in f:
#         label_id, label = line.strip().split("\t")
#         label_id = int(label_id)
#         all_labels[label] = label_id
#         all_labels_inverse[label_id] = label
        
#     return all_labels, all_labels_inverse


