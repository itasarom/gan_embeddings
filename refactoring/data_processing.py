
import numpy as np
import string
import re
import itertools
import os

from vocab import Vocab

def normalize_embeddings(embeddings):
    EPS = 1e-9
    mean = embeddings[1:].mean(axis=0, keepdims=True)
    se = (embeddings[1:].var(axis=0, keepdims=True)  + EPS )**0.5
#     embeddings
    embeddings = (embeddings - mean)/se
    embeddings[0, :] = 0
    return embeddings

digits = set(string.digits)
def normalize_sentence(sent):
    try:
        words = list(re.findall(r"[\w]+", sent, flags=re.UNICODE))
    except:
        print(sent)
        raise
    words = list(map(lambda w: w.lower(), words))
    numbers_filtered = []
    return words

# class Vocab:
#     def __init__(self, embeddings_path, max_sent_length=100):
#         self.pad = '<pad>'
#         self.eos = '</s>'
        
#         f = open(embeddings_path, "r")
#         self.embeddings  = []
#         self.words = [self.pad]
# #         self.embedding_dim = None
#         n_words, self.embedding_dim = map(int, f.readline().strip().split())
#         bad_words = 0

#         word_set = set()
#         for line in f:
#             line = line.strip().split(" ")
#             word = line[0]
#             vec = np.array(list(map(float, line[1:]))).reshape(1, -1)

#             word_set.add(word)
#             self.words.append(word)
#             self.embeddings.append(vec)
#             assert self.embedding_dim == vec.shape[1], (self.embedding_dim, vec.shape[1], vec)
                
#         self.embeddings = [np.zeros(shape=(1, self.embedding_dim))] + self.embeddings
#         self.max_sent_length = max_sent_length
        
#         self.transformation = dict(zip(self.words, range(len(self.words))))
#         self.embeddings  = np.vstack(self.embeddings)
        
#         assert n_words + 1 == len(self.embeddings) + bad_words, (n_words + 1, len(self.embeddings) + bad_words)
#         assert n_words + 1 == len(self.transformation) + bad_words, (n_words + 1, len(self.transformation) + bad_words)
    
#     def transform_one(self, sent):
        
#         result = np.zeros(shape=(len(sent) + 1, ), dtype=np.int32)
#         for word_id, word in enumerate(sent):
#             result[word_id] = self.transformation[word]
        
#         result[word_id + 1] = self.transformation[self.eos]
        
#         return result
        
#     def transform(self, sents):
        
#         batch_max_length = 0
#         for sent in sents:
#             batch_max_length = min(max(batch_max_length, len(sent)), self.max_sent_length) 
        
#         result = np.zeros(shape=(len(sents), batch_max_length + 1))
#         mask = np.zeros(shape=(len(sents), batch_max_length + 1))
        
#         for sent_id, sent in enumerate(sents):
#             sent = sent[:batch_max_length]
#             current = self.transform_one(sent)
#             result[sent_id, :len(current)] = current
#             mask[sent_id, :len(current)] = 1.0
            
#         return result, mask



def read_all_labels(file_path):
    f = open(file_path, "r")
    all_labels = {}
    all_labels_inverse = {}
    for line in f:
        label_id, label = line.strip().split("\t")
        label_id = int(label_id)
        all_labels[label] = label_id
        all_labels_inverse[label_id] = label
        
    return all_labels, all_labels_inverse




def read_data(path, vocab, all_labels, texts_file_name, labels_file_name):
    def process_sentence(sent):
        tmp = []
        for word in normalize_sentence(sent):

            if word not in vocab.transformation:
                word = vocab.pad
            tmp.append(word)
            
        return tmp
        
    
    # text_file = open(path + "/contents.tsv", "r")
    # label_file = open(path + "/labels.tsv", "r")

    text_file = open(os.path.join(path, texts_file_name), "r")
    label_file = open(os.path.join(path, labels_file_name), "r")
    
    sents = []
    labels = []
    
    punctuation = set(string.punctuation)
    bad_sentences = 0
    for sent, label in itertools.zip_longest(text_file, label_file):
#         sent = sent.strip().split(" ")
        
        sent = process_sentence(sent)
        if sent is None:
            bad_sentences += 1
            continue

        label = label.strip()
        
        sents.append(sent)
        labels.append(label)
        
    # print("bad | good: ", bad_sentences, len(sents))
    
    return sents, labels, len(all_labels)
        
        

def load_problem(lang, max_sent_length, data_path, embeddings_file_name, texts_file_name, labels_file_name, topics_file_name):
    # vocab = Vocab("../data_texts/{}/embeddings.vec".format(lang), max_sent_length)
    vocab = Vocab(os.path.join(data_path, lang, embeddings_file_name), max_sent_length)
    # all_labels, all_labels_inverse  = read_all_labels("../data_texts/topics.csv")
    all_labels, all_labels_inverse  = read_all_labels(os.path.join(data_path, topics_file_name))
    # sents, labels, n_topics = read_data("../data_texts/{}/".format(lang), vocab, all_labels)
    sents, labels, n_topics = read_data(os.path.join(data_path, lang), vocab, all_labels, texts_file_name, labels_file_name)

    return vocab, all_labels, sents, labels


def write_embeds(file_path, embeds, words):
    words = words[1:]
    embeds = embeds[1:]
    assert len(words) == len(embeds)
    assert len(embeds.shape) == 2

    with open(file_path, "w") as f:
        print(len(words), embeds.shape[1], file=f)
        for word, vector in zip(words, embeds):
            s = " ".join([str(item) for item in vector])[:-1]
            print(word, s, file=f)
            # f.write(word + " ")


def write_vocab(file_path, vocab):
    words = vocab.words[1:]
    embeds = vocab.embedddings[1:]

    write_embeds(file_path, embeds, words)



class DummyVocab:
    def __init__(self, max_sent_length=10):
        self.pad = '<pad>'
        self.eos = '<eos>'
        
        self.max_sent_length = max_sent_length
        self.words = [self.pad] + [chr(ord('a') + i) for i in range(26)] + [self.eos]
        self.transformation = dict(zip(self.words, np.arange(len(self.words))))
        self.embeddings  = np.random.rand(26 + 2, 256).astype(np.float32)
    
    def transform_one(self, sent):
        
        result = np.zeros(shape=(len(sent) + 1, ))
        for word_id, word in enumerate(sent):
            result[word_id] = self.transformation[word]
        
        result[word_id + 1] = self.transformation[self.eos]
        
        return result
        
    def transform(self, sents):
        
        batch_max_length = 0
        for sent in sents:
            batch_max_length = min(max(batch_max_length, len(sent)), self.max_sent_length) 
        
        result = np.zeros(shape=(len(sents), batch_max_length + 1))
        
        
        for sent_id, sent in enumerate(sents):
            sent = sent[:batch_max_length]
            current = self.transform_one(sent)
            result[sent_id, :len(current)] = current
            
        return result
