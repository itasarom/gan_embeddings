
import numpy as np
import string
import re
import itertools


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
        
        self.transformation = dict(zip(self.words, range(len(self.words))))
        self.embeddings  = np.vstack(self.embeddings)
        
        assert n_words + 1 == len(self.embeddings) + bad_words, (n_words + 1, len(self.embeddings) + bad_words)
        assert n_words + 1 == len(self.transformation) + bad_words, (n_words + 1, len(self.transformation) + bad_words)
    
    def transform_one(self, sent):
        
        result = np.zeros(shape=(len(sent) + 1, ), dtype=np.int32)
        for word_id, word in enumerate(sent):
            result[word_id] = self.transformation[word]
        
        result[word_id + 1] = self.transformation[self.eos]
        
        return result
        
    def transform(self, sents):
        
        batch_max_length = 0
        for sent in sents:
            batch_max_length = min(max(batch_max_length, len(sent)), self.max_sent_length) 
        
        result = np.zeros(shape=(len(sents), batch_max_length + 1))
        mask = np.zeros(shape=(len(sents), batch_max_length + 1))
        
        for sent_id, sent in enumerate(sents):
            sent = sent[:batch_max_length]
            current = self.transform_one(sent)
            result[sent_id, :len(current)] = current
            mask[sent_id, :len(current)] = 1.0
            
        return result, mask



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




def read_data(path, vocab, all_labels):
    def process_sentence(sent):
        tmp = []
        for word in normalize_sentence(sent):

            if word not in vocab.transformation:
                word = vocab.pad
            tmp.append(word)
            
        return tmp
        
    
    text_file = open(path + "/contents.tsv", "r")
    label_file = open(path + "/labels.tsv", "r")
    
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
        
        

def load_problem(lang):
    vocab = Vocab("../data_texts/{}/embeddings.vec".format(lang))
    all_labels, all_labels_inverse  = read_all_labels("../data_texts/topics.csv")
    sents, labels, n_topics = read_data("../data_texts/{}/".format(lang), vocab, all_labels)

    return vocab, all_labels, sents, labels



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