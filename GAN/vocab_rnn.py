
import numpy as np
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


class BatchSampler:
    def __init__(self, vocab, all_labels, sents, labels, batch_size=32, validation_size=1024, test_size=1024):
        self.vocab = vocab
        self.sents = np.array(sents)
        self.labels = np.array(labels)
        self.unique_labels = list(all_labels.keys())
        assert len(sents) == len(labels)
        
        indices = np.random.permutation(np.arange(len(self.sents)))
        self.train_indices = indices[:len(self.sents) - (validation_size + test_size)]
        self.validation_indices = indices[len(self.sents) - (validation_size + test_size):len(self.sents) - test_size]
        self.test_indices = indices[len(self.sents) - test_size:]
        
        self.train = (self.sents[self.train_indices], self.labels[self.train_indices])
        self.valid = (self.sents[self.validation_indices], self.labels[self.validation_indices])
        self.test = (self.sents[self.test_indices], self.labels[self.test_indices])
           
        
        self.batch_size = batch_size
        
        self.n_labels = len(all_labels)
        self.label_encoder = all_labels
        
    def __iter__(self):
        self.position = 0
        return self
    
    def reset(self):
        self.position = 0
    
    def get_batch(self, x, y):
        res_x, res_mask = self.vocab.transform(x)
        res_y = np.zeros(shape=(len(y), self.n_labels))

        for item_id, cur_label in enumerate(y):
            res_y[item_id, self.label_encoder[cur_label]] = 1
        
        return res_x.astype(np.int64), res_mask, res_y
    
        
    def __next__(self):
            if self.position >= len(self.train[0]):
                raise StopIteration()
                
            x = self.train[0][self.position:self.position + self.batch_size]
            y = self.train[1][self.position:self.position + self.batch_size]
            
            self.position += self.batch_size

        
            return self.get_batch(x, y)
            
    
    def get_valid(self):
        return self.get_batch(self.valid[0], self.valid[1])
    
    def get_test(self):
        return self.get_batch(self.test[0], self.test[1])