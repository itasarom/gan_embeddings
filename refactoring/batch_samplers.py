import numpy as np

import torch


class BatchSamplerDiscriminator:
    """
    Samples the embeddings as if they were noise in usual GANs
    """

    def __init__(self, vocab):
        # self.batch_size = batch_size
        self.vocab = vocab
        self.word_indices = np.arange(self.vocab.embeddings.shape[0])
    
        
    # def __next__(self):
    #         x = np.random.choice(self.word_indices, size=(self.batch_size, 1), with_return=False).astype(np.int64)
    #         x = self.vocab_tensor.forward(x)
    #         return x        

    def get_batch(self, batch_size):
        x = np.random.choice(self.word_indices[1:], size=(batch_size, ), replace=False).astype(np.int64)
        # x = self.vocab_tensor.forward(x)
        return self.vocab.embeddings[x]
    









class BatchSamplerRegularizer:
    def __init__(self, vocab, all_labels, sents, labels, max_sent_length=None, validation_size=1024, test_size=1024, seed=None):
        self.vocab = vocab
        self.sents = np.array(sents)
        self.labels = np.array(labels)

        if max_sent_length is not None:
            good_sents = [len(sent) <= vocab.max_sent_length for sent in self.sents]
            self.sents = self.sents[good_sents]
            self.labels = self.labels[good_sents]

        self.vocab_size, self.embedding_dim = self.vocab.embeddings.shape

        self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(self.vocab.embeddings).float(), requires_grad=False)

        self.unique_labels = list(all_labels.keys())
        assert len(sents) == len(labels)
        
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.sents)))
        self.train_indices = indices[:len(self.sents) - (validation_size + test_size)]
        self.validation_indices = indices[len(self.sents) - (validation_size + test_size):len(self.sents) - test_size]
        self.test_indices = indices[len(self.sents) - test_size:]
        
        self.train = (self.sents[self.train_indices], self.labels[self.train_indices])
        self.valid = (self.sents[self.validation_indices], self.labels[self.validation_indices])
        self.test = (self.sents[self.test_indices], self.labels[self.test_indices])
        
        self.batch_indices = np.arange(len(self.train_indices))
        
        # self.batch_size = batch_size
        
        self.n_labels = len(all_labels)
        self.label_encoder = all_labels
        
    def __iter__(self):
        self.position = 0
        return self
    
    def reset(self):
        self.position = 0

    def prepare_batch(self, x, y):
        res_y = np.zeros(shape=(len(y), ), dtype=np.int64)
        res_x, res_mask = self.vocab.transform(x)

        for item_id, cur_label in enumerate(y):
            res_y[item_id] = self.label_encoder[cur_label]
            # res

        # res_x = self.vocab_tensor.forward(res_x.astype(np.int64))
        res_x = torch.autograd.Variable(torch.from_numpy(res_x.astype(np.int64)))
        # print(res_x)
        
        return self.word_embeddings(res_x), res_mask, res_y

    
    def get_batch(self, batch_size):

        current_indices = np.random.choice(self.batch_indices, size=(batch_size, ), replace=False)

        x = self.train[0][current_indices]
        y = self.train[1][current_indices]


        return self.prepare_batch(x, y)
    
        
    # def __next__(self):
    #         if self.position >= len(self.train[0]):
    #             raise StopIteration()
                
    #         x = self.train[0][self.position:self.position + self.batch_size]
    #         y = self.train[1][self.position:self.position + self.batch_size]
            
    #         self.position += self.batch_size

        
    #         return self.get_batch(x, y)
            
    def get_train_valid(self):
        valid_size = len(self.valid[0])
        return self.prepare_batch(self.train[0][:valid_size], self.train[1][:valid_size])
    
    def get_valid(self):
        return self.prepare_batch(self.valid[0], self.valid[1])
        # return self.get_batch(self.valid[0], self.valid[1])
    
    def get_test(self):
        return self.prepare_batch(self.test[0], self.test[1])
        # return self.get_batch(self.test[0], self.test[1])
