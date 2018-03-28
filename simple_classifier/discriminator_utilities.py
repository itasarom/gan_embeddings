import tqdm
import numpy as np
import torch
import datetime
import model_utilities_torch as model_utilities
from model_utilities_torch import unwrap_tensor


class Trainer(model_utilities.Trainer):
    def __init__(self, model):
        super(Trainer, self).__init__(model)
        
    
    def train(self, batch_sampler, n_epochs, max_iterations, save_path=".", save_every=None):
        if self.total_epochs > 0:
            print("Already trained for {} epochs, training further".format(self.total_epochs))
            
        valid = batch_sampler.get_valid()
        for epoch_id in range(self.total_epochs, self.total_epochs + n_epochs):
            print("Starting epoch ", epoch_id)
            train_losses = []
            for iteration_id, (x, y) in tqdm.tqdm(enumerate(batch_sampler)):
                # print(x.dtype, mask.dtype, y.dtype)
                loss, _ = self.model.step(x, y)
                train_losses.append(unwrap_tensor(loss.data, self.model))

            validation_loss = self.model.get_loss(valid[0], valid[1])


            self.log_epoch(epoch_id, np.mean(train_losses), unwrap_tensor(validation_loss.data, self.model)[0])
            self.total_epochs += 1

            if isinstance(save_every, int) and self.total_epochs % save_every == 0:
                self.save(save_path)

        if save_every == "after":
            self.save(save_path)



    def evaluate(self, x):
        logits, probs = self.model.forward(x)
                
        return unwrap_tensor(probs.data, self.model)



class BatchSamplerStupid:
    def __init__(self, vocab1, vocab2, batch_size=32, validation_size=1024, test_size=1024):

        self.batch_size = batch_size
        self.vocab1 = vocab1.embeddings
        self.vocab2 = vocab2.embeddings
        self.vocab = np.vstack([self.vocab1, self.vocab2])

        self.y = np.vstack([np.ones(shape=(len(self.vocab1), 1)), np.zeros(shape=(len(self.vocab2), 1))])
        # self.y = np.random.randint(0, high=2, size=(len(self.vocab1) + len(self.vocab2), 1))
        # self.y = self.y[np.random.permutation(np.arange(len(self.vocab)))]
        
        indices = np.random.permutation(np.arange(len(self.vocab)))

        border_train = len(self.vocab) - (validation_size + test_size)
        border_valid = border_train + validation_size
        self.train_indices = indices[:border_train]
        self.validation_indices = indices[border_train:border_valid]
        self.test_indices = indices[border_valid:]
        
        self.train = (self.vocab[self.train_indices], self.y[self.train_indices])
        self.valid = (self.vocab[self.validation_indices], self.y[self.validation_indices])
        self.test = (self.vocab[self.test_indices], self.y[self.test_indices])
        



        
    def __iter__(self):
        self.position = 0
        return self
    
    def reset(self):
        self.position = 0
    
    
        
    def __next__(self):
            if self.position >= len(self.train[0]):
                raise StopIteration()
                
            x = self.train[0][self.position:self.position + self.batch_size]
            y = self.train[1][self.position:self.position + self.batch_size]
            
            self.position += self.batch_size

            # print(x.shape, y.shape)
            return x, y            
    
    def get_valid(self):
        return self.valid[0], self.valid[1]
    
    def get_test(self):
        return self.test[0], self.test[1]




        
        