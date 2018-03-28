import tqdm
import numpy as np
import tensorflow as tf
import datetime

class Trainer:
    def __init__(self, model):
        self.model = model
        self.session = tf.Session(graph=self.model.graph)
        with self.model.graph.as_default():
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())
                
        
        self.train_loss = []
        self.valid_loss = []
        self.total_epochs = 0
        
    def save(self, path):
        with self.model.graph.as_default():
            with self.session.as_default():
                self.model.saver.save(self.session, path, global_step=self.total_epochs)
                
    def restore(self, path):
        if self.total_epochs > 0:
            raise ValueError(
                "Cannot restore variables to an already trained model! (Trained for {} epochs)"\
                    .format(self.total_epochs))
            
        with self.model.graph.as_default():
            with self.session.as_default():
                self.model.saver.restore(self.session, path)


    def restore_latest(self, path):
        self.restore(tf.train.latest_checkpoint(path))
        
    
    def log_epoch(self, epoch_id, train_loss, validation_loss):
        print("After epoch {} validation_loss = {}, train_loss = {}".format(epoch_id, validation_loss, train_loss))
        self.train_loss.append(train_loss)
        self.valid_loss.append(validation_loss)
    
    def train(self, batch_sampler, n_epochs, max_iterations, save_path=".", save_every=None):
        if self.total_epochs > 0:
            print("Already trained for {} epochs, training further".format(self.total_epochs))
            
        valid = batch_sampler.get_valid()
        with self.model.graph.as_default():
            with self.session.as_default():
                for epoch_id in range(self.total_epochs, self.total_epochs + n_epochs):
                    print("Starting epoch ", epoch_id)
                    train_losses = []
                    for iteration_id, (x, mask, y) in tqdm.tqdm(enumerate(batch_sampler)):
                        loss, _ = self.session.run(
                                    [self.model.loss, self.model.optimization_step], 
                                    feed_dict={
                                        self.model.inp : x,
                                        self.model.inp_y : y,
                                        self.model.mask : mask
                                    }
                                )
                        train_losses.append(loss)

                    validation_loss = self.session.run(
                                self.model.loss,
                                feed_dict={
                                        self.model.inp : valid[0],
                                        self.model.inp_y : valid[2],
                                        self.model.mask : valid[1]
                                    }
                            )


                    self.log_epoch(epoch_id, np.mean(train_losses), validation_loss)
                    self.total_epochs += 1

                    if isinstance(save_every, int) and self.total_epochs % save_every == 0:
                        self.save(save_path)




                if save_every == "after":
                    self.save(save_path)
                
    def evaluate(self, x, mask):
        with self.model.graph.as_default():
            with self.session.as_default():
                probs = self.session.run(
                                self.model.probs, 
                                feed_dict={
                                    self.model.inp : x,
                                    self.model.mask : mask
                                }
                            )
                
        return probs



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
        
        return res_x, res_mask, res_y
    
        
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
        
        