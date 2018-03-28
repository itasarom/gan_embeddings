import tqdm
import numpy as np
import torch
import datetime


def unwrap_tensor(tensor, model):
    if model.is_cuda:
        result = tensor.cpu().numpy()
    else:
        result = tensor.numpy()

    return result


class Trainer:
    def __init__(self, model):
        self.model = model
        # self.session = tf.Session(graph=self.model.graph)
        # with self.model.graph.as_default():
            # with self.session.as_default():
                # self.session.run(tf.global_variables_initializer())
                
        
        self.train_loss = []
        self.valid_loss = []
        self.total_epochs = 0
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

                
    def restore(self, path):
        if self.total_epochs > 0:
            raise ValueError(
                "Cannot restore variables to an already trained model! (Trained for {} epochs)"\
                    .format(self.total_epochs))
            
        self.model.load_state_dict(torch.load(path))


    # def restore_latest(self, path):
    #     self.restore(tf.train.latest_checkpoint(path))

    
    def log_epoch(self, epoch_id, train_loss, validation_loss):
        print("After epoch {} validation_loss = {}, train_loss = {}".format(epoch_id, validation_loss, train_loss))
        self.train_loss.append(train_loss)
        self.valid_loss.append(validation_loss)
    
    def train(self, batch_sampler, n_epochs, max_iterations, save_path=".", save_every=None):
        if self.total_epochs > 0:
            print("Already trained for {} epochs, training further".format(self.total_epochs))
            
        valid = batch_sampler.get_valid()
        for epoch_id in range(self.total_epochs, self.total_epochs + n_epochs):
            print("Starting epoch ", epoch_id)
            train_losses = []
            for iteration_id, (x, mask, y) in tqdm.tqdm(enumerate(batch_sampler)):
                # print(x.dtype, mask.dtype, y.dtype)
                loss, _ = self.model.step(x, mask, y)
                train_losses.append(unwrap_tensor(loss.data, self.model))

            validation_loss = self.model.get_loss(x, mask, y)


            self.log_epoch(epoch_id, np.mean(train_losses), unwrap_tensor(validation_loss.data, self.model)[0])
            self.total_epochs += 1

            if isinstance(save_every, int) and self.total_epochs % save_every == 0:
                self.save(save_path)




        if save_every == "after":
            self.save(save_path)
                
    def evaluate(self, x, mask):
        logits, probs = self.model.forward(x, mask)
                
        return unwrap_tensor(probs.data, self.model)




        
        