import tqdm
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from IPython import display
from collections import defaultdict

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
        self.valid_loss = defaultdict(list)
        self.total_epochs = 0

    def prepare_data(self, inp, mask, y):

        
        mask = torch.autograd.Variable(torch.from_numpy(mask), requires_grad=False)
        if self.model.is_cuda:
            mask = mask.cuda()
            inp = inp.cuda()

        y = torch.autograd.Variable(torch.from_numpy(y))

        if self.model.is_cuda:
            y = y.cuda()

        mask = mask.float()
        return inp, mask, y
        
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

    
    def log_epoch(self, epoch_id, train_loss, validation_losses):
        # print("After epoch {} validation_loss = {}, train_loss = {}".format(epoch_id, validation_loss, train_loss))
        display.clear_output(wait=True)
        print("After epoch ", epoch_id)
        self.train_loss.append(train_loss)
        plt.plot(train_loss, label="train")
        for validation_name, validation_loss in validation_losses.items():
            self.valid_loss[validation_name].append(validation_loss)
            plt.plot(self.valid_loss[validation_name], label=validation_name)

        plt.legend()
        plt.show()
    
    def train(self, batch_sampler, n_epochs, max_iterations, validation_sets, save_path=".", save_every=None):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        if self.total_epochs > 0:
            print("Already trained for {} epochs, training further".format(self.total_epochs))
            
        # valid = batch_sampler.get_valid()
        for epoch_id in range(self.total_epochs, self.total_epochs + n_epochs):
            print("Starting epoch ", epoch_id)
            self.model.train()
            train_losses = []
            for iteration_id, (x, mask, y) in tqdm.tqdm(enumerate(batch_sampler)):
                # print(x.dtype, mask.dtype, y.dtype)
                x, mask, y = self.prepare_data(x, mask, y)
                loss = self.model.get_loss(x, mask, y)
                self.optimizer.zero_grad()
                loss.backward()
                opt = self.optimizer.step()
                train_losses.append(unwrap_tensor(loss.data, self.model))

            current_losses = {}
            for validation_name, valid in validation_sets.items():
                x, mask, y = self.prepare_data(*valid)
                validation_loss = self.model.get_loss(x, mask, y)
                current_losses[validation_name] = unwrap_tensor(validation_loss.data, self.model)[0]


            self.log_epoch(epoch_id, np.mean(train_losses), current_losses)
            self.total_epochs += 1

            if isinstance(save_every, int) and self.total_epochs % save_every == 0:
                self.save(save_path)




        if save_every == "after":
            self.save(save_path)
                
    def evaluate(self, x, mask):
        logits, probs = self.model.forward(x, mask)
                
        return unwrap_tensor(probs.data, self.model)



class BatchSampler:
    def __init__(self, vocab, all_labels, sents, labels, batch_size, max_sent_length=None, validation_size=1024, test_size=1024, seed=None):
        self.vocab = vocab
        self.sents = np.array(sents)
        self.labels = np.array(labels)

        self.batch_size = batch_size

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

        res_x = torch.autograd.Variable(torch.from_numpy(res_x.astype(np.int64)))
        
        return self.word_embeddings(res_x), res_mask, res_y

    def __next__(self):
        if self.position >= len(self.train[0]):
            raise StopIteration()
            
        x = self.train[0][self.position:self.position + self.batch_size]
        y = self.train[1][self.position:self.position + self.batch_size]
        
        self.position += self.batch_size

    
        return self.prepare_batch(x, y)
            

    
    # def get_batch(self, batch_size):

    #     current_indices = np.random.choice(self.batch_indices, size=(batch_size, ), replace=False)

    #     x = self.train[0][current_indices]
    #     y = self.train[1][current_indices]


    #     return self.prepare_batch(x, y)
    
    def get_train_valid(self):
        valid_size = len(self.valid[0])
        return self.prepare_batch(self.train[0][:valid_size], self.train[1][:valid_size])
    
            
    
    def get_valid(self):
        return self.prepare_batch(self.valid[0], self.valid[1])
        # return self.get_batch(self.valid[0], self.valid[1])
    
    def get_test(self):
        return self.prepare_batch(self.test[0], self.test[1])
        # return self.get_batch(self.test[0], self.test[1])


# def validate_sentences(sent_sampler, trainer):
#     trainer.model.eval()
#     true_y = np.zeros(shape=(len(y), len(sent_sampler.unique_labels)), dtype=np.int32)
#     for idx, current_y in enumerate(y):
#         true_y[idx, current_y] = 1
    
#     x, mask, y = model.prepare_data_for_classifier(x, mask, y, transformation)
    
#     if model.is_cuda:
#         x = x.cuda()
#         y = y.cuda()
#         mask = mask.cuda()
    

    
#     loss = model.classifier.get_loss(x, mask, y).data.cpu().numpy()
#     probs = model.classifier(x, mask)[1].data.cpu().numpy()
    
#     pred = np.argmax(probs, axis=1)
    
#     acc = evaluation.accuracy(predicted_probs=probs, true_y=true_y)
#     prec = {}
#     rec = {}
#     for cls in range(true_y.shape[1]):
#         prec[cls] = evaluation.precision_by_class(probs, true_y, cls)
#         rec[cls] = evaluation.recall_by_class(probs, true_y, cls)
    
#     return acc, prec, rec, loss, evaluation.build_confusion_matrix(probs, true_y)
