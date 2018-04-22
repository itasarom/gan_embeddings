import numpy as np
import torch
import copy

class RnnClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, n_topics, config):
        # print(self.__class__,super(RnnClassifier, self).__class__)
        
        super(RnnClassifier, self).__init__()

        self.config = config
        
        self.embedding_dim = embedding_dim
        
        self.is_cuda = False
        self.n_topics = n_topics
        self.create_graph()


    def forward(self, embeds, mask):


        
        # embeds = self.word_embeddings(indices)
        embeds = self.dropout(embeds)
        _, (unrolled, _) = self.rnn_cell(embeds)
        unrolled = unrolled[0, :, :]
        relu = torch.nn.functional.relu(self.dense(unrolled))
        logits = self.logits_layer(relu)
        probs = self.softmax_layer(logits)

        
        return logits, probs


    def get_loss(self, inp, mask, inp_y):
        logits, probs = self.forward(inp, mask)



        loss = self.loss_function(logits, inp_y)

        return loss


    # def step(self, inp, mask, inp_y, optimizer):
        
    #     loss = self.get_loss(inp, mask, inp_y)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     opt = optimizer.step()

    #     return loss, opt


    def cuda(self):
        self.is_cuda = True
        super(RnnClassifier, self).cuda()

        return self


    def parameters(self):
        return list(filter(lambda p: p.requires_grad, super(RnnClassifier, self).parameters()))
            
            
    def create_graph(self):
    
        
        # self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(self.embeddings).float(), requires_grad=False)

        # hidden_size = 128
        hidden_size = self.config["hidden_size"]

        self.rnn_cell = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)

        # dense_layer_size = 256
        # logits_layer_size = 256
        logits_layer_size = self.config["logits_layer_size"]
        self.dense = torch.nn.Linear(in_features=hidden_size, out_features=logits_layer_size)
        self.logits_layer = torch.nn.Linear(in_features=logits_layer_size, out_features=self.n_topics)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()


        
        self.dropout = torch.nn.Dropout(p=self.config["dropout_rate"])
