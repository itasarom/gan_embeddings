import numpy as np
import torch

class RnnClassifier(torch.nn.Module):
    def __init__(self, vocab, n_topics):
        
        super(RnnClassifier, self).__init__()
        
        self.words = vocab.words
        
        self.embeddings = vocab.embeddings
        self.vocab_size, self.embedding_dim = self.embeddings.shape
        self.max_sent_length = vocab.max_sent_length
        
        self.is_cuda = False
        self.n_topics = n_topics
        self.create_graph()


    def forward(self, inp, mask):
        inp = torch.from_numpy(inp)
        mask = torch.autograd.Variable(torch.from_numpy(mask).float(), requires_grad=False)
        indices = torch.LongTensor(inp)
        

        if self.is_cuda:
            inp = inp.cuda()
            mask = mask.cuda()
            indices = indices.cuda()

        
        embeds = self.word_embeddings(indices)
        _, (unrolled, _) = self.rnn_cell(embeds)
        # print(unrolled.size())
        # unrolled = torch.cat((unrolled[0], unrolled[1]), dim=1)
        unrolled = unrolled[0, :, :]
        # print(unrolled.size())
        # print(unrolled.size())
        relu = torch.nn.functional.relu(self.dense(unrolled))
        # print(relu.size())
        logits = self.logits_layer(relu)
        probs = self.softmax_layer(logits)

        # gc.collect()

        # print(probs.size(), logits.size())
        
        return logits, probs


    def get_loss(self, inp, mask, inp_y):
        logits, probs = self.forward(inp, mask)

        inp_y = torch.autograd.Variable(torch.from_numpy(inp_y.argmax(axis=1)))

        if self.cuda:
            inp_y = inp_y.cuda()


        loss = self.loss_function(logits, inp_y)

        return loss


    def step(self, inp, mask, inp_y):
        
        loss = self.get_loss(inp, mask, inp_y)
        self.optimizer.zero_grad()
        loss.backward()
        opt = self.optimizer.step()



        return loss, opt


    def cuda(self):
        self.is_cuda = True
        super(RnnClassifier, self).cuda()

        return self


    def parameters(self):
        return list(filter(lambda p: p.requires_grad, super(RnnClassifier, self).parameters()))
            
            
    def create_graph(self):
    
        
        self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(self.embeddings).float(), requires_grad=False)

        hidden_size = 128

        self.rnn_cell = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)

        # dense_layer_size = 256
        logits_layer_size = 256
        self.dense = torch.nn.Linear(in_features=hidden_size, out_features=logits_layer_size)
        self.logits_layer = torch.nn.Linear(in_features=logits_layer_size, out_features=self.n_topics)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())
        
        