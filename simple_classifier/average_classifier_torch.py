import numpy as np
import torch
import gc


class AverageClassifier(torch.nn.Module):
    def __init__(self, vocab, n_topics):
        super(AverageClassifier, self).__init__()
        
        self.words = vocab.words
        
        self.embeddings = vocab.embeddings
        self.vocab_size, self.embedding_dim = self.embeddings.shape
        self.max_sent_length = vocab.max_sent_length
        
        # self.graph = tf.Graph()
        self.is_cuda = False
        self.n_topics = n_topics
        self.create_graph()
        # with self.graph.as_default():
            # self.create_graph()
            # self.saver = tf.train.Saver()
    
    def forward(self, inp, mask):
        inp = torch.from_numpy(inp)
        mask = torch.autograd.Variable(torch.from_numpy(mask).float(), requires_grad=False)
        indices = torch.LongTensor(inp)
        

        if self.is_cuda:
            inp = inp.cuda()
            mask = mask.cuda()
            indices = indices.cuda()

        
        embeds = self.word_embeddings(indices)
        # print(embeds)
        # print(embeds.data.type())
        summed = torch.sum(embeds, dim=1)/torch.sum(mask, dim=1).view(-1, 1)
        # print(summed.data.type())
        relu = self.dense(summed).clamp(min=0)
        logits = self.logits_layer(relu)
        probs = self.softmax_layer(logits)

        gc.collect()
        
        return logits, probs


    def get_loss(self, inp, mask, inp_y):
        logits, probs = self.forward(inp, mask)

        inp_y = torch.autograd.Variable(torch.from_numpy(inp_y.argmax(axis=1)))

        if self.is_cuda:
            inp_y = inp_y.cuda()


        loss = self.loss_function(logits, inp_y)

        return loss


    def step(self, inp, mask, inp_y):
        
        loss = self.get_loss(inp, mask, inp_y)

        # print(loss.data.type())

        self.optimizer.zero_grad()
        loss.backward()
        opt = self.optimizer.step()



        return loss, opt


    def cuda(self):
        self.is_cuda = True
        super(AverageClassifier, self).cuda()
    #     for param in self.parameters():
    #         print("yeeee", param)
    #         param.cuda()

        return self


    def parameters(self):
        return list(filter(lambda p: p.requires_grad, super(AverageClassifier, self).parameters()))

            
    def create_graph(self):
 


        self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(self.embeddings).float(), requires_grad=False)
        self.dense = torch.nn.Linear(in_features=self.embedding_dim, out_features=1024)
        torch.nn.init.xavier_uniform(self.dense.weight, gain=torch.nn.init.calculate_gain('relu'))
        # print(self.dense.weight.data.type())
        self.logits_layer = torch.nn.Linear(in_features=1024, out_features=self.n_topics)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())


        