import numpy as np
import torch

class Discriminator(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.create_graph()

    def forward(self, inp):
        inp = torch.autograd.Variable(torch.from_numpy(inp.astype(np.float32)))      

        if self.is_cuda:
            inp = inp.cuda()



        logits = self.actual_model.forward(inp)
        
        probs = self.softmax(logits)
        
        return logits, probs


    def get_loss(self, inp,  inp_y):
        logits, probs = self.forward(inp)

        inp_y = torch.autograd.Variable(torch.from_numpy(inp_y.astype(np.int64).ravel()))

        if self.cuda:
            inp_y = inp_y.cuda()

        loss = self.loss_function(logits, inp_y)

        return loss


    def step(self, inp,  inp_y):
        
        loss = self.get_loss(inp, inp_y)
        self.optimizer.zero_grad()
        loss.backward()
        opt = self.optimizer.step()

        return loss, opt


    def cuda(self):
        self.is_cuda = True
        super(Discriminator, self).cuda()
        return self


    def parameters(self):
        return list(filter(lambda p: p.requires_grad, super(Discriminator, self).parameters()))
            
            
    def create_graph(self):
    
        n_hidden_1 = 2048
        n_hidden_2 = 512

        self.actual_model = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, n_hidden_1),
                torch.nn.Linear(n_hidden_1, n_hidden_2),
                torch.nn.Linear(n_hidden_2, 2)
            )


        self.softmax = torch.nn.Softmax(dim=1)

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())

        
        