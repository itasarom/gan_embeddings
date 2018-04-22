import numpy as np
import torch

class Discriminator(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.create_graph()

    def forward(self, inp):

        logits = self.actual_model.forward(inp)
        
        probs = self.softmax(logits)
        
        return logits, probs


    def get_loss(self, inp,  inp_y):
        logits, probs = self.forward(inp)
        # print(inp.size(), inp_y.size())
        # print(logits.size(), probs.size())

        loss = torch.nn.functional.binary_cross_entropy(probs.view(-1), inp_y)

        return loss


    # def step(self, inp,  inp_y, optimizer):
        
    #     loss = self.get_loss(inp, inp_y)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     opt = optimizer.step()

    #     return loss, opt


    def cuda(self):
        self.is_cuda = True
        super(Discriminator, self).cuda()
        return self


    def parameters(self):
        return list(filter(lambda p: p.requires_grad, super(Discriminator, self).parameters()))
            
            
    def create_graph(self):
    
        n_hidden_1 = 2048
        n_hidden_2 = 2048

        self.actual_model = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, n_hidden_1),
                # torch.nn.BatchNorm1d(n_hidden_1),
                torch.nn.LeakyReLU(negative_slope=0.2),
                torch.nn.Linear(n_hidden_1, n_hidden_2),
                # torch.nn.Dropout(),
                # torch.nn.BatchNorm1d(n_hidden_2),
                torch.nn.LeakyReLU(negative_slope=0.2),
                torch.nn.Linear(n_hidden_2, 1),
                # torch.nn.Linear(n_hidden_1, 2)

            )


        self.softmax = torch.nn.Sigmoid()

        # self.loss_function = torch.nn.CrossEntropyLoss()
        # self.loss_function = torch.nn.functional.binary_cross_entropy

        self.optimizer = torch.optim.Adam(self.parameters())

        
        