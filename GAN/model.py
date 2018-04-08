from discriminator import Discriminator
from rnn_classifier import RnnClassifier
import torch
import numpy as np

from itertools import chain

class VocabTensors(torch.nn.Module):
    def __init__(self, embeddings, transformation=None, requires_grad=False):
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.word_embeddings = torch.nn.EmbedTding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(self.embeddings).float(), requires_grad=False)

        self.transformation = transformation

    def get_size(self):
        return self.vocab_size


    def forward(self, indices):
        """
            Performs the embedding process,
            applies the transformation 
            and returns the result
            x: tensor
            returns: tensor
        """

        embeds = self.word_embeddings(indices)
        if self.transformation is not None:
            return self.transformation.forward(embeds)
        else:
            return embeds


    def backward(self):
        if self.transformation is not None:
            return self.transformation.backward()
        else:
            return None


class IdentityTransformation(torch.nn.Module):
    def forward(self, x):
        return x


class GAN(torch.nn.Module):
    def __init__(self, embedding_dim, n_topics):
        super(GAN, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_topics = n_topics



        self.discriminator = Discriminator(embedding_dim)
        self.classifier = RnnClassifier(embedding_dim, n_topics)

        n_hidden_1 = 1024
        n_hidden_2 = 512
        n_hidden_3 = 512

        # self.transformation_1 = torch.nn.Sequential (
        #                         torch.nn.Linear(self.embedding_dim, n_hidden_1),
        #                         torch.nn.BatchNorm1d(n_hidden_1),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_1, n_hidden_2),
        #                         torch.nn.BatchNorm1d(n_hidden_2),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_2, n_hidden_3),
        #                         torch.nn.BatchNorm1d(n_hidden_3),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_3, self.embedding_dim),
        #                         # torch.nn.Linear(1024, self.embedding_dim),
        #                     )

        # self.transformation_2 = torch.nn.Sequential (
        #                         torch.nn.Linear(self.embedding_dim, n_hidden_1),
        #                         torch.nn.BatchNorm1d(n_hidden_1),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_1, n_hidden_2),
        #                         torch.nn.BatchNorm1d(n_hidden_2),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_2, n_hidden_3),
        #                         torch.nn.BatchNorm1d(n_hidden_3),
        #                         torch.nn.LeakyReLU(negative_slope=0.3),
        #                         torch.nn.Linear(n_hidden_3, self.embedding_dim),
        #                         # torch.nn.Linear(1024, self.embedding_dim),
        #                     )

        # self.transformation_1 = torch.nn.Sequential (
                                # torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                            # )
        
        self.transformation_1 = IdentityTransformation()
        #self.transformation_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.transformation_2 = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)


        #self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=0.1)
        self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer, step_size=5000, gamma=0.98) 
        #self.transformation_optimizer = torch.optim.Adam(chain(self.transformation_1.parameters(), self.transformation_2.parameters()), lr=0.1)
        self.transformation_optimizer = torch.optim.SGD(chain(self.transformation_1.parameters(), self.transformation_2.parameters()), lr=0.1)
        self.transformation_scheduler = torch.optim.lr_scheduler.StepLR(self.transformation_optimizer, step_size=25000, gamma=0.98) 
        self.classifier_optimizer = torch.optim.Adam(chain(self.classifier.parameters(), self.transformation_1.parameters(), self.transformation_2.parameters()))

    def orthogonalize(self):
        #return
        beta = 0.001
        W = self.transformation_2.weight.data
        W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        return 
        W = self.transformation_1.weight.data
        W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def numpy(self, tensor):
        if self.is_cuda:
            result = tensor.cpu().numpy()
        else:
            result = tensor.numpy()

        return result


    def cuda(self):
        self.is_cuda = True
        super(GAN, self).cuda()
        return self

    def cpu(self):
        self.is_cuda = False
        super(GAN, self).cpu()
        return self

    def prepare_data_for_classifier(self, inp, mask, y, transform):
        # print(inp)
        # inp = torch.autograd.Variable(torch.from_numpy(inp.astype(np.int64)), requires_grad=False)
        # inp = torch.autograd.Variable(torch.from_numpy(inp), requires_grad=False)
        
        # inp = torch.LongTensor(inp)
        
        mask = torch.autograd.Variable(torch.from_numpy(mask), requires_grad=False)
        if self.is_cuda:
            mask = mask.cuda()
            inp = inp.cuda()

        # print(mask)

        # print(inp[:, -1])
        y = torch.autograd.Variable(torch.from_numpy(y))

        if self.is_cuda:
            y = y.cuda()

        batch_size, max_sentence_len, embedding_dim = inp.size()

        inp = transform(inp.view(batch_size * max_sentence_len, embedding_dim))
        # print(inp[:, -1])
        inp[mask.view(batch_size * max_sentence_len) ^ 1] = 0.0
        # print(inp[:, -1])
        inp = inp.view(batch_size, max_sentence_len, embedding_dim)
        # print("After")
        # print(inp[:, -1])
        mask = mask.float()
        return inp, mask, y

    def prepare_data_for_discriminator(self, inp, inp_y):
        inp = torch.autograd.Variable(torch.from_numpy(inp.astype(np.float32)))      
        # print(inp)
        if self.is_cuda:
            inp = inp.cuda()

        # inp_y = torch.autograd.Variable(torch.from_numpy(inp_y.astype(np.int64).ravel()))
        inp_y = torch.autograd.Variable(torch.from_numpy(inp_y))
        # print(inp_y)

        if self.is_cuda:
            inp_y = inp_y.cuda()

        return inp, inp_y




    def transform1(self, x):
        # print(x)
        return self.transformation_1(x)

    def transform2(self, x):
        # print(x)
        return self.transformation_2(x)



    def classifier_step(self, x, mask, y):
        
        loss = self.classifier.get_loss(x, mask, y)
        self.classifier_optimizer.zero_grad()
        loss.backward()
        opt = self.classifier_optimizer.step()

        return loss, opt

    def discriminator_step(self, x, y):
        
        loss = self.discriminator.get_loss(x, y)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        opt = self.discriminator_optimizer.step()

        return loss, opt

    def transformation_step(self, x, y):
        loss = self.discriminator.get_loss(x, y)
        self.transformation_optimizer.zero_grad()
        loss.backward()
        opt = self.transformation_optimizer.step()

        self.orthogonalize()

        return loss, opt
