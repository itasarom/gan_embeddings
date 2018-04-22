import tqdm
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from IPython import display
import sklearn
from sklearn.metrics import log_loss, accuracy_score
import evaluation
import data_processing
import os

def unwrap_tensor(tensor, model):
    if model.is_cuda:
        result = tensor.cpu().numpy()
    else:
        result = tensor.numpy()

    return result




class Trainer:
    def __init__(self, model):
        self.model = model
                
        
        self.classifier_losses = []
        self.transformation_losses = []
        self.discriminator_losses = []
        self.global_iterations = 0

        self.embedding_accuracies = []
        self.validation_discriminator_losses = []
        self.sents1_accuracy = []
        self.sents2_accuracy = []
        # self.sents1_
        self.sents_train_accuracy = []
        self.sents1_loss = []
        self.sents2_loss = []
        self.validation_iterations = []
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

                
    def restore(self, path):
        if self.global_iterations > 0:
            raise ValueError(
                "Cannot restore variables to an already trained model! (Trained for {} global iterations)"\
                    .format(self.global_iterations))
            
        self.model.load_state_dict(torch.load(path))


    
    def log_global_iteration(self, global_iteration, classifier_losses, discriminator_losses, transformation_losses):
        c, d, t = np.mean(classifier_losses), np.mean(discriminator_losses), np.mean(transformation_losses)
        # c, d, t = np.max(classifier_losses), np.max(discriminator_losses), np.max(transformation_losses)

        display.clear_output(wait=True)



        plt.title("current discriminator loss")
        plt.plot(discriminator_losses)
        plt.show()


        plt.title("current transformation loss")
        plt.plot(transformation_losses)
        plt.show()

        plt.title("current classifier loss")
        plt.plot(classifier_losses)
        plt.show()



        print("Iter {} class {}, discr{}, transform {}".format(global_iteration, c, d, t))
        self.transformation_losses.append(t)
        self.classifier_losses.append(c)
        self.discriminator_losses.append(d)

        plt.title("classifier loss")
        plt.plot(self.classifier_losses, color='green', label="train")
        plt.plot(self.validation_iterations, self.sents1_loss, color='orange', label='sents_1')
        plt.plot(self.validation_iterations, self.sents2_loss, color='blue', label='sents_2')
        plt.legend()
        plt.show()



        plt.title("discriminator loss")
        plt.plot(self.discriminator_losses, color='green', label='train')
        plt.plot(self.validation_iterations, self.validation_discriminator_losses, color='red', label='valid')
        plt.legend()
        plt.show()

        plt.title("transformation loss")
        plt.plot(self.transformation_losses)
        plt.show()



    # def classifier_step(self, x, y):
    #     loss = self.classifier.get_loss(inp, mask, inp_y)
    #     self.classifier.optimizer.zero_grad()
    #     loss.backward()
    #     opt = self.classifier_optimizer.step()

    def validate(self, epoch_id, sents1, sents2, embeds1, embeds2):
        self.validation_iterations.append(epoch_id)
        print("after epoch_id")
        self.model.eval()
        result = validate_embeddings(self.model, sents1.vocab, sents2.vocab, embeds1, embeds2, 200, use_cuda=self.model.is_cuda)
        print("Embedding accuracy ", result[0])
        self.embedding_accuracies.append(result[0])
        print("Embedding loss ", result[1])
        self.validation_discriminator_losses.append(result[1])
        print("Confusion matrix\n", result[2])
        vars = result[-1].var(axis=0)
        print("Min component variance {}, median component variance {}, mean component variance {}, max c v {}".format(
                min(vars), np.median(vars), np.mean(vars), max(vars)
            ))

        acc, _, _, loss, _ = validate_sentences(sents1, self.model, self.model.transform1)
        self.sents1_accuracy.append(acc[-1])
        self.sents1_loss.append(loss)
        print("Sents1", acc, loss)

        acc, _, _, loss, _ = validate_sentences(sents2, self.model, self.model.transform2)
        self.sents2_accuracy.append(acc[-1])
        self.sents2_loss.append(loss)
        print("Sents2", acc, loss)

        plt.title("Embedding accuracy")
        plt.plot(self.validation_iterations, self.embedding_accuracies)
        plt.show()

        plt.title("Sents accuracy")
        plt.plot(self.validation_iterations, self.sents1_accuracy, color='orange', label='sents_1')
        plt.plot(self.validation_iterations, self.sents2_accuracy, color='blue', label='sents_2')
        plt.legend()
        plt.show()



    
    def train(self, sents1, sents2, embeds1, embeds2, params):
        save_path = params['save_path']
        save_every = params['save_every']

        if self.global_iterations > 0:
            print("Already trained for {} global iterations, training further".format(self.global_iterations))
            
        sent_valid_1 = sents1.get_valid()
        sent_valid_2 = sents2.get_valid()

        embeds_valid_1 = embeds1.get_batch(params['n_discr_1'])
        embeds_valid_2 = embeds2.get_batch(params['n_discr_2'])

        smoothing = 0.0
        for epoch_id in range(self.global_iterations, self.global_iterations + params['n_iter']):
            self.model.train(True)
            discriminator_losses = []

            for discr_iter_id in range(params['discr_iterations']):
                cur_embeds1 = embeds1.get_batch(params['n_discr_1'])
                cur_embeds2 = embeds2.get_batch(params['n_discr_2'])

                x1, y1 = self.model.prepare_data_for_discriminator(cur_embeds1, smoothing * np.ones(shape=(cur_embeds1.shape[0], ), dtype=np.float32))
                x1 = self.model.transform1(x1)
                x2, y2 = self.model.prepare_data_for_discriminator(cur_embeds2, (1 - smoothing) * np.ones(shape=(cur_embeds2.shape[0], ), dtype=np.float32))
                x2 = self.model.transform2(x2)


                x = torch.cat([x1, x2], 0)
                y = torch.cat([y1, y2], 0)

                # print(x.size())
                # print(y.size())

                discriminator_loss, _ = self.model.discriminator_step(x, y)
                discriminator_losses.append(unwrap_tensor(discriminator_loss.data, self.model)[0])

            transformation_losses = []
            for discr_iter_id in range(params['transform_iterations']):
                cur_embeds1 = embeds1.get_batch(params['n_discr_1'])
                cur_embeds2 = embeds2.get_batch(params['n_discr_2'])

                # x1, y1 = self.model.prepare_data_for_discriminator(cur_embeds1, (1 - smoothing) * np.ones(shape=(cur_embeds1.shape[0], ), dtype=np.float32))
                # x1 = self.model.transform1(x1)
                # x2, y2 = self.model.prepare_data_for_discriminator(cur_embeds2, smoothing * np.ones(shape=(cur_embeds2.shape[0], ), dtype=np.float32))
                # x2 = self.model.transform2(x2)

                y1 = np.random.binomial(1, 0.5, cur_embeds1.shape[0])
                x1, y1 = self.model.prepare_data_for_discriminator(cur_embeds1, y1)
                x1 = self.model.transform1(x1)

                y2 = np.random.binomial(1, 0.5, cur_embeds2.shape[0])
                x2, y2 = self.model.prepare_data_for_discriminator(cur_embeds2, y2)
                x2 = self.model.transform2(x2)


                x = torch.cat([x1, x2], 0)
                y = torch.cat([y1, y2], 0).float()
                # y = torch.bernoulli(0.5 * torch.ones(x.size()[0]))

                transformation_loss, _ = self.model.transformation_step(x, y)
                transformation_losses.append(unwrap_tensor(transformation_loss.data, self.model)[0])


            # classifier_losses = [-1]
            classifier_accuracies = []
            classifier_losses = []
            for sent_iter_id  in range(params['sentence_iterations']):
                    # break
                # for i in range(params['sents_1_iter']):
                    x1, mask1, y1 = self.model.prepare_data_for_classifier(*sents1.get_batch(params['n_sents_1']), self.model.transformation_1)
                    # x1 = self.model.transform1(x1)

                    x2, mask2, y2 = self.model.prepare_data_for_classifier(*sents2.get_batch(params['n_sents_2']), self.model.transformation_2)
                    # x2 = self.model.transform2(x2)

                    x = torch.cat([x1, x2], 0)
                    mask = torch.cat([mask1, mask2], 0)
                    y = torch.cat([y1, y2], 0)

                    # print(x)
                    # print(mask)
                    # print(y)

                    classifier_loss, _ = self.model.classifier_step(x, mask, y)
                    classifier_losses.append(unwrap_tensor(classifier_loss.data, self.model)[0])

                    # probs = self.model.classifier.forward(x, mask)



            self.log_global_iteration(epoch_id, classifier_losses, discriminator_losses, transformation_losses)
            if epoch_id % params["validate_every"] == 0:
                self.validate(epoch_id, sents1, sents2, embeds1.vocab.embeddings, embeds2.vocab.embeddings)

            # x1, y1 = self.model.prepare_data_for_discriminator(embeds_valid_1, np.zeros(shape=(cur_embeds1.shape[0], )))
            # x1 = self.model.transform1(x1)
            # x2, y2 = self.model.prepare_data_for_discriminator(embeds_valid_2 np.ones(shape=(cur_embeds2.shape[0], )))
            # x2 = self.model.transform2(x2)


            # x = torch.cat([x1, x2], 0)
            # y = torch.cat([y1, y2], 0)

            # valid_discriminator_loss = unwrap_tensor(self.model.discriminator.get_loss(x, y).data, self.model)[0]


            # x1, y1 = self.model.prepare_data_for_discriminator(embeds_valid_1, np.ones(shape=(cur_embeds1.shape[0], )))
            # x1 = self.model.transform1(x1)
            # x2, y2 = self.model.prepare_data_for_discriminator(embeds_valid_2 np.zeros(shape=(cur_embeds2.shape[0], )))
            # x2 = self.model.transform2(x2)


            # x = torch.cat([x1, x2], 0)
            # y = torch.cat([y1, y2], 0)

            # valid_transformation_loss = unwrap_tensor(self.model.discriminator.get_loss(x, y).data, self.model)[0]

            # print(unwrap_tensor(classifier_loss.data, self.model))
 
            # plt.clf()

            # print(unwrap_tensor(classifier_loss.data, self.model), unwrap_tensor(discriminator_loss.data, self.model),  unwrap_tensor(transformation_loss.data, self.model))


            self.global_iterations += 1

            if isinstance(save_every, int) and self.global_iterations % save_every == 0:
                self.save(save_path)


        if save_every == "after":
            self.save(save_path)
                
    # def evaluate(self, x, mask):
    #     logits, probs = self.model.forward(x, mask)
                
    #     return unwrap_tensor(probs.data, self.model)


def get_probs(transformation, model, embeddings, batch_size, use_cuda):
    result = []
#     embed_sum = 0.0
#     squares_sum = 0.0
    embeds = []
    for position in range(0, len(embeddings), batch_size):
        x = embeddings[position:position + batch_size]
        x = torch.autograd.Variable(torch.from_numpy(x)).float()
        if use_cuda:
            x = x.cuda()
        
        x = transformation(x)
        _, probs = model.discriminator.forward(x)
        probs = probs.data.cpu().numpy()
        result.append(probs)
        
        x = x.cpu().data.numpy()
#         embed_sum += x
        embeds.append(x)
    
    result = np.vstack(result)
    z = 1 - result
    result = np.hstack([z, result])
    return result, np.vstack(embeds)   
        



def build_confusion_matrix(predicted_probs, true_y):
    n_labels = predicted_probs.shape[1]
    result = np.zeros(shape=(n_labels, n_labels))
    
    pred = predicted_probs.argmax(axis=1)
    true = true_y
    
    for pred_cls in range(n_labels):
        for true_cls in range(n_labels):
            result[true_cls, pred_cls] = np.count_nonzero(true[pred == pred_cls] == true_cls)
    return result


def validate_embeddings(model, vocab1, vocab2, embeddings_1, embeddings_2, batch_size, use_cuda):
    probs_1, t1 = get_probs(model.transform1, model, embeddings_1, batch_size, use_cuda)
    probs_2, t2 = get_probs(model.transform2, model, embeddings_2, batch_size, use_cuda)
    probs = np.vstack([probs_1, probs_2])

    t1 = data_processing.normalize_embeddings(t1)
    t2 = data_processing.normalize_embeddings(t2)

    data_processing.write_embeds("./embeds_1_tmp.vec", t1, vocab1.words)
    data_processing.write_embeds("./embeds_2_tmp.vec", t2, vocab2.words)

    os.system("./run_muse_validation.sh")
    
    pred_1 = probs_1.argmax(axis=1).reshape(-1, 1)
    pred_2 = probs_2.argmax(axis=1).reshape(-1, 1)    
    pred = np.vstack([pred_1, pred_2])
    
    y_true = np.concatenate([np.zeros((len(pred_1),)), np.ones((len(pred_2)))])
    
    
    acc = accuracy_score(y_pred=pred, y_true=y_true)
    loss = log_loss(y_pred=probs, y_true=y_true)
    cm = build_confusion_matrix(probs, y_true)
    
    plt.hist(probs[y_true == 1, 1], bins=100, label="1", color='orange')
    plt.hist(probs[y_true == 0, 1], bins=100, label="0", color='blue')
    
    plt.legend()
    plt.show()
    

    plt.hist(probs[y_true == 0, 1], bins=100, label="0", color='blue')
    plt.hist(probs[y_true == 1, 1], bins=100, label="1", color='orange')
    
    plt.legend()
    plt.show()
    
    t = np.vstack([t1, t2])
    
    return acc, loss, cm, probs, y_true , t

def validate_sentences(sent_sampler, model, transformation):
    model.eval()
    x, mask, y = sent_sampler.get_test()
    true_y = np.zeros(shape=(len(y), len(sent_sampler.unique_labels)), dtype=np.int32)
    for idx, current_y in enumerate(y):
        true_y[idx, current_y] = 1
    
    x, mask, y = model.prepare_data_for_classifier(x, mask, y, transformation)
    
    if model.is_cuda:
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
    

    
    loss = model.classifier.get_loss(x, mask, y).data.cpu().numpy()
    probs = model.classifier(x, mask)[1].data.cpu().numpy()
    
    pred = np.argmax(probs, axis=1)
    
    acc = evaluation.accuracy(predicted_probs=probs, true_y=true_y)
    prec = {}
    rec = {}
    for cls in range(true_y.shape[1]):
        prec[cls] = evaluation.precision_by_class(probs, true_y, cls)
        rec[cls] = evaluation.recall_by_class(probs, true_y, cls)
    
    return acc, prec, rec, loss, evaluation.build_confusion_matrix(probs, true_y)

    
