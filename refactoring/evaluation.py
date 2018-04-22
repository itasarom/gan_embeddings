import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training(trainer):
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(len(trainer.train_loss)), np.minimum(trainer.train_loss, 100), color='r', label='train')
    plt.plot(np.arange(len(trainer.train_loss)), np.minimum(trainer.valid_loss, 100), color='b', label='valid')
    plt.legend()
    plt.show()


def accuracy(predicted_probs, true_y):
    res = np.count_nonzero(predicted_probs.argmax(axis=1) == true_y.argmax(axis=1))
    denom = max(len(true_y), 1)
    return res, len(true_y), res/denom


def recall_by_class(predicted_probs, true_y, cls):
    indices = true_y[:, cls] == 1
    predicted_probs = predicted_probs[indices]
    true_y = true_y[indices]
    res = np.count_nonzero(predicted_probs.argmax(axis=1) == true_y.argmax(axis=1))
    denom = max(len(true_y), 1)
    return res, len(true_y), res/denom


def precision_by_class(predicted_probs, true_y, cls):
    indices = predicted_probs.argmax(axis=1) == cls
    predicted_probs = predicted_probs[indices]
    true_y = true_y[indices]
    res = np.count_nonzero(predicted_probs.argmax(axis=1) == true_y.argmax(axis=1))
    denom = max(len(true_y), 1)
    return res, len(true_y), res/denom


def build_confusion_matrix(predicted_probs, true_y):
    n_labels = true_y.shape[1]
    assert true_y.shape == predicted_probs.shape
    result = np.zeros(shape=(n_labels, n_labels))
    
    pred = predicted_probs.argmax(axis=1)
    true = true_y.argmax(axis=1)
    
    for pred_cls in range(n_labels):
        for true_cls in range(n_labels):
            result[true_cls, pred_cls] = np.count_nonzero(true[pred == pred_cls] == true_cls)
    norm = result.sum(axis=1)
    norm = np.maximum(norm, 1)
    result /= norm[:, None]
    return result


def evaluate(trainer, x, mask, y):
    probs = trainer.evaluate(x, mask)
    prec = {}
    rec = {}
    for cls in range(y.shape[1]):
        prec[cls] = precision_by_class(probs, y, cls)
        rec[cls] = recall_by_class(probs, y, cls)
    
    return accuracy(probs, y), prec, rec, build_confusion_matrix(probs, y)

def plot_confusion_matrix(confusion_matrix):
    fig = plt.figure( figsize=(20, 20))
    plt.xlabel("True classes")
    plt.ylabel("Predicted classes")
    sns.heatmap(confusion_matrix, annot=True, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        


def describe(trainer, x, mask, y):

    acc, prec, rec, confusion_matrix = evaluate(trainer, x, mask, y)
    #plot_training(trainer)
    plot_confusion_matrix(confusion_matrix)

    return acc, prec, rec
 
