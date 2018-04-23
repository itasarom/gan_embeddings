import tqdm
import numpy as np
import torch
import datetime

import sklearn
from sklearn.metrics import log_loss, accuracy_score
import evaluation
import data_processing
import os


import os
import argparse
from collections import OrderedDict

from muse.src.utils import bool_flag, initialize_exp
# from muse.src.models import build_model
# from muse.src.trainer import Trainer
from muse.src.my_evaluator import Evaluator
from muse.src.dictionary import Dictionary

from collections import namedtuple

Params = namedtuple


def my_vocab_to_facebook(data, use_cuda):
    vocab, embeddings, lang = data
    words = vocab.words[1:]

    # print(len(words))
    # print(embeddings.shape)
    result = Dictionary(dict(zip(range(len(words)), words)), dict(zip(words, range(len(words )))), lang)

    embeddings = embeddings[1:]
    embedding_result = torch.nn.Embedding(*embeddings.shape)
    embedding_result.weight = torch.nn.Parameter(torch.from_numpy(embeddings).float(), requires_grad=False)
    if use_cuda:
        embedding_result = embedding_result.cuda()
    return result, embedding_result


def run_muse_validation(src, tgt, path, use_cuda):
    src_vocab, src_embeddings = my_vocab_to_facebook(src, use_cuda)
    tgt_vocab, tgt_embeddings = my_vocab_to_facebook(tgt, use_cuda)

    evaluator = Evaluator(src_vocab, src_embeddings, tgt_vocab, tgt_embeddings)
    to_log = OrderedDict({'n_iter': 0})
    evaluator.monolingual_wordsim(to_log)
    # if params.tgt_lang:
    evaluator.crosslingual_wordsim(to_log)
    evaluator.word_translation(to_log, path)
    # evaluator.sent_translation(to_log)

    return to_log

