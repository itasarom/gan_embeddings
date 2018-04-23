import argparse
import json
import torch
import os
import sys
import logging
import logging.config



import numpy as np
import torch
import pandas as pd


import data_processing as dp
import model
import model_utilities as util
import vocab
import batch_samplers

import logging
import logging.config

from launch_utils import LOGGING_BASE, find_latest_experiment, \
  create_new_experiment, log_experiment_info, log_parameters_info, \
logger, translate_to_all_loggers

# def normalize_embeddings(embeddings):
#     EPS = 1e-9
#     mean = embeddings[1:].mean(axis=0, keepdims=True)
#     se = (embeddings[1:].var(axis=0, keepdims=True)  + EPS )**0.5
#     embeddings = (embeddings - mean)/se
#     embeddings[0, :] = 0
#     return embeddings

global_config = {
    "model_name":"debug",
    "trained_models":"trained_models_result",
    "data":"data_texts",
    "cuda":"7",
    "use_cuda":True
}

training_params = {
        'save_path': "./model_checkpoint.tc",
        'save_every':10,
        'sentence_iterations':10,
        'discr_iterations':5,
        'transform_iterations':25,
        'n_sents_1':256,
        'n_sents_2':256,
        'n_discr_1':1024,
        'n_discr_2':1024,
        'n_iter':2500,
        'validate_every':100
}

model_params = {
    "src_lang":'en',
    "tgt_lang":'es',
    "max_sent_length":100,
    "sentence_seed":42,
    "embedding_dim":300,
    "transformation_config": {
        "transform_1":"identity",
        "transform_2":"linear",
    },
    "orthogonalization_beta":0.001,
    "discriminator_optimizer":{
        "class":"SGD",
        "params":{
            "lr":0.1
        }
    },

    "transformation_optimizer":{
        "class":"SGD",
        "params":{
            "lr":0.1
        }
    },

    "classifier_optimizer":{
        "class":"Adam",
        "params":{
        }
    },

    "classifier_config":{
        "hidden_size":128,
        "logits_layer_size":256,
        "dropout_rate":0.5,
    },

    "discriminator_config":{
        "n_hidden_1":2048,
         "n_hidden_2":2048,
    },
}

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config["cuda"]

    TASK_NAME = "diplom"
    TRAINED_MODELS_FOLDER = os.path.join(os.path.abspath(os.path.join(__file__ ,"../")), global_config["trained_models"])
    DATASET_DIR = os.path.join(os.path.abspath(os.path.join(__file__ ,"../")), global_config["data"])

    logger_config = LOGGING_BASE

    model_name = os.path.join("", global_config["model_name"])
    model_folder = os.path.join(TRAINED_MODELS_FOLDER, model_name)
    latest_folder = find_latest_experiment(model_folder) if os.path.exists(model_folder) else None
    new_folder = create_new_experiment(model_folder, latest_folder)

    global_config["experiment_dir"] = new_folder
    training_params["experiment_dir"] = new_folder


    logger_config['handlers']['debug']['filename'] = os.path.join(new_folder, 'debug_logs')
    logger_config['handlers']['stdout']['filename'] = os.path.join(new_folder, 'stdout_logs')
    logging.config.dictConfig(logger_config)

    logger.info("Using python binary at {}".format(sys.executable))

    if torch.cuda.is_available() and global_config["use_cuda"]:
        logger.info('GPU found, running on device {}'.format(torch.cuda.current_device()))
    elif training_params.use_cuda:
        logger.warning('GPU not found, running on CPU. Overriding use_cuda to False.')
        global_config["use_cuda"] = False
    else:
        logger.debug('GPU found, but use_cuda=False, consider using GPU.')

    log_experiment_info(model_name, new_folder, latest_folder)


    vocab1, all_labels, sents1, labels1 = dp.load_problem(lang=model_params["src_lang"], max_sent_length=model_params["max_sent_length"])
    vocab2, all_labels, sents2, labels2 = dp.load_problem(lang=model_params["tgt_lang"], max_sent_length=model_params["max_sent_length"])
    vocab1.embeddings = dp.normalize_embeddings(vocab1.embeddings)
    vocab2.embeddings = dp.normalize_embeddings(vocab2.embeddings)

    model_params["n_topics"] = len(all_labels)

    sent_sampler_1 = batch_samplers.BatchSamplerRegularizer(sents=sents1, labels=labels1, vocab=vocab1, all_labels=all_labels, max_sent_length=None, seed=model_params["sentence_seed"])
    sent_sampler_2 = batch_samplers.BatchSamplerRegularizer(sents=sents2, labels=labels2, vocab=vocab2, all_labels=all_labels, max_sent_length=None, seed=model_params["sentence_seed"])
    embed_sampler_1 = batch_samplers.BatchSamplerDiscriminator(vocab1)
    embed_sampler_2 = batch_samplers.BatchSamplerDiscriminator(vocab2)

    cls = model.GAN(model_params)
    if global_config["use_cuda"]:
        cls = cls.cuda()
    trainer = util.Trainer(cls, global_config)

    trainer.train(sent_sampler_1, sent_sampler_2, embed_sampler_1, embed_sampler_2, training_params)
    return 0

if __name__ == "__main__":
    main()
