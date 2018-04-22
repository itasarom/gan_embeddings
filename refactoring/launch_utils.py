# This file is based on 
# https://github.com/sb-nmt-team/sb-nmt/blob/master/toy_problem/baseline/utils/launch_utils.py
# A YSDA project I take part in.

import os
import time
import shutil
import json
from logging import FileHandler, DEBUG
import logging
import inspect

class DebugFileHandler(FileHandler):
  def __init__(self, filename, mode='a', encoding=None, delay=False):
    FileHandler.__init__(self, filename, mode, encoding, delay)

  def emit(self, record):
    if not record.levelno == DEBUG:
      return
    FileHandler.emit(self, record)

LOGGING_BASE = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(asctime)-15s %(module)-10s %(levelname)-8s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level':'INFO',
            'class':'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'stdout': {
            'level':'INFO',
            'class':'logging.FileHandler',
            'formatter': 'verbose',
            'mode': 'w'
        },
        'debug': {
            'level':'DEBUG',
            'class':'utils.launch_utils.DebugFileHandler',
            'formatter': 'verbose',
            'mode': 'w'
        }
    },
    'loggers': {
        'logger': {
            'handlers': ['console', 'stdout', 'debug'],
            'level': 'DEBUG',
        }
    }
}

logger = logging.getLogger('logger')

def log_func(func):
  global logger
  def wrapper(*args, **kwargs):
    isclass = inspect.signature(func).parameters.get('self', None)
    if isclass:
      name = type(args[0]).__name__ + '.' + func.__name__
    else:
      name = func.__name__
    logger.debug("Executing {}.".format(name))

    return func(*args, **kwargs)
  return wrapper


def find_latest_experiment(folder):
  if not len(os.listdir(folder)):
    return None
  latest_experiment_timestamp = sorted([t.split('@')[-1]
                                        for t in os.listdir(folder) if t.split('@')[0] == 'experiment'])[-1]
  return os.path.join(folder, 'experiment@{}'.format(latest_experiment_timestamp))


def create_new_experiment(abs_path, old_folder=None):
  new_folder = os.path.join(abs_path, "experiment@{}".format(time.strftime("%Y-%m-%d_%H_%M_%S", time.gmtime())))

  if old_folder:
    shutil.copytree(old_folder, new_folder)
  else:
    os.makedirs(new_folder)
  return new_folder


def log_experiment_info(model_name, new_folder, latest_folder):
  global logger
  if latest_folder:
    logger.info('Model {} already exists.'.format(model_name))
    logger.info('Creating new experiment based on that model configs at {}.'.format(model_name))
    logger.debug('Copying model from {} to {}.'.format(latest_folder, new_folder))
  else:
    logger.info('Creating completely new experiment at {}.'.format(model_name))
    logger.debug('Creating new folder at: {}'.format(new_folder))

def log_parameters_info(hps, training_params):
  global logger
  translate_to_all_loggers('Final model parameters:')
  for k, v in hps.items():
    translate_to_all_loggers('\t{0: <30} {1}'.format(k, v))

  translate_to_all_loggers('Final training parameters:')
  for k, v in training_params.items():
    translate_to_all_loggers('\t{0: <30} {1}'.format(k, v))

def translate_to_all_loggers(message):
  global logger
  logger.info(message)
  logger.debug(message)