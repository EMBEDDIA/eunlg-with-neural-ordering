
import os

import torch
import torch.nn.functional as F
from experiments.src.models.merge import *

# choose path as desired
PROJ_DIR = '~/embeddia'

WRKDIR = os.path.join(PROJ_DIR, 'embeddia-experiments', 'experiments')
MODEL_DIR = os.path.join(PROJ_DIR, 'models', 'trained')
SCORER_DIR = os.path.join(PROJ_DIR, 'models', 'scorers')
EMB_DIR = os.path.join(PROJ_DIR, 'embs')
SPLITS_DIR = os.path.join(WRKDIR, 'splits')     # dir with tr/te indices
STATFI_PATH = os.path.join(WRKDIR, 'statfi')       # path to corpus dir

STATFI_FILES = {'english': 'statfi_en.txt', 'finnish': 'statfi_fi.txt', 'swedish': 'statfi_sv.txt'}

berts = {'finnish': 'bert-base-finnish-cased-v1', 'english': 'bert-base-uncased',
         'swedish': 'bert-base-swedish-cased', 'finest': 'finest-bert-pytorch'}
sberts = {'english': ['roberta-large-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens',
                      'bert-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens'],
          'multi': ['distiluse-base-multilingual-cased-v2', 'xlm-r-distilroberta-base-paraphrase-v1',
                    'xlm-r-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking']}

elmos = {'finnish': 'finnish-elmo', 'swedish': 'swedish-elmo',
         'english': 'elmo_2x1024_128_2048cnn_1xhighway'}
EMB_MODELS = {'bert': berts, 'elmo': elmos}

MERGES = {'bln': torch.nn.Bilinear, 'abs': abs_diff, 'sqd': squared_diff, 'cct': concatenate,
          'avg': mean_emb}

LOGPATH = os.path.join(WRKDIR, 'output', 'log.txt')


DEVICE = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

if not os.getcwd() != WRKDIR:
    os.chdir(WRKDIR)

LOSSES = {'mse': torch.nn.MSELoss, 'bce': torch.nn.BCEWithLogitsLoss, 'nll': torch.nn.NLLLoss,
          'ce': torch.nn.CrossEntropyLoss}
OPTIMS = {'sgd': t.optim.SGD, 'adadelta': t.optim.Adadelta, 'adam': t.optim.Adam}

ACTIVATIONS = {'relu': t.nn.ReLU(), 'sig': t.nn.Sigmoid(), 'tanh': t.nn.Tanh()}

ACT_FNS = {'relu': F.relu, 'sig': F.sigmoid, 'tanh': F.tanh}

EMB_DIMS = {'elmo': {'finnish': 1024, 'swedish': 1024, 'english': 256},
            'bert': 768,  'word2vec': 300, 'glove': 300, 'rand': 300, 'fasttext': 300,
            'sbert': {'finnish': 768, 'swedish': 768, 'english': 1024}
            }

MAXLENS = {'elmo': 100, 'bert': 100}

LANGS = {'fi': 'finnish', 'sw': 'swedish', 'en': 'english'}

MODELS = {'pw': ['PairCNN', 'PairLSTM', 'Bilinear', 'PairFC', 'PairRNN'], 'pos': ['PositionNet'],
          'order': ['PointerNet', 'ATTOrderNet', 'TGCM']
          }
