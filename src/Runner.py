import os
import argparse

from MWEPreProcessor import MWEPreProcessor
from WordEmbedding import set_fastText_word_embeddings
from Operations import load_pickle, get_logger, dump_pickle
from MWEIdentifier import MWEIdentifier

parser = argparse.ArgumentParser(prog='CICLing_42')
parser.add_argument('-l', '--lang')  # -l EN
parser.add_argument('-t', '--tag')  # -t IOB

args = parser.parse_args()
lang = args.lang.upper()
tag = args.tag

# current_path = os.getcwd()
root_path = os.getcwd()
input_path = os.path.join(root_path, 'input', 'corpora', 'sharedtask-data-master', '1.1')
output_path = os.path.join(root_path, 'output', lang)
we_path = os.path.join(root_path, 'input', 'embeddings')

mwe_path = os.path.join(input_path, lang)
logger = get_logger(mwe_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)


# preprocessing
logger.info('Preparing mwe...')
mwepp = MWEPreProcessor(lang, input_path)
mwepp.set_tagging(tag)
mwepp.set_train_dev()
mwepp.tag()
mwepp.set_test_corpus()
mwepp.update_test_corpus()
mwepp.prepare_to_lstm()
dump_pickle(mwepp, mwepp.train_pkl_path)

mwe_train_path = mwepp.train_pkl_path
mwe_test_path = mwepp.test_pkl_path
mwe_model_path = mwepp.model_pkl_path

logger.info('Reading word embedding...')
word_emb = 'cc.%s.300.vec.gz' % lang.lower()
we_path = os.path.join(we_path, word_emb)
set_fastText_word_embeddings(we_path, mwe_train_path)

# model
params = {'BG': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'DE': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'EL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'EN': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 8, 'epochs': 15},
          'ES': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'EU': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'FA': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
          'FR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'HE': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'HI': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'HR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 8, 'epochs': 15},
          'HU': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
          'IT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
          'LT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'PL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
          'PT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'RO': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
          'SL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
          'TR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12}
          }

logger.info('Reading mwe pkl...')
mwe = load_pickle(mwe_train_path)
mwe_identifier = MWEIdentifier(lang, mwe, logger)
mwe_identifier.set_params(params[lang])
mwe_identifier.set_test()
mwe_identifier.build_model()
mwe_identifier.fit_model()
mwe_identifier.predict()
mwe_identifier.add_tags_to_test()
logger.info('Saving model...')
mwe_identifier.model.save(mwe_model_path)
dump_pickle(mwe_identifier.mwe, mwe_test_path)

# evaluation
logger.info('Evaluating...')
evaluate_script_path = os.path.join(root_path, 'input', 'corpora', 'sharedtask-data-master', '1.1', 'bin')
command = 'python %s/evaluate.py --gold %s/test.cupt --pred %s/test_tagged.cupt --train %s/train.cupt > %s/eval.txt' % (
    evaluate_script_path, mwe_path, mwe_path, mwe_path, output_path)

eval_cmd = os.path.join(root_path, 'eval.cmd')
with open('eval.cmd', 'w') as the_file:
    the_file.write(command)
