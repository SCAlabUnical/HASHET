import os
from pickle import dump, load

import constants as c
import dataset_handler as dh
import evaluation as ev
import model
import word_embedding_model as emb

# creating folder for save files
if not os.path.exists(c.SAVE_FOLDER):
    os.mkdir(c.SAVE_FOLDER)

print("> Preprocessing data for Word2Vec model")
dh.preprocess_data(save_file=c.W2V_INPUT)
# loading preprocessed sentences from saved file
sentences = dh.load(c.W2V_INPUT)

print("> Training Word2Vec model")
emb.create_and_train_Word2Vec_model(sentences, mincount=c.W2V_MINCOUNT)
print("W2V MODEL TRAINING SUCCEDED")

# Get data for Google Universal Sentence Encoder
input_file = "Sample.txt"
print("> Preprocessing data for sentence embeddings")
dh.preprocess_data_for_sentence_embedding(input_file)

dh.prepare_train_test(c.PERC_TEST)

print("> Loading Word2Vec model for calculating targets")
w_emb = emb.load_Word2Vec_model()

print("> Preparing data for neural network training")
emb_sentences_train, emb_sentences_test, targets_train, targets_test, ht_lists = dh.prepare_mlp_inputs_and_targets(
    w_emb)

# saving pickles
dump(emb_sentences_train, open(c.SAVE_FOLDER + 'emb_sentences_train.pkl', 'wb'))
dump(emb_sentences_test, open(c.SAVE_FOLDER + 'emb_sentences_test.pkl', 'wb'))
dump(targets_train, open(c.SAVE_FOLDER + 'targets_train.pkl', 'wb'))
dump(targets_test, open(c.SAVE_FOLDER + 'targets_test.pkl', 'wb'))
dump(ht_lists, open(c.SAVE_FOLDER + 'ht_lists.pkl', 'wb'))
# loading pickles
emb_sentences_train = load(open(c.SAVE_FOLDER + 'emb_sentences_train.pkl', 'rb'))
emb_sentences_test = load(open(c.SAVE_FOLDER + 'emb_sentences_test.pkl', 'rb'))
targets_train = load(open(c.SAVE_FOLDER + 'targets_train.pkl', 'rb'))
targets_test = load(open(c.SAVE_FOLDER + 'targets_test.pkl', 'rb'))

print("> Training MLP model")
model.create_and_train_MLP(emb_sentences_train, emb_sentences_test, targets_train, targets_test)
print("> Loading MLP model and making predictions")
model.predict_hashtags_and_store_results(c.TEST_CORPUS, emb_sentences_test)

# Evaluate model
ht_lists = load(open(c.SAVE_FOLDER + 'ht_lists.pkl', 'rb'))

if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
    print("> Local expansion evaluation")
    ev.local_nhe_evaluation(ht_lists, emb_sentences_test, model, c.MAX_EXPANSION_ITERATIONS)
else:
    print("> Global expansion evaluation")
    ev.global_nhe_evaluation(ht_lists, emb_sentences_test, model, c.MAX_EXPANSION_ITERATIONS)

print("> Tweet polarization evaluation")
ev.global_nhe_polarization_evaluation(model, c.MAX_EXPANSION_ITERATIONS)
