#!/usr/bin/env python

import multiprocessing

import gensim
import numpy as np
from gensim.models import Word2Vec

import constants as c

URL_REGEX = r"@\w*|https?:?\/?\/?[\w.\/]*|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
not_lemmatize = []
LATENT_SPACE_DIM = 150
WINDOW_SIZE = 5
REMOVE_WORDS = ['rt', 'ht', 'htt', 'https', 'http', 'https t']
INPUT_PATH = "input"


def create_and_train_Word2Vec_model(sentences, mincount=5):
    """
        Create and stores Word2Vec model

        :param1 sentences.
        :param2 mincount.
        :return: Word2Vec model
    """

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

    w2v_model = Word2Vec(min_count=mincount, window=WINDOW_SIZE, size=LATENT_SPACE_DIM, sample=1e-5, alpha=0.01,
                         min_alpha=0.001, negative=10,
                         workers=cores - 1)
    w2v_model.build_vocab(sentences, progress_per=10)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
    w2v_model.save(c.W2V_SAVE_FILE_NAME)

    return w2v_model


def load_Word2Vec_model():
    """
        Load Word2Vec model and return it

        :returns model: Word2Vec model.
    """
    model = Word2Vec.load(c.W2V_SAVE_FILE_NAME)
    return model


def retain_hts(top_emb):
    top_emb_hts = []
    for tuple in top_emb:
        if '#' in tuple[0]:
            if '_' in tuple[0]:  # bigrams
                split = tuple[0].split(sep='_')
                for h in split:
                    if '#' in h:
                        top_emb_hts.append((h, tuple[1]))
            else:
                top_emb_hts.append(tuple)
    return top_emb_hts


def tweet_arith_embedding(we_model, tweet):
    tw_list = tweet.split()
    den = 0
    sent_embedding = np.zeros(LATENT_SPACE_DIM)
    for word in tw_list:
        try:
            emb = we_model.wv[word]
            den += 1
            sent_embedding += emb
        except:
            pass
    return None if den == 0 else sent_embedding / den


def check_support():
    support = gensim.models.doc2vec.FAST_VERSION
    if support > -1:
        print("Cython supported: Gensim fast version is running")
    else:
        print("Cython not supported: no speed up is performed")
