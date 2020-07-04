#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.models import model_from_json

import constants as c
import constants as ctx
import word_embedding_model
import dataset_handler as dh

SCALE_1 = 2 / 3


def cosine_distance(y_true, y_pred):
    return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0)


def create_and_train_MLP(emb_sentences_train, emb_sentences_test, targets_train, targets_test):
    """
        Create and train multi level perceptron with Keras API and save it.

        :param1 emb_sentences_train: train sentence embeddings.
        :param2 emb_sentences_test: test sentence embeddings.
        :param3 targets_train: train target embeddings.
        :param4 targets_test: test target embeddings.
    """
    # MLP
    model = Sequential()
    n_input = len(emb_sentences_train[0])

    model.add(Dense(int(n_input), input_dim=n_input))
    model.add(Activation('relu'))
    model.add(Dense(int(n_input * SCALE_1)))
    model.add(Activation('relu'))
    model.add(Dense(int(n_input * SCALE_1 * SCALE_1)))
    model.add(Activation('relu'))
    model.add(Dense(len(targets_train[0]), activation='linear'))

    # compile model
    model.compile(loss=cosine_distance, optimizer='adam', metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=ctx.LOG_LEVEL, patience=ctx.PATIENCE)

    best_weights_file = c.MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=ctx.ONE_LINE_PER_EPOCH,
                         save_best_only=True)
    # train model testing it on each epoch
    model.fit(emb_sentences_train, targets_train, validation_data=(emb_sentences_test, targets_test),
              batch_size=ctx.BATCH_SIZE, epochs=ctx.MAX_EPOCHS, verbose=ctx.ONE_LINE_PER_EPOCH, callbacks=[es, mc])
    # serialize model to JSON
    model_json = model.to_json()
    with open(c.MODEL_JSON_FILE_NAME, "w") as json_file:
        json_file.write(model_json)
    print("model saved")


def predict_top_k_hashtags(embeddings_list, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 embeddings_list: sentence embeddings.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """
    # Model reconstruction from JSON file
    with open(c.MODEL_JSON_FILE_NAME, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(c.MODEL_WEIGHTS_FILE_NAME)

    # make probability predictions with the model
    h_list = model.predict(embeddings_list)

    h_list = [np.reshape(h_vect, (len(h_vect),)) for h_vect in h_list]

    emb_model = word_embedding_model.load_Word2Vec_model()
    top_n_words = 1000
    result = [word_embedding_model.retain_hts(emb_model.wv.similar_by_vector(h_vect, topn=top_n_words))[:k] for h_vect
              in h_list]

    return result


def predict_hashtags_and_store_results(test_file, embeddings_list):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 test_file: train sentence embeddings.
        :param2 embeddings_list: test sentence embeddings.
    """
    test_corpus = dh.load(test_file)
    predictions = predict_top_k_hashtags(embeddings_list, 100)
    with open("results.txt", "w", encoding="UTF-8") as out:
        for tweet, hashtags in zip(test_corpus, predictions):
            original_hashtags = dh.hashtags_list(tweet)
            needed_predicted_hashtags = hashtags[:len(original_hashtags)]
            out.write("Tweet: " + " ".join(tweet) + "\nOriginal hashtags: " + str(
                original_hashtags) + "\tPredicted hashtags: " +
                      str(needed_predicted_hashtags) + "\n\n")


def global_nhe(hashtags_test, predicted_hashtag_list, n):
    result = []
    for indexInputTweet in range(0, len(predicted_hashtag_list)):
        real_hashtags = hashtags_test[indexInputTweet]
        k = len(real_hashtags)
        predicted_hts = predicted_hashtag_list[indexInputTweet][:k + n]
        result.append(predicted_hts)
    return result


def local_nhe(hashtags_test, predicted_hashtag_list, n):
    result = []
    emb_model = word_embedding_model.load_Word2Vec_model()
    top_n_words = 1000
    for indexInputTweet in range(0, len(predicted_hashtag_list)):
        real_hashtags = hashtags_test[indexInputTweet]
        k = len(real_hashtags)
        predicted_ht_tuples = predicted_hashtag_list[indexInputTweet][:k]
        pred_hashtags = [tuple[0] for tuple in predicted_ht_tuples]
        already_seen = []  # for distinct hashtags in pred_list
        already_seen.extend(pred_hashtags)
        # add k nearest neighbors for each predicted hashtag
        if n > 0:
            for pred_h in pred_hashtags:
                nearest_neighbors = []
                try:
                    nearest_neighbors.extend(
                        word_embedding_model.retain_hts(emb_model.wv.most_similar(pred_h, topn=top_n_words))[:n])
                except:
                    pass
                for nn in nearest_neighbors:
                    if nn[0] not in already_seen:
                        predicted_ht_tuples.append(nn)
                        already_seen.append(nn[0])
        result.append(predicted_ht_tuples)
    return result
