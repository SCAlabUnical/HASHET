#!/usr/bin/env python

import constants as c
import word_embedding_model
import dataset_handler as dh

if c.SENTENCE_ENCODING_MODEL == c.GUSE:
    import model_GUSE as model
else:
    import model_BERT as model


def transfer_and_fine_tune(sentences_train, sentences_test, targets_train, targets_test):
    """
        Create and train multi level perceptron with Keras API and save it.

        :param1 sentences_train: train sentences.
        :param2 sentences_test: test sentences.
        :param3 targets_train: train target embeddings.
        :param4 targets_test: test target embeddings.
    """
    model.transfer_and_fine_tune(sentences_train, sentences_test, targets_train, targets_test)



def predict_top_k_hashtags(sentences, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 sentences: sentences.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """
    return model.predict_top_k_hashtags(sentences, k)


def predict_hashtags_and_store_results(test_file, sentences):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 test_file: train sentence embeddings.
        :param2 sentences: test sentence embeddings.
    """
    w2v_model = word_embedding_model.load_Word2Vec_model()
    test_corpus = dh.load(test_file)
    predictions = predict_top_k_hashtags(sentences, 100)
    with open("results.txt", "w", encoding="UTF-8") as out:
        for tweet, hashtags in zip(test_corpus, predictions):
            original_hashtags = dh.hashtags_list(tweet, w2v_model)
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
