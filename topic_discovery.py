import numpy as np

import constants as c
import dataset_handler as dh
import word_embedding_model as emb
import pickle

NEUTRAL = -1
AMBIGUOUS = -2

topic_matrix = c.TOPICS


# calculates the polarization from a list of hashtags, a polarized value is greater than zero
def classify(ht_list):
    matches = []
    for h in ht_list:
        for i in range(0, len(topic_matrix)):
            if h in topic_matrix[i]:
                if i not in matches:
                    matches.append(i)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return AMBIGUOUS
    return NEUTRAL


def evaluate_topic_discovery(model, input_file, max_expansions):
    pol_tweets, ht_test_lists = prepare_data(input_file)
    pol_tweets = np.array(pol_tweets)
    predicted_hashtag_list = model.predict_top_k_hashtags(pol_tweets, 50)
    correct = 0
    neutral = 0
    ambiguous = 0
    incorrect = 0
    den = len(pol_tweets)
    n_max = max_expansions
    for i in range(len(pol_tweets)):
        n = 0
        k = len(ht_test_lists[i])
        predicted_hts = hashtags_only(predicted_hashtag_list[i][:k + n])
        predicted_polarization = classify(predicted_hts)
        target_pol = classify(ht_test_lists[i])
        if predicted_polarization == NEUTRAL:
            while n < n_max:
                n += 1
                predicted_hts = hashtags_only(predicted_hashtag_list[i][:k + n])
                predicted_polarization = classify(predicted_hts)
                if not predicted_polarization == NEUTRAL:
                    break
        if predicted_polarization == NEUTRAL:
            neutral += 1
        elif predicted_polarization == target_pol:
            correct += 1
        elif predicted_polarization == AMBIGUOUS:
            ambiguous += 1
        else:
            incorrect += 1

    print('correct score: ')
    print(correct / den)
    print('neutral score: ')
    print(neutral / den)
    print('ambiguous score: ')
    print(ambiguous / den)
    print('incorrect score: ')
    print(incorrect / den)


def prepare_data(input_file):
    test = dh.load(input_file)
    print('Test tweets: ' + str(len(test)))
    ht_test_list = []
    tweet_list = []
    w2v_model = emb.load_Word2Vec_model()
    for tweet in test:
        ht_list = dh.hashtags_list(tweet, w2v_model)
        pol = classify(ht_list)
        if len(ht_list) > 0 and pol >= 0:
            ht_test_list.append(ht_list)
            tweet_list.append(" ".join(tweet))
    print('Topic tweets: ' + str(len(tweet_list)))
    return tweet_list, ht_test_list


def hashtags_only(tuple_list):
    h_list = [t[0] for t in tuple_list]
    return h_list
