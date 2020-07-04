import numpy as np

import constants as c
import dataset_handler as dh
import model as mlp
import word_embedding_model as emb

NEUTRAL = -1
AMBIGUOUS = -2


def prepare_data(test_embedding, test_hts_lists):
    test = dh.load(c.TEST_CORPUS)
    hashtags_test = []
    words_test = []
    tweet_emb_list = None
    if test_embedding is not None and test_hts_lists is not None:
        print('using input parameters for evaluation')
        tweet_emb_list = test_embedding
        hashtags_test = test_hts_lists
    else:
        print('evaluation from corpus')
        for tweet in test:
            ht_list = dh.hashtags_list(tweet)

            if len(ht_list) > 0:
                hashtags_test.append(ht_list)
                words_test.append(tweet)
        tweet_emb_list, hashtags_test = dh.get_sentence_embeddings_for_testing(words_test, hashtags_test)

    return tweet_emb_list, hashtags_test


def recall_score(predicted, original):
    countPredicted = 0
    for i in range(0, len(predicted)):
        for j in range(0, len(original)):
            if (predicted[i][0] == original[j]):
                countPredicted += 1
    return (countPredicted / len(original))


def global_nhe_evaluation(hashtags_test, emb_sentences_test, model, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on on each expansion iteration.
        Use global nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: emb_sentences_test: input sentence embeddings for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    predicted_hashtag_list = model.predict_top_k_hashtags(emb_sentences_test, 50)

    for n in range(0, max_iterations):
        print('RECALL VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.global_nhe(hashtags_test, predicted_hashtag_list, n)
        compute_scores(hashtags_test, predicted_hashtag_list_bounded)
        print()


def local_nhe_evaluation(hashtags_test, emb_sentences_test, model, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on on each expansion iteration.
        Use local nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: emb_sentences_test: input sentence embeddings for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    predicted_hashtag_list = model.predict_top_k_hashtags(emb_sentences_test, 50)

    for n in range(0, max_iterations):
        print('VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.local_nhe(hashtags_test, predicted_hashtag_list, n)
        compute_scores(hashtags_test, predicted_hashtag_list_bounded)
        print()


def compute_scores(hashtags_test, predicted_hashtag_list, count=5):
    numberTweetsAnalyzed = [0, 0, 0, 0, 0, 0]
    recallSumPerTweet = [0, 0, 0, 0, 0, 0]
    precisionSumPerTweet = [0, 0, 0, 0, 0, 0]
    for indexInputTweet in range(0, len(predicted_hashtag_list)):
        real_hashtags = hashtags_test[indexInputTweet]
        k = len(real_hashtags)
        if k > count:
            numberTweetsAnalyzed[count] += 1
        else:
            numberTweetsAnalyzed[k - 1] += 1
        predicted_hts = predicted_hashtag_list[indexInputTweet]

        singleRecall = recall_score(predicted_hts, real_hashtags)
        singlePrecision = singleRecall * len(real_hashtags) / len(predicted_hts)
        if k > count:
            recallSumPerTweet[count] += singleRecall
            precisionSumPerTweet[count] += singlePrecision
        else:
            recallSumPerTweet[k - 1] += singleRecall
            precisionSumPerTweet[k - 1] += singlePrecision

    print("N° OF TWEETS PER HASHTAG NUMBER")
    print(numberTweetsAnalyzed)

    recallPerHashtag = []
    precisionPerHashtag = []
    f1scorePerHashtag = []
    eps = 0.00000000000001
    for i in range(0, 6):
        if numberTweetsAnalyzed[i] == 0:
            numberTweetsAnalyzed[i] = eps
        recall_hashtag_number = recallSumPerTweet[i] / numberTweetsAnalyzed[i]
        precision_hashtag_number = precisionSumPerTweet[i] / numberTweetsAnalyzed[i]
        recallPerHashtag.append(round(recall_hashtag_number, 3))
        precisionPerHashtag.append(round(precision_hashtag_number, 3))
        f1den = recall_hashtag_number + precision_hashtag_number
        if f1den == 0:
            f1den = eps
        f1scorePerHashtag.append(round(
            2 * recall_hashtag_number * precision_hashtag_number / f1den,
            3))
    print("RECALL PER HASHTAG NUMBER")
    print(recallPerHashtag)
    print("PRECISION PER HASHTAG NUMBER")
    print(precisionPerHashtag)
    print("F1 SCORE PER HASHTAG NUMBER")
    print(f1scorePerHashtag)

    totalTweets = 0
    numberTweetsAnalyzed1 = numberTweetsAnalyzed[:len(numberTweetsAnalyzed) - 1]
    recallPerHashtag = recallPerHashtag[:len(recallPerHashtag) - 1]
    precisionPerHashtag = precisionPerHashtag[:len(precisionPerHashtag) - 1]
    f1scorePerHashtag = f1scorePerHashtag[:len(f1scorePerHashtag) - 1]

    for i in numberTweetsAnalyzed1:
        totalTweets += i
    weightedRecall = np.sum([(r * i) / totalTweets for r, i in zip(recallPerHashtag, numberTweetsAnalyzed1)])
    weightedPrecision = np.sum([(p * i) / totalTweets for p, i in zip(precisionPerHashtag, numberTweetsAnalyzed1)])
    weightedF1score = np.sum([(f * i) / totalTweets for f, i in zip(f1scorePerHashtag, numberTweetsAnalyzed1)])

    print("AVERAGE RECALL")
    print(np.round(weightedRecall, 3))
    print("AVERAGE PRECISION")
    print(np.round(weightedPrecision, 3))
    print("AVERAGE F1 SCORE")
    print(np.round(weightedF1score, 3))
    print()


# -------------------------------------------------------------------------------------------------------


# calculates the polarization from a list of hashtags, a polarized value is greater than zero
def polarization(ht_list):
    hillary = False
    trump = False
    for h in ht_list:
        if h[1:] in c.KEYS[c.HILLARY]:
            hillary = True
        elif h[1:] in c.KEYS[c.TRUMP]:
            trump = True
    if hillary and not trump:
        return c.HILLARY
    elif trump and not hillary:
        return c.TRUMP
    elif trump and hillary:
        return AMBIGUOUS
    return NEUTRAL


def hashtags_only(tuple_list):
    h_list = [t[0] for t in tuple_list]
    return h_list


def prepare_polarized_data():
    test = dh.load(c.TEST_CORPUS)
    print('Test tweets: ' + str(len(test)))
    ht_test_list = []
    words_test = []

    for tweet in test:
        ht_list = dh.hashtags_list(tweet)

        pol = polarization(ht_list)
        if len(ht_list) > 0 and pol >= 0:
            ht_test_list.append(ht_list)
            words_test.append(" ".join(tweet))

    tweet_emb_list, ht_test_list = dh.get_sentence_embeddings_for_testing(words_test, ht_test_list)

    return tweet_emb_list, ht_test_list


def global_nhe_polarization_evaluation(model, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on on each expansion iteration.
        Use global nearest hashtag expansion.

        param1: model: mlp model.
        param2: max_iterations: max number of expansions.
    """
    tweet_emb_list, ht_test_list = prepare_polarized_data()
    predicted_hashtag_list = mlp.predict_top_k_hashtags(tweet_emb_list, 50)
    print('Polarized tweets: ' + str(len(ht_test_list)))

    for n in range(0, max_iterations):
        print('POLARIZATION VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.global_nhe(ht_test_list, predicted_hashtag_list, n)
        evaluate_polarization(ht_test_list, predicted_hashtag_list_bounded)
        print()


def evaluate_polarization(ht_test_list, predicted_h_list):
    """
        Evaluates polarization case study.

        param1: ht_test_list: original tweets hashtags.
        param2: predicted_h_list: predicted tweets hashtags
    """
    correct = 0
    neutral = 0
    ambiguous = 0
    incorrect = 0
    den = len(ht_test_list)
    for ht_test, pred_list in zip(ht_test_list, predicted_h_list):
        predicted_polarization = polarization(hashtags_only(pred_list))

        target_pol = polarization(ht_test)

        if predicted_polarization == target_pol:
            correct += 1
        elif predicted_polarization == NEUTRAL:
            neutral += 1
        elif predicted_polarization == AMBIGUOUS:
            ambiguous += 1
        else:
            incorrect += 1

    print('Correct polarization score: ')
    print(round(correct / den, 4))
    print('Neutral polarization score: ')
    print(round(neutral / den, 4))
    print('Ambiguous polarization score: ')
    print(round(ambiguous / den, 4))
    print('Incorrect polarization score: ')
    print(round(incorrect / den, 4))
