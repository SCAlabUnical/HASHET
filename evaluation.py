import numpy as np

NEUTRAL = -1
AMBIGUOUS = -2


def recall_score(predicted, original):
    countPredicted = 0
    for i in range(0, len(predicted)):
        for j in range(0, len(original)):
            if (predicted[i][0] == original[j]):
                countPredicted += 1
    return (countPredicted / len(original))


def global_nhe_evaluation(hashtags_test, sentences_test, model, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on on each expansion iteration.
        Use global nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: sentences_test: input sentences for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    predicted_hashtag_list = model.predict_top_k_hashtags(sentences_test, 50)

    for n in range(0, max_iterations):
        print('RECALL VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.global_nhe(hashtags_test, predicted_hashtag_list, n)
        compute_scores(hashtags_test, predicted_hashtag_list_bounded)
        print()


def local_nhe_evaluation(hashtags_test, sentences_test, model, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on on each expansion iteration.
        Use local nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: sentences_test: input sentences for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    predicted_hashtag_list = model.predict_top_k_hashtags(sentences_test, 50)

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
