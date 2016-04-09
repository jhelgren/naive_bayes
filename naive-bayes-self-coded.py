# Jason Helgren
# jhelgren@gmail.com
# MSAN 593, hw 6


import argparse
import os
import random
import string
import glob
import collections
import itertools

from math import log
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")



def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def list_directories(dir):
    """ Lists the names of all directories in a given directory.

    In this application, the directory names will be the movie review
    sentiments ("pos" or "neg").

    Args:
        dir: a directory in the current working directory

    Returns:
        A list of strings. Each element of the list is a directory name.
    """

    return [name for name in os.listdir(dir) if
            os.path.isdir(os.path.join(dir, name))]


def list_files(dir, subdirs):
    """ Lists files in a given path.

    In this application, the files correspond to movie reviews.

    Args:
        dir: a directory
        subdirs: a list of strings with subdirectory names.

    Returns:
        A dictionary mapping subdirectory names to the files found within each
        subdirectory.
    """
    files = dict()
    for s in subdirs:
        paths = glob.glob(os.path.join(dir, s, '*.txt'))
        files[s] = [os.path.basename(path) for path in paths]
    return files


def get_file_text(name, dir, subdir):
    """ Read a text file and return the contents as a string.

    Read the contents of a text file. Remove punctuation and stop words and
    return the processed text as a string.

    Args:
        name: the file name.
        dir: a directory in the current working directory.
        subdir: a subdirectory in dir.

    Returns:
        A string.
    """
    f = open(os.path.join(dir, subdir, name))
    text = f.read()
    f.close()
    text = text.translate(string.maketrans("",""), string.punctuation).lower()
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text


def files_to_text(dir, files_dict):
    """ Read text from a dictionary of file names.

    Given a directory and a dictionary mapping subdirectories to file names,
    read the contents of each file. Create a new dictionary mapping
    subdirectories to lists of strings.

    In this application the subdirectories represent sentiments and the files
    contain movie reviews.

    Args:
        dir: a directory in the current working directory.
        files_dict: a dictionary mapping subdirectories to file names.

    Returns:
        A dictionary mapping subdirectories to lists of strings.
    """
    texts_dict = dict()
    for k, v in files_dict.iteritems():
        texts_dict[k] = list()
        for file in v:
            texts_dict[k].append(get_file_text(file, dir, k))
    return texts_dict


def divide_list(big_list, num_groups):
    """ Randomly divide a list into a specified number of sublists.

    Args:
        big_list: a list to be divided.
        num_groups: the desired number of sublists.

    Returns:
        A list of lists. The number of elements in the outer lists will be
        equal to to num_groups.
    """
    seed = 10
    random.seed(seed)
    random.shuffle(big_list)
    return [big_list[i::num_groups] for i in xrange(num_groups)]


def divide_dict(d, num_groups):
    """ Given a dictionary that keys to lists, divide the lists into sublists.

    This function modifies the provided dictionary.

    Args:
        d: a dictionary mapping keys to lists.
        num_groups: the desired number of groups to divide each list.
    """
    for k, v in d.iteritems():
        d[k] = divide_list(v, num_groups)


def make_lookup_table(d):
    """ Create a naive Bayes based probability lookup table.

    Given a dictionary mapping a classifier to a lists of strings, calculate
    probabilities for each word given each classifier.

    Args:
        d: a dictionary mapping classifiers (e.g. "pos" and "neg") to a list of
        strings. The strings represent movie reviews.
    Returns:
        A dictionary mapping each classifier to another dictionary, which then
        maps words to their associated conditional probabilities.

    """
    all_unique_words = set()
    word_counts = dict()
    counts_per_sentiment = dict()
    probs = dict()
    sentiments = d.keys()


    for sentiment, texts in d.iteritems():
        words = ' '.join(texts).split()
        counts_per_sentiment[sentiment] = len(words)
        word_counts[sentiment] = collections.Counter(words)
        all_unique_words = all_unique_words.union(words)

    num_unique_words = len(all_unique_words)

    for sentiment in sentiments:
        probs[sentiment] = dict()
        prob_sum = 1
        for word in all_unique_words:
            word_count = word_counts[sentiment][word]
            prob = float((word_count + 1)) / (num_unique_words + counts_per_sentiment[sentiment] + 1)
            prob_sum -= prob
            probs[sentiment][word] = prob
        probs[sentiment]['unk'] = prob_sum

    return probs


def calc_sentiment_probs(d):
    """ Find the proportion of each sentiment.

    Given a dictionary mapping sentiments to lists of strings, find the
    proportion of files in each sentiment. For instance, if there are two
    sentiments and each has 10 files, the returned dictionary would indicate
    would map each of the two sentiments to 0.5.

    Args:
        d: a dictionary mapping sentiments to lists of strings.

    Returns:
        A dictionary mapping sentiments to the proportion in which they occur.
    """
    sentiment_probs = dict()
    total_reviews = 0
    for sentiment, reviews in d.iteritems():
        sentiment_probs[sentiment] = len(reviews)
        total_reviews += len(reviews)
    for sentiment, probs in sentiment_probs.iteritems():
        sentiment_probs[sentiment] = float(sentiment_probs[sentiment]) / total_reviews
    return sentiment_probs


def classify_review(review, prob_lookup, sentiment_probs):
    """ Classify a review using naive Bayes.

    Given a movie review use a naive Bayes lookup table to assign a sentiment.

    Args:
        review: a string representing a movie review.
        prob_lookup: a dictionary representing a naive Bayes lookup table.
        sentiment_probs: a dictionary with the probability of each sentiment.

    Returns:
        A string containing the classification of the review.
    """
    probs = dict()
    for sentiment, prob in sentiment_probs.iteritems():
        probs[sentiment] = log(prob)

    for word in review.split():
        for sentiment, word_probs in prob_lookup.iteritems():
            if word in word_probs:
                probs[sentiment] += log(prob_lookup[sentiment][word])
            else:
                probs[sentiment] += log(prob_lookup[sentiment]['unk'])

    high_prob = float('-inf')
    decision = None

    for sentiment, prob in probs.iteritems():
        if prob > high_prob:
            high_prob = prob
            decision = sentiment

    return decision


def classify_reviews(d, prob_lookup, sentiment_probs):
    """ Classify a dictionary of movie reviews.

    Given a dictionary mapping a known sentiment to a list of movie reviews,
    analyze the text of each review to determine a sentiment, which may or may
    not match the known sentiment.

    Args:
        d: a dictionary mapping sentiments to lists of strings containing
            movie reviews.
        prob_lookup: a dictionary containing a conditional probability lookup
            table.
        sentiment_probs: a ditionary containing the probability of each
            sentiment.

    Returns:
        A dictionary mapping sentiments to lists of classifications.

    """
    decisions = dict()
    for sentiment, reviews in d.iteritems():
        decisions[sentiment] = []
        for review in reviews:
            decisions[sentiment].append(classify_review(review, prob_lookup, sentiment_probs))
    return decisions


def evaluate_decisions(decision_dict):
    """ Evaluate how well a calculated classification matches its expected
        value.

    Args:
        decision_dict: a dictionary mapping sentiments to lists of
            classifications.

    Returns:
        The number of correct classifications for each sentiment.
    """
    evals = dict()
    for sentiment, decisions in decision_dict.iteritems():
        correct = 0
        for decision in decisions:
            if sentiment == decision:
                correct += 1
        evals[sentiment] = correct, len(decisions)
    return evals


def calculate_accuracy(correct_counts):
    total_correct, total = 0, 0
    for sentiment, counts in correct_counts.iteritems():
        total_correct += counts[0]
        total += counts[1]
    return float(total_correct) / total


def main():
    groups = 3
    args = parseArgument()
    directory = args['d'][0]

    file_list = list_files(directory, list_directories(directory))
    review_text = files_to_text(directory, file_list)
    divide_dict(review_text, groups)

    test_set = dict()
    training_set = dict()

    accuracy_sum = 0

    for i in range(0, groups):
        print 'iteration %d:' %(i + 1)

        for sentiment, texts in review_text.iteritems():
            temp_texts = list(texts)
            test_set[sentiment] = temp_texts.pop(i)
            training_set[sentiment] = list(itertools.chain.from_iterable(temp_texts))

        p = make_lookup_table(training_set)
        s_p = calc_sentiment_probs(training_set)
        decisions = classify_reviews(test_set, p, s_p)
        correct_counts = evaluate_decisions(decisions)

        for sentiment in review_text:
            print 'num_' + sentiment + '_test_docs:', len(test_set[sentiment])
            print 'num_' + sentiment + '_training_docs:', len(training_set[sentiment])
            print 'num_' + sentiment + '_correct_docs:', correct_counts[sentiment][0]

        accuracy = calculate_accuracy(correct_counts)
        accuracy_sum += accuracy
        print 'accuracy:', "{:.1%}".format(accuracy)

    print 'ave_accuracy:', "{:.1%}".format(accuracy_sum / groups)


main()
