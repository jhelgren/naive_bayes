# Jason Helgren
# Naive Bayes Demonstration using scikit-learn

import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
import numpy as np

def parseArgument():
    """
    Parse an arguement from the command line that indicates the directory
    containing the data.
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    directory = args['d'][0]
    return directory

def get_reviews(data_path):
    """
    Use sklearn's load_files function to read labeled text files. Note that OSX
    .DS_Store files seem to wreak havoc with load_files, so remove them.

    Args:
        data_path: the directory with the labeled text files. Labeles are
        determined by the subdirectory name containing the files.

    Returns: a Bunch object, which is a dictionary like object the data and
    labels.
    """
    full_data_path = os.path.join(os.getcwd(), data_path)

    for root, dirs, files in os.walk(full_data_path):
        try:
            os.remove(os.path.join(root, '.DS_Store'))
        except OSError:
            pass

    reviews = load_files(full_data_path, load_content = True,
                         encoding = 'utf-8', decode_error = 'ignore')
    return reviews


def main():
    directory = parseArgument()
    reviews = get_reviews(directory)
    text = reviews['data']
    y = reviews['target']

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(text)
    k_fold = cross_validation.KFold(X.shape[0], shuffle = True)
    clf = MultinomialNB(alpha = 1.0)

    scores = list()
    for k, (train, test) in enumerate(k_fold):
        clf.fit(X[train], y[train])
        accuracy = clf.score(X[test], y[test])
        scores.append(accuracy)
        print "Iteration {} accuracy: {:.1%}".format(k + 1, accuracy)
    print "Average accuracy: {:.1%}".format(np.mean(scores))

main()
