## Naive Bayes for Sentiment Analysis

This code includes two examples of using naive Bayes for sentiment analysis, using the [polarity_dataset_v2.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) from Bo Pang and Lillian Lee's [movie review data](https://www.cs.cornell.edu/people/pabo/movie-review-data/). These examples were coded during two course projects in the University of San Francisco's [Master of Science in Analytics](https://www.usfca.edu/arts-sciences/graduate-programs/analytics) program.

Both example assume that labels as assigned according to the directory containing the text file, so positive samples might be in a directory named "pos" and negative samples might be in a directory called "neg". Specify the parent directory on the command line, e.g. if the parent directory is called "txt_sentoken" as in the linked data sets, run the first example with "naive-bayes-self-coded.py -d txt_sentoken".

The first example (naive-bayes-self-coded.py) is entirely self coded, meaning it relies only on the Python Standard Library. It reads the movie review text, counts words in each review, and builds a naive Bayes classifier. With three-fold cross validation the classifier demonstrates an average accuracy of about 80%.

The second example demonstrates the same functionality and accuracy, but uses scikit-learn. This latter example is much faster and more compact.
