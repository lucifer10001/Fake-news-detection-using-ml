# Fake-news-detection-using-ml


## Overview  
The topic of fake news detection on social media has recently attracted tremendous attention. The basic countermeasure of comparing websites against a list of labeled fake news sources is inflexible, and so a machine learning approach is desirable.  Our project aims to use Natural Language Processing to detect fake news directly, based on the text content of news articles. 

## Problem Definition
Develop a machine learning program to identify when a news source may be producing fake news. We aim to use a corpus of labeled real and fake new articles to build a classifier that can make decisions about information based on the content from the corpus. The model will focus on identifying fake news sources, based on multiple articles originating from a source.  Once a source is labeled as a producer of fake news, we can predict with high confidence that any future articles from that source will also be fake news.  Focusing on sources widens our article misclassification tolerance, because we will have multiple data points coming from each source.  

The intended application of the project is for use in applying visibility weights in social media.  Using weights produced by this model, social networks can make stories which are highly likely to be fake news less visible.


## Data Exploration
To begin, we should always take a quick look at the data and get a feel for its contents. To do so, use a Pandas DataFrame and check the shape, head and apply any necessary transformations.

## Extracting the training data

Now that the DataFrame looks closer to whatwe need, we want to separate the labels and set up training and test datasets. 

For this notebook,we decided to focus on using the longer article text. Because we knew we would be using bag-of-words and Term Frequency–Inverse Document Frequency (TF-IDF) to extract features, this seemed like a good choice. Using longer text will hopefully allow for distinct words and features for my real and fake news data.


## Building Vectorizer Classifiers

Now that we have our training and testing data, we can build our classifiers. To get a good idea if the words and tokens in the articles had a significant impact on whether the news was fake or real, we begin by using CountVectorizer and TfidfVectorizer.

we'll see the example has a max threshhold set at .7 for the TF-IDF vectorizer tfidf_vectorizer using the max_df argument. This removes words which appear in more than 70% of the articles. Also, the built-in stop_words parameter will remove English stop words from the data before making vectors.

There are many more parameters avialable and we can read all about them in the scikit-learn documentation for TfidfVectorizer and CountVectorizer.

## Comparing Models

Now it's time to train and test our models.

Here, we'll begin with an NLP favorite, MultinomialNB. We can use this to compare TF-IDF versus bag-of-words. Our intuition was that bag-of-words (CountVectorizer) would perform better with this model..

We personally find Confusion Matrices easier to compare and read, so we used the scikit-learn documentation to build some easily-readable confusion matrices (thanks open source!). A confusion matrix shows the proper labels on the main diagonal (top left to bottom right). The other cells show the incorrect labels, often referred to as false positives or false negatives. Depending on our problem, one of these might be more significant. For example, for the fake news problem, is it more important that we don't label real news articles as fake news? If so, we might want to eventually weight our accuracy score to better reflect this concern.

Other than Confusion Matrices, scikit-learn comes with many ways to visualize and compare our models. One popular way is to use a ROC Curve. There are many other ways to evaluate our model available in the scikit-learn metrics modul And indeed, with absolutely no parameter tuning, our count vectorized training set count_train is visibly outperforming our TF-IDF vectors!

## Training models


we will compare the following models (and training data):

 multinomialNB with counts (sgd_count_clf)
 multinomialNB with tf-idf (mn_tfidf_clf)
 passive aggressive with tf-idf (pa_tfidf_clf)
 linear svc with tf-idf (svc_tfidf_clf)
 linear sgd with tf-idf (sgd_tfidf_clf)

For speed and clarity, we are primarily not doing parameter tuning, although this could be added as a step (perhaps in a scikit-learn Pipeline).


## Testing Linear Models

There are a lot of great write-ups about how linear models work well with TF-IDF vectorizers (take a look at word2vec for classification, SVM reference in scikit-learn text analysis, and many more).


## Conclusion

As expected from the outset, defining fake news with simple bag-of-words or TF-IDF vectors is an oversimplified approach. Especially with a multilingual dataset full of noisy tokens. If we hadn't taken a look at what our model had actually learned, we might have thought the model learned something meaningful. So, remember: always introspect our models
The bag-of-words and TF-IDF vectors didn't do much to determine meaningful features to classify fake or real news. this problem is a lot harder than simple text classification.

That said, we did learn a few things. Namely, that linear models handle noise in this case better than the Naive Bayes multinomial classifier did. Also, finding a good dataset that has been scraped from the web and tagged for this problem would likely be a great help, and worth more of our time than parameter tuning on a clearly noisy and error prone dataset.


