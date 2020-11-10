import xlrd 
import numpy as np
import re
import random
import nltk

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from gensim.sklearn_api import W2VTransformer
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors

def clean_text(X):
    cleaned_X = []
    for element in X:
        words_only = re.sub("[^a-zA-Z]", " ", element)

        lower_words = words_only.lower().split()

        #nltk.download("stopwords")
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in lower_words if not w in stops]

        porter = PorterStemmer()
        stemmed_only = [porter.stem(w) for w in lower_words]
        stemmed_words = [porter.stem(w) for w in meaningful_words]

        joined_words = (" ".join(lower_words))
        #joined_words = (" ".join(meaningful_words))
        #joined_words = (" ".join(stemmed_only))
        #joined_words = (" ".join(stemmed_words))

        cleaned_X.append(joined_words)
    return cleaned_X

def centroid_vectors(X, vocab, size, mat1, mat2, dense):
    vec1 = []
    vec2 = []
    for sentence in X:
        if(len(sentence) != 0):
            sum1 = np.zeros(size)
            sum2 = np.zeros(size)
            for word in sentence.split():
                if word in vocab:
                    if dense:
                        sum1 += mat1[word]
                    else:
                        sum1 += mat1[vocab.index(word)]
                        sum2 += mat2[vocab.index(word)]
            vec1.append(np.true_divide(sum1,len(sentence)))
            if not dense:
                vec2.append(np.true_divide(sum2, len(sentence)))
        else:
            vec1.append(np.zeros(size))
            vec2.append(np.zeros(size))

    vec1 = np.array(vec1)
    if not dense:
        vec2 = np.array(vec2)
        return vec1, vec2
    else:
        return vec1

def co_occurance(X, vocab, window, tfidf):
    cooc = np.zeros((len(vocab), len(vocab)))
    for sentence in X:
        if(len(sentence) != 0):
            words = sentence.split()
            for i in range(len(words)):
                lower = i- window
                upper = i + window
                if lower < 0:
                    lower = 0
                if upper >= len(words):
                    upper = len(words)-1
                next_word = words[lower : upper]
                for j in range(len(next_word)):
                    if (words[i] in vocab) and (next_word[j] in vocab) and (words[i] != next_word[j]):
                        cooc[vocab.index(words[i]), vocab.index(next_word[j])] += 1
                        cooc[vocab.index(next_word[j]), vocab.index(words[i])] += 1

    tfidf_cooc = tfidf.fit_transform(cooc).toarray()
    return cooc, tfidf_cooc

def train_maj_classifier(X, Y):
    words = []
    counts = []

    idx = 0
    count = 0
    for sentence in X:
        if Y[idx] == 1:
            count = 1
        elif Y[idx] == 0:
            count = -1
        for word in sentence.split():
            if word in words:
                counts[words.index(word)] += count
            else:
                words.append(word)
                counts.append(0)
        idx += 1

    return words, counts

def train_bag_of_words(X):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, ngram_range=(1,1), max_features=100000)
    train_data_features = vectorizer.fit_transform(X).toarray()

    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    vocab = vectorizer.get_feature_names()
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf

def alt_sparse_vecs(X, vocab, window_size):
    tfidf = TfidfTransformer()
    df, tfidf_sv = co_occurance(X, vocab, window_size, tfidf)

    sparse_vecs, tfidf_sparse_vecs = centroid_vectors(X, vocab, len(vocab), df, tfidf_sv, False)

    return df, sparse_vecs, tfidf_sparse_vecs, tfidf


def train_dense_vecs(X, dense_model):
    size = len(dense_model['test'])
    dense_vecs = centroid_vectors(X, dense_model.vocab, size, dense_model, dense_model, True)
 
    tfidf = TfidfTransformer()
    tfidf_dense_vecs = tfidf.fit_transform(dense_vecs).toarray()

    return dense_vecs, tfidf_dense_vecs, tfidf

# CONFIGURATION VARS
cross_valid = 2
window = 5
MAJ=False
BOW=True
SV=True
DV=False

# READ FILES AND SET UP MODELS
file_location = 'D:/2020-2021 Academic Year\Fall 2020\CS 2731 - Introduction to Natural Language Processing\Hwk 3\ML-Vector-Semantics-Classifier/SFUcorpus.xlsx'
wb = xlrd.open_workbook(file_location)
sheet = wb.sheet_by_index(0)

X = list(sheet.col_values(5))
X.pop(0)
strY = list(sheet.col_values(6))
Y = list()
for label in strY:
    if label == 'no':
        Y.append(0)  
    if label == 'yes':
        Y.append(1)

ml_model = LogisticRegression(C=1, random_state=0)
sparse_model = LogisticRegression(C=0.5, max_iter=10000)
filename = 'GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
if(DV):
    dense_model = KeyedVectors.load_word2vec_format(filename, binary=True)

maj_acc = 0
bow_acc = 0
bow_tfidf_acc = 0
sparse_acc = 0
sparse_tfidf_acc = 0
dense_acc = 0
dense_tfidf_acc = 0

for iter in range(cross_valid):
    seed = random.randrange(100000)
    print('Iter',iter,'seed is',seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/(cross_valid), random_state=seed)

    X_train_clean = clean_text(X_train)
    X_test_clean = clean_text(X_test)

    #################################################### MAJORITY CLASS ####################################################
    if MAJ:
        words, counts = train_maj_classifier(X_train_clean, Y_train)

        idx = 0
        correct_Y = 0
        for sentence in X_test_clean:
            curr_sum = 0
            for word in sentence.split():
                if word in words:
                    count = counts[words.index(word)]
                    if count > 0:
                        curr_sum += 1
                    elif count < 0:
                        curr_sum -= 1
            if (curr_sum > 0 and Y_test[idx] == 1):
                correct_Y += 1
            elif(curr_sum < 0 and Y_test[idx] == 0):
                correct_Y += 1
            idx += 1

        maj_acc += (100.0*correct_Y)/len(Y_test)
        #print('MAJ Accuracy = %.0f%%' %maj_acc)

    #################################################### BAG OF WORDS ######################################################
    if BOW:
        vectorizer, vocab, train_data_features, tfidf_features, tfidf = train_bag_of_words(X_train_clean)

        test_data_features = vectorizer.transform(X_test_clean).toarray()
        test_data_tfidf_features = tfidf.fit_transform(test_data_features).toarray()

        ml_model.fit(train_data_features, Y_train)
        predicted_Y = ml_model.predict(test_data_features)
        correct_Y = predicted_Y == Y_test
        bow_acc += np.mean(correct_Y) * 100
        #print('BOW Accuracy = %.0f%%' %bow_acc)

        ml_model.fit(tfidf_features, Y_train)
        predicted_Y = ml_model.predict(test_data_tfidf_features)
        correct_Y = predicted_Y == Y_test
        bow_tfidf_acc += np.mean(correct_Y) * 100
        #print('BOW TFIDF Accuracy = %.0f%%' %bow_tfidf_acc)
    print('Vocab size is',len(vocab))
    #################################################### SPARSE VECTORS ####################################################
    if SV:
        df, sparse_vecs, tfidf_sparse_vecs, tfidf = alt_sparse_vecs(X_train_clean, vocab, window)

        testc, tfidf_testc = co_occurance(X_test_clean, vocab, window, tfidf)

        test_sparse_vecs, test_tfidf_sparse_vecs = centroid_vectors(X_test_clean, vocab, len(vocab), testc, tfidf_testc, False)

        sparse_model.fit(sparse_vecs, Y_train)
        predicted_Y = sparse_model.predict(test_sparse_vecs)
        correct_Y = predicted_Y == Y_test
        sparse_acc += np.mean(correct_Y) * 100
        #print('SV Accuracy = %.0f%%' %sparse_acc)

        sparse_model.fit(tfidf_sparse_vecs, Y_train)
        predicted_Y = sparse_model.predict(test_tfidf_sparse_vecs)
        correct_Y = predicted_Y == Y_test
        sparse_tfidf_acc += np.mean(correct_Y) * 100
        #print('SV TFIDF Accuracy = %.0f%%' %sparse_tfidf_acc)

    #################################################### DENSE VECTORS ####################################################
    if DV:
        dense_vecs, tfidf_dense_vecs, tfidf = train_dense_vecs(X_train_clean, dense_model)
        
        size = len(dense_model['test'])
        test_dense_vecs = centroid_vectors(X_test_clean, dense_model.vocab, size, dense_model, dense_model, True)

        test_tfidf_dense_vecs = tfidf.fit_transform(test_dense_vecs).toarray()
        
        ml_model.fit(dense_vecs, Y_train)
        predicted_Y = ml_model.predict(test_dense_vecs)
        correct_Y = predicted_Y == Y_test
        dense_acc += np.mean(correct_Y) * 100
        #print('DV Accuracy = %.0f%%' %dense_acc)

        ml_model.fit(tfidf_dense_vecs, Y_train)
        predicted_Y = ml_model.predict(test_tfidf_dense_vecs)
        correct_Y = predicted_Y == Y_test
        dense_tfidf_acc += np.mean(correct_Y) * 100
        #print('DV TFIDF Accuracy = %.0f%%' %dense_tfidf_acc)

print('Total majority accuracy:',maj_acc/(1.0*cross_valid),'\nTotal bag-of-words accuracy:',bow_acc/(1.0*cross_valid),'\nTotal tfidf bag-of-words accuracy:',bow_tfidf_acc/(1.0*cross_valid),'\nTotal sparse vector accuracy:',sparse_acc/(1.0*cross_valid),'\nTotal tfidf sparse vector accuracy:',sparse_tfidf_acc/(1.0*cross_valid),'\nTotal dense vector accuracy:',dense_acc/(1.0*cross_valid),'\nTotal tfidf dense vector accuracy:',dense_tfidf_acc/(1.0*cross_valid))