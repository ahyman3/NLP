
# coding: utf-8

# In[31]:


import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import re
import numpy as np


# In[2]:


dirPath = "corpus/"
limitStr = 1500


# In[3]:


# convert the limit argument from a string to an int
limit = int(limitStr)

# start lists for spam and ham email texts
hamtexts = []
spamtexts = []
os.chdir(dirPath)
# process all files in directory that end in .txt up to the limit
#    assuming that the emails are sufficiently randomized
for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()


# In[4]:


print ("Number of spam files:",len(spamtexts))
print ("Number of ham files:",len(hamtexts))


# In[5]:


# create list of mixed spam and ham email documents as (list of words, label)
emaildocs = []
# add all the spam
for spam in spamtexts:
    tokens = nltk.word_tokenize(spam)
    emaildocs.append((tokens, 'spam'))
# add all the regular emails
for ham in hamtexts:
    tokens = nltk.word_tokenize(ham)
    emaildocs.append((tokens, 'ham'))


# In[6]:


# randomize the list
random.shuffle(emaildocs)

# print a few token lists
for email in emaildocs[:4]:
    print (email)


# In[7]:


#getting all the words
all_words_list = [word for sent, cat in emaildocs for word in sent]
#Creating a frequency list
all_words = nltk.FreqDist(all_words_list)
#Top 2500 words
word_items = all_words.most_common(1000)
#Words only
word_features = [word for word, count in word_items]


# In[187]:


word_items[:10]


# In[181]:


# continue as usual to get all words and create word features
#word features only
def get_word_features(document, word_features):
    document_words = list(set(document))
    features = {}
    for word in word_features:
        features["V_{:s}".format(word)] = (word in document_words)
    return features


# In[41]:


# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('Label\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the precision, recall and F score for each fold 
#.  (it does not compute the average over the folds)
def cross_validation_PRF(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # call the function with our data
        eval_measures(goldlist, predictedlist)
    # this version doesn't save measures and compute averages
## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)


# ## All Words

# In[183]:


# train classifier and show performance in cross-validation
word_features_only = [(get_word_features(doc, word_features), cat) for doc, cat in emaildocs]
#cross-validation
cross_validation_PRF(10, word_features_only)
cross_validation_accuracy(10, word_features_only)


# ## No Stopwords and punctuation

# In[62]:


#Getting stopwords
stopwords = nltk.corpus.stopwords.words("english")
#Removing stop words
no_stopwords_all = [word for sent, cat in emaildocs for word in sent if word not in stopwords]
#removing punctuation and numbers
pat = re.compile("[^A-Za-z]+")
only_words = [word for word in no_stopwords_all if re.match(pat, word) is None]
#Putting in frequency distribution
stopwords_removed_dist = nltk.FreqDist(only_words)
#Getting top 1000 word features
top_words_only = stopwords_removed_dist.most_common(1000)
#Putting the words in a list
word_features_stop_removed = [word for word, count in top_words_only]
#Creating featureset
no_stop_documents = [(get_word_features(doc, word_features_stop_removed),cat) for
                     doc, cat in emaildocs]


# In[64]:


#cross-validation
cross_validation_PRF(10, no_stop_documents)
cross_validation_accuracy(10, no_stop_documents)


# ## POS Tag (with stopwords removed)

# In[188]:


def get_wordfeatures_pos(document, word_features):
    #regex pattern
    all_caps = re.compile("[A-Z]+")
    #all words in the document
    document_words = list(set(document))
    #Features dictionary
    features = {}
    #Creating features with the word features
    for word in word_features:
        features["V_{:s}".format(word)] = (word in document_words)
    #initializing the pos count
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0
    #Tagging the words
    tagged_words = nltk.pos_tag(document)
    #If word falls in the noun, adj, adv, or verb category, add to count
    for word, tag in tagged_words:
        if tag.startswith("N"): noun_count += 1
        if tag.startswith("J"): adj_count += 1
        if tag.startswith("V"): verb_count += 1
        if tag.startswith("R"): adv_count += 1
    #Save count in the features dictionary
    features["verbcount"] = verb_count
    features["adjcount"] = adj_count
    features["nouncount"] = noun_count
    features["advcount"] = adv_count
    return features
#Creating the part of speech category dataset
pos_features = [(get_wordfeatures_pos(doc, word_features_stop_removed), cat) for
               doc, cat in emaildocs]


# In[189]:


#cross-validation for POS features added
cross_validation_PRF(10, pos_features)
cross_validation_accuracy(10, pos_features)


# # TF-IDF Manually

# In[21]:


#Getting stopwords
stopwords = nltk.corpus.stopwords.words("english")
#Removing stop words
no_stopwords_all = [word for sent, cat in emaildocs for word in sent if word not in stopwords]
#removing punctuation and numbers
pat = re.compile("[^A-Za-z]+")
only_words = [word for word in no_stopwords_all if re.match(pat, word) is None]
#Putting in frequency distribution
stopwords_removed_dist = nltk.FreqDist(only_words)
#Getting top 1000 word features
top_words_only = stopwords_removed_dist.most_common(1000)
top_words = [w for w, c in top_words_only]

#Doc frequency
doc_count = {}
#Looping through each document
for email, _ in emaildocs:
    #getting a list of all unique words
    list(set(email))
    #Going through a list of only the top words
    for word in top_words:
        #If the word feature is in the document, add 1 to the value
        doc_count[word] = doc_count.get(word, 0) + int(word in email)


# In[29]:


#For each word being used
for w in doc_count.keys():
    #Find the word in the dictionary, and change the present value to 
    #be the document count divided by 3000 
    doc_count[w] = doc_count.get(w) / 3000


# In[37]:


def create_tfidf(doc, docfreq_terms):
    """
    This function takes in a document an a preprocessed document frequenct dictionary
    and returns the tf * log10(1 / df) for each word in the word features
    """
    #Creating a frequency distribution
    termfreq = nltk.FreqDist(doc)
    #looping through each unique word in the email document
    for w in termfreq.keys():
        #Getting the term frequency in the document
        termfreq[w] = termfreq.get(w) / len(doc)
    #Feature dictionary to be returned
    features = {}
    for w, df in docfreq_terms.items():
        #Creates the feature dictionary to return
        features["Tfidf_{:s}".format(w)] = termfreq.get(w, 0) * np.log10(1 / df)
    #return the feature dictionary
    return features


# In[38]:


tfidf_features = [(create_tfidf(doc, doc_count), cat) for doc, cat in emaildocs]


# In[42]:


#cross-validation for tfidf features added
cross_validation_PRF(10, tfidf_features)
cross_validation_accuracy(10, tfidf_features)


# # TFIDF sklearn

# In[227]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#Creating tfidf vectorizer object
tfidf = TfidfVectorizer(tokenizer= nltk.word_tokenize, stop_words= stopwords, lowercase = True,
                       max_df = 0.9, max_features = 1000)

random.seed(1000)
#0 is normal text, 1 is spam
spam_tuple = [(doc, 1) for doc in spamtexts]
ham_tuple = [(doc, 0) for doc in hamtexts]
#combining all texts
all_texts = spam_tuple + ham_tuple
#randomly shuffling all texts
random.shuffle(all_texts)
#text only
email_text = [doc for doc, cat in all_texts]
labels = [cat for doc, cat in all_texts]


# In[228]:


#fitting the tfidf based on all texts
X = tfidf.fit_transform(email_text)
#making the labels for y
y = np.array(labels)
y = y.reshape(3000, )


# In[229]:


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[230]:


#PArameter grid for finding the best fit on the train set
param_grid = {"C":[1,5,10], "kernel": ["linear", "rbf", "poly"], "degree":[2,3]}
#creating an svc model
svc = SVC()
#fitting the parameter grid to the model
gs = GridSearchCV(svc, param_grid)
gs.fit(X_train, y_train)


# In[231]:


#Getting the accuracy on the test set
print("Accuracy on hold out set: {:.4f}".format(
    accuracy_score(y_pred=gs.best_estimator_.predict(X_test), y_true=y_test)))


# In[232]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#Accuracy of the model with a 10-fold crossvalidation on all data
cv_results = cross_validate(gs.best_estimator_, X, y, cv = 10, scoring = ("accuracy"))


# In[233]:


print("Average Test Accuracy (10-fold): {:.4f}".format(cv_results["test_score"].mean()))


# In[234]:


predictions = gs.best_estimator_.predict(X)
print("Spam f1 score: {:.4f}".format(f1_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=1)))
print("Spam precision score: {:.4f}".format(precision_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=1)))
print("Spam recall score: {:.4f}\n".format(recall_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=1)))

print("Ham f1 score: {:.4f}".format(f1_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=0)))
print("Ham precision score: {:.4f}".format(precision_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=0)))
print("Ham recall score: {:.4f}".format(recall_score(y_pred=predictions, y_true=y, labels=["Ham", "Spam"], pos_label=0)))


# In[235]:


gs.best_estimator_

