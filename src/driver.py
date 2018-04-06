"""
Created on Apr 6, 2018

@author: AdityAS
email: asaditya1195@gmail.com
Github: https://github.com/AdityaAS

The goal of this test is to verifiy that the PUAdapter really allows a regular estimator to
achieve better accuracy in the case where the \"negative\" examples are contaminated with a
number of positive examples.

Here we use the breast cancer dataset from UCI. We purposely take a few malignant examples and
assign them the bening label and consider the bening examples as being \"unlabled\". We then compare
the performance of the estimator while using the PUAdapter and without using the PUAdapter. To
asses the performance, we use the F1 score, precision and recall.

Results show that PUAdapter greatly increases the performance of an estimator in the case where
the negative examples are contaminated with positive examples. We call this situation positive and
unlabled learning.
"""
import numpy as np
import matplotlib.pyplot as plt
from puLearning.puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def readPath(path):
    fileList = os.listdir(path)
    contentList = []

    print fileList[0:10]
    
    for file in fileList:
        with open(path + file, 'r') as inputFile:
            content = inputFile.read().replace('\n', ' ')
            contentList.append(content)
    return contentList


readPath('/home/aditya/Aditya_AS/IISc/inell/pu-learning/src/datasets/N/')

quit()

if __name__ == '__main__':
    np.random.seed(42)
    
    print "Loading dataset"
    print
    # X = feature matrix (nxf) Y = label vector (nx1)
    X,y = load_breast_cancer('./datasets/breast-cancer-wisconsin.data')
    
    #Shuffle dataset
    print "Shuffling dataset"
    print
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    
    # X = X[0:10]
    # y = y[0:10]
    # print X
    # print y
    #make the labels -1.,+1. I don't like 2 and 4 (:
    
    # In the dataset file 2 represents bening tumour and 4 represents malignant tumour. Convert 2 tp -1 and 4 to +1
    # np.where(y==2) returns a list of indices where the condition [y==2 or y==4] is met. All the elements in those indices are replaced with -1 and +1 respectively
    y[np.where(y == 2)[0]] = -1.
    y[np.where(y == 4)[0]] = +1.

    print "Loaded ", len(y), " examples"
    print len(np.where(y == -1.)[0])," are bening"
    print len(np.where(y == +1.)[0])," are malignant"
    print

    # print len(np.where(y==-1)[0])
    # print np.where(y==-1)[0]

    #Split test/train
    print "Splitting dataset in test/train sets"
    print
    split = 2*len(y)/3
    # Split the dataset into train / test (66% and 33%)
    X_train = X[:split] #Select elements from 0 to split-1 (including both ends)
    y_train = y[:split]

    X_test = X[split:] # Select elements from index select to end.
    y_test = y[split:]
    
    #Training data stats
    print "Training set contains ", len(y_train), " examples"
    print len(np.where(y_train == -1.)[0])," are bening"
    print len(np.where(y_train == +1.)[0])," are malignant"
    print

    #
    pu_f1_scores = []
    reg_f1_scores = []


    # Not sure what this is? but seems like an array having numbers starting from 0 to total positive examples in the train split - 21 separated by intervals of 5. 
    # Still not sure what the significance of 21 here is though.

    # Totally not sure what the significance of 21 here is?
    n_sacrifice_iter = range(0, len(np.where(y_train == +1.)[0])-21, 5)

    print n_sacrifice_iter
    print len(n_sacrifice_iter)
    # quit()

    # Iteration i basically represents the case where exactly i instances of positive labels have been corrupted.


    for n_sacrifice in n_sacrifice_iter:
        #send some positives to the negative class! :)
        
        #I'm still not sure why he is sending some positive examples as negative? How is this helping????
        print "PU transformation in progress."
        #What is the PU transformation????
        print "Making ", n_sacrifice, " malignant examples bening."
        print

        y_train_pu = np.copy(y_train)

        # Get indices of all positive values in the train data. 
        pos = np.where(y_train == +1.)[0]
        
        np.random.shuffle(pos)

        sacrifice = pos[:n_sacrifice]
        
        # Corrupt the labels of the first n_sacrifice positive labels i.e. make them -1 from +1. This is basically a hacky way of representing the fact that these sacrificed
        # positive instances are shifted from Positive labeled set to Unlabeled set. You're essentially making s=-1 by making y=-1 because for the traditional classifier
        # this y is equivalent to s.
        y_train_pu[sacrifice] = -1.
        

        print "PU transformation applied. We now have:"
        print len(np.where(y_train_pu == -1.)[0])," are bening"
        print len(np.where(y_train_pu == +1.)[0])," are malignant"
        print "-------------------"
        
        #Get f1 score with pu_learning
        print "PU learning in progress..."

        # Fit a traditional classifier to the data. In this case a random forest.
        estimator = RandomForestClassifier(n_estimators=100,
                                           criterion='gini', 
                                           bootstrap=True,
                                           n_jobs=1)


        # Get the PU estimator for this traditional estimator
        pu_estimator = PUAdapter(estimator)
        
        print pu_estimator
        
        pu_estimator.fit(X_train, y_train_pu)
        
        y_pred = pu_estimator.predict(X_test)
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        
        pu_f1_scores.append(f1_score[1])
        
        print "F1 score: ", f1_score[1]
        print "Precision: ", precision[1]
        print "Recall: ", recall[1]
        print
        
        #Get f1 score without pu_learning
        print "Regular learning in progress..."
        estimator = RandomForestClassifier(n_estimators=100,
                                           bootstrap=True,
                                           n_jobs=1)
        estimator.fit(X_train,y_train_pu)
        y_pred = estimator.predict(X_test)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        reg_f1_scores.append(f1_score[1])
        print "F1 score: ", f1_score[1]
        print "Precision: ", precision[1]
        print "Recall: ", recall[1]
        print
        print
    plt.title("Random forest with/without PU learning")
    plt.plot(n_sacrifice_iter, pu_f1_scores, label='PU Adapted Random Forest')
    plt.plot(n_sacrifice_iter, reg_f1_scores, label='Random Forest')
    plt.xlabel('Number of positive examples hidden in the unlabled set')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()
    