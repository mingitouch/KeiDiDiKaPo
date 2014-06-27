import numpy as np
from splitData import *
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

word2vec = {}
doc2vec = {}
projectID = []
#trainID, testID = splitData('', '')

def getVec():
    file = open('output')
    count = 0
    for line in file:
        if count == 0:
            print line.strip()
            count += 1
            continue
        if count % 10000 == 0:
            print count
        count += 1
        line = line.strip().split(' ')
        word2vec[line[0]] = np.array(map(float, line[1:]))
    file.close()

def getdocVec():
    import pickle
    file = open('newAll')
    project = open('../projectID')
    for line in project:
        projectID.append(line.strip())
    count = 0
    for line in file:
        if count % 10000 == 0:
            print count
        line = line.strip().split(' ')
        temp = np.zeros(100)
        for item in line:
            if word2vec.has_key(item):
                temp = temp + word2vec[item]
        doc2vec[projectID[count]] = temp
        count += 1

    print 'Begin to dump'
    pickle.dump(doc2vec, open('doc2vec','w'))

def trainModel():
    import csv
    import pickle
    print 'load document vector'
    import time
    begin = time.time()
    doc2vec = pickle.load(open('doc2vec'))
    end = time.time()
    print end - begin
    target = {}
    file = csv.reader(open('../data/Y'))
    for key, value in file:
        target[key] = int(value)

    trainID, testID = splitData('', '')
    
    Y = []
    train = []
    test = []
    print 'split train and test'
    for item in trainID:
#        print key
        Y.append(target[item])
        train.append(doc2vec[item])
    for item in testID:
        test.append(doc2vec[item])

    print 'begin to logistic regression'
    clf = LogisticRegression(C=2.0)
    train = np.array(train)
    test = np.array(test)
    print train.shape
    print test.shape
    Y = np.array(Y)
#    print 'begin to cv'
#    print np.mean(cross_validation.cross_val_score(clf,train,Y,cv=3, n_jobs=-1, scoring='roc_auc'))
    clf.fit(train, Y)
    result = clf.predict_proba(test)[:,1]
    n = len(test)
#    print result
    out = open('word2vecResult.csv','w')
    for i in range(n):
        out.write(str(testID[i])+','+str(result[i])+'\n')

if __name__ == '__main__':
    import time
#    a = time.time()
#    getVec()
#    b = time.time()
#    print b - a
#    getdocVec()
    c = time.time()
#    print c - b
    trainModel()
    d = time.time()
    print d - c
#    print doc2vec
