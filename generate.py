#coding:utf-8
from sklearn.linear_model import SGDClassifier
import csv
from splitData import *
from benchmark import *
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
import pickle as pc
from sklearn.neural_network import BernoulliRBM
from donor import *

print 'Hello everybody!'

g = lambda x : 1 if x == 't' else 0
t = lambda x : 0 if x == '' else x
def check(dictionary, key1, key2):
    if dictionary.has_key(key1):
        if dictionary[key1].has_key(key2):
            return dictionary[key1][key2]
        else:
            return [0,0,0,0,0,0,0,0,0,0]
    else:
        return [0,0,0,0,0,0,0,0,0,0]

mapping = {}
category = [{}] * 20
value = generateMoreFeature()
#readFile()
#print 'get document vector'
#begin = time.time()
#doc2vec = pc.load(open('shell/doc2vec'))
#end = time.time()
#print end - begin
#print 'Get TF-IDF'
#begin = time.time()
#trainMap = pc.load(open('trainTFIDF'))
#print time.time() - begin
#for item in trainMap:
#    print item
#    print np.array(trainMap[item]).shape
#    break
#testMap = pc.load(open('testTFIDF'))

def getMonth(date):
    year, month, day = date.split('-')
    return month,day

def getEassyLength():
    reader = csv.reader(open('essays.csv'))
#    reader = csv.reader(open('thisisgreat'))
    count = 0
    for line in reader:
        if count == 0:
            count += 1
            continue

        if count % 10000 == 0:
            print count
        count += 1
        mapping[line[0]] = len(line[5])
#        print len(line[1])
#        print ' '.join(line[5:])
#        mapping[line[0]] = len(' '.join(line[5:]))
        mapping[line[0]] = {}
        mapping[line[0]]['character'] = len(' '.join(line[5:]))
        mapping[line[0]]['sentence'] = len(' '.join(line[5:]).split('.'))
        mapping[line[0]]['word'] = len(','.join(line[5:]).split(' '))

def generateData(filename, output):
    reader = csv.reader(open(filename))

#    writer = csv.writer(open(output, 'w'))
    
    count = 0
    X = []
    for line in reader:
        feature = []
        count += 1
        addition = [value['state'][line[7]][0], value['state'][line[7]][4], value['state'][line[7]][7], check(value,'school',line[2])[7], check(value,'teacher',line[1])[7], value['state'][line[7]][3], value['state'][line[7]][5]]

#        for item in doc2vec[line[0]]:
#            addition.append(item)
#        for item in stateInfo[line[7]]:
#            addition.append(item)
        feature = [float(t(line[28])), float(line[29]), float(t(line[30])), float(t(line[31])), mapping[line[0]]['character']] + addition
#        feature = np.array(feature)
#        if stateInfo.has_key(line[7]):
#            feature = np.append(feature, stateInfo[line[7]], 1)
#        else:
#            feature = np.append(feature, stateInfo[''], 1)
        X.append(feature)
    return np.array(X)

def categoryFeature(filename):
    reader = csv.reader(open(filename))
    count = 0
    X = []
    index = [7,9,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,32,33]
    for line in reader:
#        line = line[6:28]
#        line[34], day = getMonth(line[34])
#        line.append(day)
        line = np.array(line)
        line = line[index]
#        line.append(day)
        n = len(line)
        feature = []
        for i in range(n):
            if category[i].has_key(line[i]):
                feature.append(category[i][line[i]])
            else:
                category[i][line[i]] = count
                feature.append(count)
#        print feature
        count += 1
        X.append(feature)

    return X
    

trainID, testID = splitData('train', 'test')
print len(testID)
print len(testID)
getEassyLength()
print 'train'
#trainX = categoryFeature('data/train/trainProject')
print 'category feature'
trainX = categoryFeature('data/train/trainProject')
#trainX = generateData('data/train/trainProject', 'train')
print 'generate data'
tempTrainX = generateData('data/train/trainProject', 'train')
print 'test'
print 'category feature'
testX = categoryFeature('data/test/testProject')
print 'generate data'
tempTestX = generateData('data/test/testProject', 'test')
#testX = generateData('data/test/testProject', 'test')

categoryID = [item.values() for item in category]
enc = OneHotEncoder()
enc.fit(trainX)
trainX = enc.transform(trainX).toarray()
testX = enc.transform(testX).toarray()



#scaler = preprocessing.StandardScaler().fit(tempTrainX)
#tempTrainX = scaler.transform(tempTrainX)
#tempTestX = scaler.transform(tempTestX)


trainX = np.append(trainX, tempTrainX, 1)
testX = np.append(testX, tempTestX, 1)

#allData = np.append(trainX, testX, 0)

#layers = (158, 100, 158)

#ae = theanets.Autoencoder(layers, 'sigmoid')

#exp = theanets.Experiment(theanets.Autoencoder, layers = (158, 100, 158))

#print 'begin autoencoder'
#exp.train(allData)

#aebegin = time.time()

#exp.run(trainX, testX)

#exp.save('para')

#ae.load('para')

#trainX = ae.predict(trainX)

#testX = ae.predict(testX)

#print 'end autoencoder'
#aeend = time.time()

#print aeend - aebegin

#print trainX.shape

#print testX.shape

#print 'begin PCA'

#pca = PCA(n_components=200, whiten=True)

#trainX = pca.fit_transform(trainX)

#testX = pca.transform(testX)

#print 'begin RBM'
#rbmbegin = time.time()

#rbm = BernoulliRBM(n_components=1000)
#trainX = rbm.fit_transform(trainX)
#testX = rbm.transform(testX)

#print time.time() - rbmbegin
#print 'end RBM' 


reader = csv.reader(open('data/Y'))
Y = []
for line in reader:
    Y.append(int(line[1]))
Y = np.array(Y)
print trainX.shape
print testX.shape
#clf = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree')
#clf = svm.SVC()
#clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=2,tol=1e-9,class_weight=None, random_state=None, intercept_scaling=1.0)
#print cross_validation.cross_val_score(clf,trainX,Y,cv=3)
#clf = LogisticRegression(penalty='l1',C=2.0)
#clf = SGDClassifier(loss='log', penalty='l2', alpha = 1e-4, class_weight='auto')
#clf =  RandomForestClassifier(n_estimators=93, max_depth=8, min_samples_leaf=4, n_jobs = -1)
clf =  RandomForestClassifier(n_estimators=93, max_depth=18, min_samples_leaf=15, criterion='entropy', oob_score=True, min_samples_split=6, n_jobs = -1)
#clf =  RandomForestClassifier(n_estimators=93, max_depth=18, min_samples_leaf=15, criterion='entropy', min_samples_split=6, n_jobs = -1)
#clf = GradientBoostingClassifier(n_estimators=93, min_samples_leaf=15, max_depth=18, min_samples_split=6)
#clf =  GaussianNB()
print 'begin CV'
print np.mean(cross_validation.cross_val_score(clf,trainX,Y,cv=3, n_jobs=-1, scoring='roc_auc'))
#auc_score = roc_auc_score(Y,clf.predict_proba(trainX)[:,1])
#print auc_score
begin = time.time()
print 'begin train'
clf.fit(trainX, Y)
end = time.time()
print end - begin
print 'Predict'
result = clf.predict_proba(testX)[:,1]
#print np.sum(result)
n = len(result)
print 'Output'
#print type(result[0])
file = open('result.csv','w')
print np.sum(result)
for i in range(n):
    file.write(str(testID[i])+','+str(result[i])+'\n')


