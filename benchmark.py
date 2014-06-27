# -*- coding: utf-8 -*-

"""

Beating the benchmark @ KDD 2014

__author__ : Now is Me

"""
import pickle as pc
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.decomposition import PCA
import pickle

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

def getTFIDF():
        print 'Read donations, projects, outcomes, resources, and essays'
        donations = pd.read_csv('donations.csv')
        projects = pd.read_csv('projects.csv')
        outcomes = pd.read_csv('outcomes.csv')
        resources = pd.read_csv('resources.csv')
        sample = pd.read_csv('sampleSubmission.csv')
        essays = pd.read_csv('essays.csv')

        print 'Sort'
        essays = essays.sort('projectid')
        projects = projects.sort('projectid')
        sample = sample.sort('projectid')
        ess_proj = pd.merge(essays, projects, on='projectid')
        outcomes = outcomes.sort('projectid')


        outcomes_arr = np.array(outcomes)


        labels = outcomes_arr[:,1]

        ess_proj['essay'] = ess_proj['essay'].apply(clean)
        
        ess_proj_arr = np.array(ess_proj)

        print 'Split Train and test based on date'

        train_idx = np.where(ess_proj_arr[:,-1] < '2014-01-01')[0]
        test_idx = np.where(ess_proj_arr[:,-1] >= '2014-01-01')[0]

#projects_arr = np.array(outcomes)
#print ess_proj_arr[train_idx,0]

        traindata = ess_proj_arr[train_idx,:]
        testdata = ess_proj_arr[test_idx,:]

        print 'Get TF-IDF'
        stoplist = stopwords.words('english')
        tfidf = TfidfVectorizer(min_df=3,  max_features=1000, ngram_range=(2,3), stop_words=stoplist)

        tfidf.fit(ess_proj_arr[:,5])
        tr = tfidf.transform(traindata[:,5])
#        for item in tr:
#                print item
#                break
        print 'Done tfidf'
        print 'Begin LR'
        ts = tfidf.transform(testdata[:,5])
        lr = linear_model.LogisticRegression()
        lr.fit(tr, labels=='t')
        preds =lr.predict_proba(ts)[:,1]

        sample['is_exciting'] = preds
        sample.to_csv('predictions.csv', index = False)

#        pickle.dump(tr, open('trainTFIDF','w'))
#        pickle.dump(ts, open('testTFIDF','w'))
        
#        print 'Begin PCA'
#        pca = PCA(n_components=200, whiten=True)

#        tr = pca.fit_transform(tr)

#        ts = pca.transform(ts)

        print 'Done LR'
        trainMap = {}
        testMap = {}
        m = len(tr)
        n = len(ts)
        trainID = ess_proj_arr[train_idx,0]
        testID = ess_proj_arr[test_idx,0]
        for i in range(m):
                trainMap[trainID[i]] = tr[i]

        for i in range(n):
                testMap[testID[i]] = ts[i]
        print 'Done'
        pc.dump(trainMap, open('trainTFIDF','w'))
        pc.dump(testMap, open('testTFIDF','w'))
        return trainMap, testMap


if __name__ == '__main__':
        getTFIDF()
