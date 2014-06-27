import pandas as pd
import numpy as np
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
import theanets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Helper functions
def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]

#Loading CSV files
#donations = pd.read_csv('Data/donations.csv')
projects = pd.read_csv('projects.csv')
outcomes = pd.read_csv('outcomes.csv')
#resources = pd.read_csv('Data/resources.csv')
sample = pd.read_csv('sampleSubmission.csv')
#essays = pd.read_csv('Data/essays.csv')



print 'Read data files.'

#Sort data according the project ID
#essays = essays.sort('projectid')
projects = projects.sort('projectid')
sample = sample.sort('projectid')
outcomes = outcomes.sort('projectid')
#donations = donations.sort('projectid')
#resources = resources.sort('projectid')

#Setting training data and test data indices
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

#Filling missing values
projects = projects.fillna(method='pad') #'pad' filling is a naive way. We have better methods.

#Set target labels
labels = np.array(outcomes.is_exciting)

#Preprocessing
#projects_numeric_columns = ['school_latitude', 'school_longitude',
#                            'fulfillment_labor_materials',
#                            'total_price_excluding_optional_support',
#                            'total_price_including_optional_support']

projects_numeric_columns = ['school_latitude', 'school_longitude','fulfillment_labor_materials','students_reached ','total_price_excluding_optional_support','total_price_including_optional_support']

#projects_id_columns = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']

projects_id_columns = ['school_state','school_metro','school_charter','school_magnet','school_year_round    ','school_nlns','school_kipp','school_charter_ready_promise','teacher_prefix','teacher_teach_for_america    ','teacher_ny_teaching_fellow','primary_focus_subject','primary_focus_area','secondary_focus_area','seco    ndary_focus_subject','resource_type','poverty_level','grade_level','eligible_double_your_impact_match','    eligible_almost_home_match']

projects_categorial_columns = diff(diff(diff(list(projects.columns), projects_id_columns), projects_numeric_columns), 
                                   ['date_posted'])

projects_categorial_values = np.array(projects[projects_categorial_columns])

label_encoder = LabelEncoder()
projects_data = label_encoder.fit_transform(projects_categorial_values[:,0])

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))

projects_data = projects_data.astype(float)

#One hot encoding!
enc = OneHotEncoder()
enc.fit(projects_data)
projects_data = enc.transform(projects_data)

#Predicting
train = projects_data[train_idx].toarray()
test = projects_data[test_idx].toarray()
#clf = LogisticRegression()
clf =  RandomForestClassifier(n_estimators=93, max_depth=18, min_samples_leaf=15, criterion='entropy', oob_score=True, min_samples_split=6, n_jobs = -1)
print 'begin CV'
print np.mean(cross_validation.cross_val_score(clf,train,np.array(labels=='t'),cv=3, n_jobs=-1, scoring='roc_auc'))

print 'train'
clf.fit(train, labels=='t')
preds = clf.predict_proba(test)[:,1]

print 'Save prediction into a file'
sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)
print np.sum(preds)
