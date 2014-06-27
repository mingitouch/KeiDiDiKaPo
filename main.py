from gensim import corpora, models, similarities
import csv
import time
import gensim
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfTransformer

dictionary = corpora.Dictionary.load('dictionary.dict')
testcorpus = corpora.MmCorpus('data/test/test.mm')
print testcorpus
begin = time.time()
test_corpus_csc = gensim.matutils.corpus2csc(testcorpus)
test_corpus_csc = test_corpus_csc.transpose()
end = time.time()
print end - begin

traincorpus = corpora.MmCorpus('data/train/train.mm')
print traincorpus
train_corpus_csc = gensim.matutils.corpus2csc(traincorpus)
train_corpus_csc = train_corpus_csc.transpose()
newend = time.time()
print newend - end
lr = LogisticRegression()
reader = csv.reader(open('data/Y'))
Y = []
for item in reader:
    Y.append(int(item[0]))

begin = time.time()
print 'Train LR'
lr.fit(train_corpus_csc, Y)
end = time.time()
print end -begin
print 'Predict'
result = lr.predict(test_corpus_csc)
print
file = open('result.csv','w')
for item in result:
    file.write(str(item)+'\n')
