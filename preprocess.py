#coding:utf-8

from gensim import corpora, models, similarities
import csv
import time
import nltk
from nltk.corpus import stopwords
import pickle

def createDictionary():
#    reader = csv.reader(open('all'))
    reader = open('all')
    writer = csv.writer(open('remove','w'))
    count = 0
    stoplist = stopwords.words('english')
    texts = []
    begin = time.time()
    print 'Load document'
    for line in reader:
        if count % 1000 == 0:
            print count 

        count += 1
        #temp = ' '.join([title, short_description, need_statement, essay])
#        print line
        temp = line.strip()
#        print temp
        item = [word for word in temp.lower().split() if word not in stoplist]
#        texts.append([word for word in temp.lower().split() if word not in stoplist])
#        print item
#        break
        writer.writerow(item)
        
    end = time.time()
    print end - begin
    print 'Creating dictionary'
#    all_tokens = sum(texts, [])
#    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    
#    texts = [[word for word in text if word not in tokens_once] for text in texts]
        

#    dictionary = corpora.Dictionary(texts)
#    dictionary.save('dictionary.dict')

#    print 'Load dictionary'
#    dictionary = corpora.Dictionary.load('dictionary.dict')
#    print 'create corpus'
#    corpus = [dictionary.doc2bow(text) for text in texts]
#    corpora.MmCorpus.serialize('corpus.mm', corpus)
#    newend = time.time()
#    print newend - end

    
#createDictionary()
#print 'Load Dictionary'
#dictionary = Dictionary.load('dictionary.dict')
#print 'Load corpus'
corpus = corpora.MmCorpus('corpus.mm')
print len(corpus)
count = 0

for item in corpus:
    if count == 0:
        count += 1
        continue
    else:
        print len(item)
        print item
        break
print 'train LSI model'
lsi = models.LsiModel(corpus, num_topics=100)

print 'Get topics'
result = lsi[corpus]

print len(result)

for item in result:
    print item
    break

pickle.dump(result, open('topic.pc','w'))
