import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder

stateInfo = {}
projectInfo = {}
myCategory = [{}] * 20
con = []
projects = {}

def readFile():
    proj = csv.reader(open('projects.csv'))
    count = 0
    for line in proj:
        if count == 0:
            count += 1
            continue
        projects[line[0]] = line
    con = []
    reader = csv.reader(open('donations.csv'))
    enc = OneHotEncoder()
    temp = list()
    count = 0
    print 'First read file'
    for line in reader:
        if count == 0:
            count += 1 
            continue
#        value = [line[6]]+line[11:-1]
        value = [line[6], line[11], line[12], line[13], line[19]]
        con.append([float(line[8]), float(line[10])])
        feature = []
        for i in range(5):
            if myCategory[i].has_key(value[i]):
                feature.append(myCategory[i][value[i]])
            else:
                myCategory[i][value[i]] = count
                feature.append(count)
        count += 1
        temp.append(feature)
#        temp.append([line[6]]+line[11:-1])

    
    print 'onehot encoder'
    temp = enc.fit_transform(temp).toarray()
    con = np.array(con)
    temp = np.append(temp, con, 1)
#    print temp.shape
    count = 0
    print 'trans File'
    reader = csv.reader(open('donations.csv'))    
    for line in reader:
        if count == 0:
            count += 1
            continue
#        if count % 10000 == 0:
#            print count
#        value = []
#        tmp = [line[6]]+line[11:-1]
#        for i in range(10):
#            value.append(myCategory[i][tmp[i]])
#        value = enc.transform([value]).toarray()[0]
#        if projectInfo.has_key(line[1]):
#            projectInfo[line[1]] = projectInfo[line[1]] + value
#        else:
#            projectInfo[line[1]] = value

        if stateInfo.has_key(projects[line[1]][7]):
            stateInfo[projects[line[1]][7]] = stateInfo[projects[line[1]][7]] + temp[count-1]
        else:
            stateInfo[projects[line[1]][7]] = temp[count-1]
        count += 1


#import time
#begin = time.time()
#readFile()
#print time.time()-begin
#print len(stateInfo)
