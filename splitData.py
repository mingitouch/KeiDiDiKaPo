#coding:utf-8

import csv
splitline = '2014-01-01'
baseline = '2010-04-14'

def splitData(trainFile, testFile):
#    train = csv.writer(open('train','w'))
#    test = csv.writer(open('test','w'))
    reader = csv.reader(open('projects.csv'))
    trainID = []
    testID = []
    count = 0
    for line in reader:
        if count == 0:
            count += 1
            continue

        if count % 10000 == 0:
            print count
        count += 1
        time = line[-1]
        if time >= splitline:
            testID.append(line[0])
#            test.writerow(line)
        elif time < splitline and time >= baseline:
            trainID.append(line[0])
#            train.writerow(line)

    return trainID, testID

def generateTarget(trainID, testID):
    reader = csv.reader(open('outcomes.csv'))
    file = open('Y','w')
    mapping = {}
    truecnt = 0
    for line in reader:
        g = lambda x: 1 if x == 't' else 0
        mapping[line[0]] = g(line[1])
        truecnt += g(line[1])
    
    traincnt = 0
    for item in trainID:
        if mapping.has_key(item):
            file.write(item+','+str(mapping[item])+'\n')

def splitEssay(trainID, testID):
    reader = csv.reader(open('essays.csv'))
    train = csv.writer(open('trainEssay.csv','w'))
    test = csv.writer(open('testEssay.csv','w'))
    tempTrain = set(trainID)
    tempTest = set(testID)
    count = 0
    
    for line in reader:
        if count == 0:
            count += 1
            continue
        if count % 10000 == 0:
            print count
        count += 1
        if line[0] in tempTrain:
            train.writerow(line)
        elif line[0] in tempTest:
            test.writerow(line)

def splitResource(trainID, testID):
    #Something wrong, do not try this function
    reader = csv.reader(open('resources.csv'))
    train = csv.writer(open('trainResources.csv','w'))
    test = csv.writer(open('testResources.csv','w'))
    tempTrain = set(trainID)
    tempTest = set(testID)
    count = 0

    for line in reader:
        if count == 0:
            count += 1
            continue
        if count % 10000 == 0:
            print count
        count += 1
        if line[0] in tempTrain:
            train.writerow(line)
        elif line[0] in tempTest:
            test.writerow(line)
        
def check(line):
    if line == 't':
        return True
    elif line == None:
        return False
    elif line == '':
        return False
    elif line == 'f':
        return False
    else:
        try:
            if float(line) > 0:
                return True
            else:
                return True
        except:
            print '***********************************'
            print type(line)
            print '***********************************'

def getValue(line):
    if line == 't':
        return 1
    elif line == None:
        return 0
    elif line == '':
        return 0
    elif line == 'f':
        return 0
    else:
        return float(line)

    

def generateMoreFeature():
    mapping = {'teacher':{}, 'school':{}, 'state':{}}
    project = {}
    temp = [0,0,0,0,0,0,0,0,0,0]
    projectReader = csv.reader(open('projects.csv'))
    outcomeReader = csv.reader(open('outcomes.csv'))
    count = 0
    for line in outcomeReader:
        if count == 0:
            count += 1
            continue
        project[line[0]] = line[1:]

    count = 0
    for line in projectReader:
        if count == 0:
            count += 1
            continue
        count += 1
        if not project.has_key(line[0]):
            if mapping['teacher'].has_key(line[1]):
                mapping['teacher'][line[1]] = [0,0,0,0,0,0,0,0,0,0]
                #pass
            else:
                mapping['teacher'][line[1]] = [0,0,0,0,0,0,0,0,0,0]
            
            if mapping['school'].has_key(line[2]):
                #pass
                mapping['school'][line[2]] =  [0,0,0,0,0,0,0,0,0,0]
            else:
                mapping['school'][line[2]] = [0,0,0,0,0,0,0,0,0,0]

            if mapping['state'].has_key(line[7]):
                #pass
                mapping['state'][line[7]] =  [0,0,0,0,0,0,0,0,0,0]
            else:
                mapping['state'][line[7]] =  [0,0,0,0,0,0,0,0,0,0]

            count += 1
            continue
        for i in range(10):

            if check(project[line[0]][i]) == True:
                if mapping['teacher'].has_key(line[1]):
                    mapping['teacher'][line[1]][i] += getValue(project[line[0]][i])
#                    print project[line[0]][i]
                else:
                    mapping['teacher'][line[1]] = [0,0,0,0,0,0,0,0,0,0]
                    mapping['teacher'][line[1]][i] = getValue(project[line[0]][i])
                
                if mapping['school'].has_key(line[2]):
                    mapping['school'][line[2]][i] += getValue(project[line[0]][i])
                else:
                    mapping['school'][line[2]] = [0,0,0,0,0,0,0,0,0,0]
                    mapping['school'][line[2]][i] = getValue(project[line[0]][i])

                if mapping['state'].has_key(line[7]):
                    mapping['state'][line[7]][i] += getValue(project[line[0]][i])
                else:
                    mapping['state'][line[7]] = [0,0,0,0,0,0,0,0,0,0]
                    mapping['state'][line[7]][i] = getValue(project[line[0]][i])



    print count
    return mapping

#trainID, testID = splitData('train','test')
#generateTarget(trainID, testID)
#mapping = generateMoreFeature()
#for item in mapping:
#    print item
#    if item != 'teacher':
#        continue
#    for it in mapping[item]:
#        print it, mapping[item][it]
#    break
'''
'''
