import csv

reader1 = csv.reader(open('result.csv'))
reader2 = csv.reader(open('predictions.csv'))
out = csv.writer(open('merge.csv','w'))
mapping = {}
for line in reader1:
    mapping[line[0]] = float(line[1])
    
for line in reader2:
    if mapping.has_key(line[0]):
        temp = float(line[1]) * (1-0.56552863436123348017621145374449)  + mapping[line[0]] * 0.56552863436123348017621145374449
        out.writerow([line[0], temp])

    else:
        out.writerow(line)
    
