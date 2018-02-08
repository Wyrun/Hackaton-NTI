import matplotlib.pyplot as plt

Data = []

with open("balance-scale.data", "r") as DataFile:
    for line in DataFile:
        Data.append(list(line[0:-1].split(',')))

people = [int(0), int(0), int(0)]

DataY = []
DataX = []

for i in Data:
    if i[0] == "R":
        people[2] += 1
        DataY.append(float(2))
    elif i[0] == "L":
        people[0] += 1
        DataY.append(float(1))
    else:
        people[1] += 1
        DataY.append(float(0))
    DataX.append(list(map(lambda x: float(x), list(i[1:]))))

print(people)
plt.bar([0, 10, 20], people, [8, 8, 8], align='center')
plt.ylabel('Psycological Health')
plt.show()

plt.show()
"""
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', activation='relu',
hidden_layer_sizes=(5, 2), random_state=1)
clf.fit([[0., 1.], [0., 0.]], [1., 0.])
#clf.predict([['5', '5', '5', '5']])
clf.predict([['1', '1']])"""
from sklearn import metrics



from sklearn.cross_validation import train_test_split

from sklearn.neural_network import MLPClassifier

expment = 1
portion = [float(0.3), float(0.1)]
actFunc = ['identity', 'logistic', 'relu']
layers = [int(4), int(5), int(6), int(7)]
result = open("docx.txt", "w")
ma = ['','','',0]
for pr in portion:
    for fun in actFunc:
        for lay in layers:
            X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=pr, random_state=42)
            clf = MLPClassifier(solver='lbfgs', activation=fun,
            hidden_layer_sizes=(lay, 12))
            clf.fit(X_train, Y_train)
            res_clf = clf.predict(X_test)
            res = metrics.accuracy_score(Y_test, res_clf)
            if pr == 0.3:
                result.write(str(expment)+"\t"+str(lay)+"\t"+str(fun)+"\t"+"70/30"+"\t"+str(res))
            else:
                result.write(str(expment)+"\t"+str(lay)+"\t"+str(fun)+"\t"+"90/10"+"\t"+str(res))
            if ma[3] < res:
                ma = [pr, fun, lay, res]
            expment += 1
            result.write("\n")
result.write("Best: "+str(ma))
