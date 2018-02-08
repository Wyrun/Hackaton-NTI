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
#print(DataY)

plt.plot(people)
plt.ylabel('Psycological Health')
#plt.line2d.drawstyle('steps')
plt.show()
"""
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', activation='relu',
hidden_layer_sizes=(5, 2), random_state=1)
clf.fit([[0., 1.], [0., 0.]], [1., 0.])
#clf.predict([['5', '5', '5', '5']])
clf.predict([['1', '1']])"""
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.2, random_state=42)



from sklearn.neural_network import MLPClassifier
X = [[5.0, 5.0, 5.0, 3.0], [5.0, 5.0, 5.0, 4.0]]
Y = [1., 1.]
clf = MLPClassifier(solver='lbfgs', activation='relu',
hidden_layer_sizes=(4, 8, 4, 4, 4))
clf.fit(X_train, Y_train)
print("Hekko Workt")
print(clf.predict([[2., 2., 1., 0.1]]))
