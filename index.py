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
        DataY.append(float(1))
    elif i[0] == "L":
        people[0] += 1
        DataY.append(float(0.5))
        DataY.append(float(0))
    else:
        people[1] += 1
    DataX.append(list(i[1:]))

print(people)

plt.plot(people)
plt.ylabel('Psycological Health')
plt.show()

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', activation='sigmoid',
hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(DataX, DataY)
clf.predict([['5', '5', '5', '5']])
