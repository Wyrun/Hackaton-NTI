import matplotlib.pyplot as plt

Data = []

with open("balance-scale.data", "r") as DataFile:
    for line in DataFile:
        Data.append(list(line[0:-1].split(',')))

print (Data)

people = [int(0), int(0), int(0)]

for i in Data:
    if i[0] == "R":
        people[2] += 1
    elif i[0] == "L":
        people[0] += 1
    else:
        people[1] += 1

plt.plot(people)
plt.ylabel('Psycological Health')
plt.show()

print(people)
