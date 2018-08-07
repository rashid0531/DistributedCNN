import matplotlib.pyplot as plt
import numpy as np
with open("output.txt","r") as file_object:
    file_contents = file_object.readlines()
    predicted = []
    original = []
    for each_line in file_contents:
        each_line = each_line.strip()
        parts = each_line.split(",")
        original.append(float(parts[1].split(":")[-1]))
        predicted.append(float(parts[2].split(":")[-1]))

#print(predicted[0],original[0])


fig=plt.figure()
#fig.show()
ax=fig.add_subplot(111)

#print(len(original),len(predicted))
dim = np.arange(0,len(original),1);
ax.plot(dim,original)
ax.plot(dim,predicted)

plt.show()
