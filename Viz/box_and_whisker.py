import numpy as np
from matplotlib import pyplot as plt

RUN_NAME = "Base0"
# ACCURACIES = [68.8, 68.9, 67.6, 68.0, 67.9]  # BertBase GAT
ACCURACIES = [60.5, 60.3, 60.0, 59.3, 60.3]


data = np.array(ACCURACIES)
std_dev = round(np.std(data), 2)
mean = round(np.mean(data), 2)
min = min(ACCURACIES)
max = max(ACCURACIES)

fig = plt.figure(figsize=(5, 6))
plt.boxplot(data, whis=2.5)

plt.title(RUN_NAME + " Dev Accuracy. n=" + repr(len(ACCURACIES)))
plt.xlabel("Mean: " + repr(mean) + " Std Deviation: " + repr(std_dev) + "\nMin: " + repr(min) + " Max: " + repr(max))
plt.ylabel('Accuracy %')

# show plot
plt.show()