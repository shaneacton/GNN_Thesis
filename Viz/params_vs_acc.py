
GROUP_NAME = "Base1"
ACCURACIES = [60 ,61, 62, 65, 60]
NUM_PARAMS = [5, 10, 12, 8, 15]

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(NUM_PARAMS, ACCURACIES, color="blue")

plt.title(GROUP_NAME + " Parameter Accuracy Correlation. n=" + repr(len(ACCURACIES)))
plt.xlabel("Num Model Parameters")
plt.ylabel('Accuracy %')

params = np.array(NUM_PARAMS)
accs = np.array(ACCURACIES)
pearR = round(np.corrcoef(params,accs)[1,0], 3)

A = np.vstack([params, np.ones(len(params))]).T
m, c = np.linalg.lstsq(A, accs)[0]
plt.plot(params,params*m+c,color="red", label="R = " + repr(pearR))

plt.legend(loc=2)
plt.show()