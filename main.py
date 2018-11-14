import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.svm import SVC

data = np.array(list(csv.reader(open("svm-data.csv", "r"), delimiter=","))).astype("float")
X = data[:, [1,2]]
Y = data[:, 0]

# Create a linear SVM classifier 
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X,Y)

# Print indexes of support vectors
print("Support vectors", clf.support_)

# Plot the data
w = clf.coef_[0]
print("Boundary coeffs", w)

a = -w[0] / w[1]

xx = np.linspace(0,1)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.title("SVM with linear kernel")
plt.show()
