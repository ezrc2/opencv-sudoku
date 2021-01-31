from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()
images = digits.images.reshape((len(digits.images), -1))
labels = digits.target

classifier = svm.SVC(gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33)

classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
predicted = classifier.predict(X_test)