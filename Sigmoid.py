from sklearn.svm import SVC
import kernelSVM as m

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(m.X_train, m.y_train)

y_pred = svclassifier.predict(m.X_test)

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # ignore any warning, note:warning is zero division error
print(confusion_matrix(m.y_test, y_pred))
print(classification_report(m.y_test, y_pred))
