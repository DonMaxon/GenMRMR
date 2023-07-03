import numpy
from GenMRMR import GenMRMR
import sklearn
from imblearn.over_sampling import RandomOverSampler
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
    
def prepareData():
    data, labels = load_wine(as_frame=True, return_X_y=True)
    return labels, data

def splitData(data: numpy.ndarray, labels: numpy.ndarray):
    oversample = RandomOverSampler(sampling_strategy="minority")
    x, x_test, y, y_test = model_selection.train_test_split(
        data, labels, test_size=0.2, train_size=0.8
    )
    x_train, x_cv, y_train, y_cv = model_selection.train_test_split(
        x, y, test_size=0.25, train_size=0.75
    )
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    x_cv, y_cv = oversample.fit_resample(x_cv, y_cv)
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x_train = scaler.transform(x_train)
    x_cv = scaler.transform(x_cv)
    x_test = scaler.transform(x_test)
    return x_train, x_test, x_cv, y_train, y_test, y_cv

def test_algorithm(data, labels, classifier, num_of_features):
    x_train, x_test, x_cv, y_train, y_test, y_cv = splitData(data, labels)
    my_alg = GenMRMR(classifier, num_of_features)
    my_alg.fit(x_train, y_train, x_cv, y_cv)
    new_train = my_alg.transform(x_train)
    new_test = my_alg.transform(x_test)
    classifier.fit(new_train, y_train)
    y_predicted = classifier.predict(new_test)
    return f1_score(y_predicted, y_test, average='weighted')

labels, data = prepareData() 
classifier = RandomForestClassifier()
s = 0
num_of_features = data.shape[1]//2
print(test_algorithm(data, labels, classifier, num_of_features))

