import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class BuildModels:
    def __init__(self):
        train = pd.read_csv("../Data/preprocessed_training_data.csv", encoding="utf-8")
        test = pd.read_csv("../Data/preprocessed_testing_data.csv", encoding="utf-8")
        self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = train.text, test.text, train.choose_one, test.choose_one
        self._series_to_list() # changing the series into a list

    def _series_to_list(self):
        Encoder = LabelEncoder()
        self.Train_Y = Encoder.fit_transform(self.Train_Y)
        self.Test_Y = Encoder.fit_transform(self.Test_Y)     

    def naive_bayes(self):
        Naive = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('nb', naive_bayes.MultinomialNB())
        ])
        Naive.fit(self.Train_X, self.Train_Y)
        predictions_NB = Naive.predict(self.Test_X)
        self.print_results("Naive Bayes", predictions_NB)
        
        unconfident_labels = pd.read_csv("../Data/data_unconfident.csv").text
        new_predictions = Naive.predict(unconfident_labels)
        pd.DataFrame(new_predictions).to_csv("../Data/unconfident_predictions.csv", index=False)
        
        
    def logistic_regression(self):
        LogReg = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('log', LogisticRegression())
        ])
        LogReg.fit(self.Train_X, self.Train_Y)
        predictions_LR = LogReg.predict(self.Test_X)
        self.print_results("Logistic Regression", predictions_LR)
        
    def svm(self):        
        SVM = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('svm', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
        ])
        SVM.fit(self.Train_X, self.Train_Y)
        predictions_SVM = SVM.predict(self.Test_X)
        self.print_results("SVM", predictions_SVM)
        
    def random_forest(self):        
        RF = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('rf', RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt'))
        ])
        RF.fit(self.Train_X, self.Train_Y)
        predictions_RF = RF.predict(self.Test_X)
        self.print_results("Random Forest", predictions_RF)
        
    def decision_tree(self):        
        DT = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('dt', DecisionTreeClassifier())
        ])
        DT.fit(self.Train_X, self.Train_Y)
        predictions_DT = DT.predict(self.Test_X)
        self.print_results("Decision Tree", predictions_DT)

    def kmeans(self):
        KM = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,1))),
            ('k-means', KMeans(n_clusters=2))
        ])    
        KM.fit(self.Train_X)
        predictions_KM = KM.predict(self.Test_X)
        self.print_results("K-Means", predictions_KM)

    def print_results(self, model_name, predictions):
        print("------------------------------------------------------------------------")
        print(model_name + " Accuracy Score: ", accuracy_score(predictions, self.Test_Y)*100)
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.Test_Y, predictions))
        print('Mean Squared Error:', metrics.mean_squared_error(self.Test_Y, predictions))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.Test_Y, predictions)))
        print("Confusion Matrix:\n", confusion_matrix(self.Test_Y, predictions))
        print(classification_report(predictions, self.Test_Y))