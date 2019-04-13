import os

TITANIC_PATH = os.path.join("datasets", "titanic")

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

def num_pipeline(data):
    num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
    num_pipeline.fit_transform(data)
    return num_pipeline

def cat_pipeline(data):
    cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
    cat_pipeline.fit_transform(data)
    return cat_pipeline

if __name__ == "__main__":
    
    train_data = load_titanic_data("train.csv")
    test_data = load_titanic_data("test.csv")

    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline(train_data)),
        ("cat_pipeline", cat_pipeline(train_data)),
    ])
    
    X_train = preprocess_pipeline.fit_transform(train_data)
    y_train = train_data["Survived"]

    
    
    svm_clf = SVC(gamma="auto")
    svm_clf.fit(X_train, y_train)

      
    X_test = preprocess_pipeline.transform(test_data)
    y_pred = svm_clf.predict(X_test)

    
      
    svm_scores = cross_val_score(svm_clf, X_test, y_pred, cv=10)
    
    print(y_pred)
    
##
##    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
##    forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
##    print(forest_scores.mean())

##    print("Precision: {:.2f}%".format(100 * precision_score(forest_scores, y_pred)))
##    print("Recall: {:.2f}%".format(100 * recall_score(forest_scores, y_pred)))  








