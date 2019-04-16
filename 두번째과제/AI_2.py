import os
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
from sklearn.metrics import precision_score, recall_score ,accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

TITANIC_PATH = os.path.join("datasets", "titanic")

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

def CreateTrainFeature(train_data):
    train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1 # 자신을 포함해야하니 1을 더합니다
    train_data["Fare"] = train_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    train_data["Embarked"].fillna('S', inplace=True)

    train_data['Age_cat'] = 0
    train_data.loc[train_data['Age'] < 10, 'Age_cat'] = 0
    train_data.loc[(10 <= train_data['Age']) & (train_data['Age'] < 20), 'Age_cat'] = 1
    train_data.loc[(20 <= train_data['Age']) & (train_data['Age'] < 30), 'Age_cat'] = 2
    train_data.loc[(30 <= train_data['Age']) & (train_data['Age'] < 40), 'Age_cat'] = 3
    train_data.loc[(40 <= train_data['Age']) & (train_data['Age'] < 50), 'Age_cat'] = 4
    train_data.loc[(50 <= train_data['Age']) & (train_data['Age'] < 60), 'Age_cat'] = 5
    train_data.loc[(60 <= train_data['Age']) & (train_data['Age'] < 70), 'Age_cat'] = 6
    train_data.loc[70 <= train_data['Age'], 'Age_cat'] = 7
    return train_data

def CreateTestFeature(test_data):
   
    
    test_data["FamilySize"] = test_data['SibSp'] + test_data["Parch"] + 1 # 자신을 포함해야하니 1을 더합니다
    
    test_data["Fare"] = test_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    
    test_data['Age_cat'] = 0
    test_data.loc[test_data['Age'] < 10, 'Age_cat'] = 0
    test_data.loc[(10 <= test_data['Age']) & (test_data['Age'] < 20), 'Age_cat'] = 1
    test_data.loc[(20 <= test_data['Age']) & (test_data['Age'] < 30), 'Age_cat'] = 2
    test_data.loc[(30 <= test_data['Age']) & (test_data['Age'] < 40), 'Age_cat'] = 3
    test_data.loc[(40 <= test_data['Age']) & (test_data['Age'] < 50), 'Age_cat'] = 4
    test_data.loc[(50 <= test_data['Age']) & (test_data['Age'] < 60), 'Age_cat'] = 5
    test_data.loc[(60 <= test_data['Age']) & (test_data['Age'] < 70), 'Age_cat'] = 6
    test_data.loc[70 <= test_data['Age'], 'Age_cat'] = 7

    return test_data

def Explanation():
    print("자질 설명")
    print("FamilySize = Parch + SibSp 로 하나 만들어주었고")
    print("Age_cat라는걸 나이는 10살 단위로 끊어서 하나의 자질을 만들어줬습니다.")
    print("train_data의 Embarked의 빈공간을 다 S로 넣어주었습니다.")

def PrintSVM(X_train, y_train,X_test,y_test):
    print("SVM")
    svm_clf = SVC(gamma="auto")
    svm_clf.fit(X_train, y_train)
    
    y_pred = svm_clf.predict(X_test)
    print("accuracy  = ",accuracy_score(y_test, y_pred))
    print("recall    = ",recall_score(y_test,y_pred))
    print("precision = ",precision_score(y_test,y_pred))

def PrintRandomForest(X_train, y_train,X_test,y_test):
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf.fit(X_train, y_train)

    y_pred = forest_clf.predict(X_test)

    print("RandomForestClassifier")
    print("accuracy  = ",accuracy_score(y_test, y_pred))
    print("recall    = ",recall_score(y_test,y_pred))
    print("precision = ",precision_score(y_test,y_pred))
    
if __name__ == "__main__":
    
    train_data = load_titanic_data("train.csv")
    test_data = load_titanic_data("test.csv")
    y_data = load_titanic_data("gender_submission.csv")

    train_data = CreateTrainFeature(train_data)#train_data 자질 셋팅
    test_data = CreateTestFeature(test_data)#test_data 자질 셋팅
        
    num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age_cat", "FamilySize", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    X_train = preprocess_pipeline.fit_transform(train_data)
    y_train = train_data["Survived"]

    X_test = preprocess_pipeline.fit_transform(test_data)
    y_test = y_data["Survived"]


    Explanation()

    print("===========================================")
    
    PrintSVM(X_train, y_train,X_test,y_test)
    
    print("===========================================")
    
    PrintRandomForest(X_train, y_train,X_test,y_test) 

   
   

        
