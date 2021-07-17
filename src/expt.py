import dvc.api
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pickle
import json
from sklearn.ensemble import RandomForestClassifier

with dvc.api.open(repo="https://github.com/jayashan10/mlops_Assignment", path="data/creditcard.csv", mode="r") as fd:
    print("inside the with ")
    df = pd.read_csv(fd)

df_train, df_test = train_test_split(df, test_size=0.2)
df_train.to_csv('../data/processed/train.csv')
df_test.to_csv('../data/processed/test.csv')


#tree_classifier = DecisionTreeClassifier(criterion="entropy")
forest_classifier = RandomForestClassifier()
#tree_classifier.fit(df_train.drop(columns='Class'), df_train['Class'])
forest_classifier.fit (df_train.drop(columns = 'Class'), df_train['Class'])
#y_test_pred = tree_classifier.predict(df_test.drop(columns='Class'))
y_test_pred = forest_classifier.predict(df_test.drop(columns='Class'))

print(f"f1 score is: {f1_score(df_test['Class'], y_test_pred)}")
print(f"the accuracy score is: {accuracy_score(df_test['Class'], y_test_pred)}")

filename = '../models/tree_classifier.sav'
pickle.dump(forest_classifier, open(filename, "wb" ))


my_dict = {}
my_dict['f1_score']=f1_score(df_test['Class'], y_test_pred)
my_dict['accuracy_score']=accuracy_score(df_test['Class'], y_test_pred)
with open("../metrics/acc_f1.json", "w") as d:
    json.dump(my_dict, d)

