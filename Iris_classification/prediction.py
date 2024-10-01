# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
def pred(sepal_len,sepal_wid,petal_len,petal_wid):

    df = pd.read_csv("iris.csv")
    print(df.head())

    X = df.drop(['Species','Id'], axis=1)  
    y = df['Species']      

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier()


    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
        
    accuracy = accuracy_score(y_test, y_pred)
        
    print(f' Accuracy: {accuracy * 100:.2f}%')

    user_input = pd.DataFrame({
    'SepalLengthCm': [sepal_len],
    'SepalWidthCm': [sepal_wid],
    'PetalLengthCm': [petal_len],
    'PetalWidthCm': [petal_wid],

    })

    predicted_species= model.predict(user_input)
    if predicted_species:
        result=str(predicted_species)

    return result




