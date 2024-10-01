import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def spam_or_not(user_input):

    data = pd.read_csv("spam.csv", encoding='ISO-8859-1')


    data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

    print("tail",data.tail())

    print(data.isnull().sum())

    print(data.duplicated().sum())

    data = data.drop_duplicates()

    print("after",data.duplicated().sum())

    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['v1'])

    X = data['v2']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    user_input_vectorized = vectorizer.transform([user_input])
    user_prediction = model.predict(user_input_vectorized)
    print("User Prediction:\n", user_prediction)
    if user_prediction[0] == 1:
        result="Spam"
        print("The message is: Spam")
        
    else:
        result="Not Spam"
        print("The message is not Spam")

    return result




