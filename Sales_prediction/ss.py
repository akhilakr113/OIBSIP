import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

def prediction_sale(tv,radio,np):
    data = pd.read_csv("Advertising.csv")
    print(data.duplicated().sum())


    data = data.drop_duplicates()


    print(data.isnull().sum())




    X = data.drop(['Sales','Unnamed: 0'], axis=1)
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_model = GradientBoostingRegressor(random_state=0)
    gb_model.fit(x_train, y_train)

    predicted_gb = gb_model.predict(x_test)
    r2_gb = r2_score(y_test, predicted_gb)
    print("Gradient Boosting Mean Squared Error:", mean_squared_error(y_test, predicted_gb))
    print(f"Gradient Boosting R²: {r2_gb * 100:.2f}%")


    user_input = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [np],
   
    })

    predicted_price = gb_model.predict(user_input)
    predicted_pricee=float(predicted_price)


    return predicted_pricee

