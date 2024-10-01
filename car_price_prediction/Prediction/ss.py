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

def prediction_price(year,pp,kd,ft,st,owner,car_name):
    data = pd.read_csv("car data.csv")

    data = data.drop_duplicates()

    label_encoder = LabelEncoder()
    data['Fuel_Type'] = label_encoder.fit_transform(data['Fuel_Type'])
    data['Seller_Type'] = label_encoder.fit_transform(data['Selling_type'])

    original_car_names = data['Car_Name'].copy()

    data = pd.get_dummies(data, columns=['Car_Name'], drop_first=True)

    X = data.drop(['Selling_Price', 'Selling_type', 'Transmission'], axis=1)
    y = data['Selling_Price']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_model = GradientBoostingRegressor(random_state=0)
    gb_model.fit(x_train, y_train)

    predicted_gb = gb_model.predict(x_test)
    r2_gb = r2_score(y_test, predicted_gb)
    user_input = pd.DataFrame({
    'Year': [year],
    'Present_Price': [pp],
    'Kms_Driven': [kd],
    'Fuel_Type': [ft],
    'Seller_Type': [st],
    'Owner': [owner]
    })

    car_name_dummies = pd.get_dummies(original_car_names, drop_first=True).columns

    car_name_encoded = pd.DataFrame(0, index=[0], columns=car_name_dummies)
    if f'Car_Name_{car_name}' in car_name_encoded.columns:
        car_name_encoded.loc[0, f'Car_Name_{car_name}'] = 1
    else:
        pass

    user_input = pd.concat([user_input, car_name_encoded], axis=1)

    user_input = user_input.reindex(columns=x_train.columns, fill_value=0)

    predicted_price = gb_model.predict(user_input)
    predicted_pricee=float(predicted_price)

    print("mmmmmmmmmmmmm", predicted_pricee)
    result = predicted_pricee 

    return result



