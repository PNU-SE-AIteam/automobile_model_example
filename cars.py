#!/usr/bin/env python
# coding: utf-8



import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import regex as re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,model_selection,linear_model
from sklearn.metrics import r2_score
from model.linear_multiple import multipleLinearRegression





full_df=pd.read_csv("car details v4.csv")





msno.matrix(full_df)





full_df.info()



full_df.head()



def extract_engine_capacity(row):
    if pd.isna(row['Engine']):
        match = re.search(r'\b(\d+\.\d+)\b', row['Model'])
        if match:
            return str(int(float(match.group(1))*1000))
        else:
            return np.NaN
    return row['Engine']

full_df['Engine'] = full_df.apply(extract_engine_capacity, axis=1)


def extract_drivetrain(group):
    mode_value = group['Drivetrain'].mode().iloc[0]
    return group.fillna({'Drivetrain': mode_value})

full_df = full_df.groupby('Make').apply(extract_drivetrain)


full_df = full_df.dropna()
full_df['Engine'] = full_df['Engine'].str.replace('cc', '').astype(int)


analysis_df = full_df.drop(['Make', 'Model', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Drivetrain', 'Max Power', 'Max Torque'], axis=1)
sns.heatmap(analysis_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.1, linecolor='black')


full_df[['bhp', 'rpm power']] = full_df['Max Power'].str.split('@', expand=True)
full_df['bhp'] = full_df['bhp'].str.replace(' bhp', '').str.strip()
full_df['rpm power'] = full_df['rpm power'].str.replace(' rpm', '').str.strip()




full_df[['Nm', 'rpm torque']] = full_df['Max Torque'].str.split('@', expand=True)
full_df['Nm'] = full_df['Nm'].str.replace(' Nm', '').str.strip()
full_df['rpm torque'] = full_df['rpm torque'].str.replace(' rpm', '').str.strip()




full_df['bhp'].replace('', np.nan, inplace=True)
full_df['rpm power'].replace('', np.nan, inplace=True)
full_df['Nm'].replace('', np.nan, inplace=True)
full_df['rpm torque'].replace('', np.nan, inplace=True)




full_df = full_df.drop(['Max Power', 'Max Torque'], axis=1)
full_df.head()




full_df = full_df.dropna()



msno.matrix(full_df)



full_df.head()



year_mapping = {year: index for index, year in enumerate(sorted(full_df['Year'].unique()))}
full_df['Year'] = full_df['Year'].map(year_mapping)



full_df = full_df.dropna()



norm1 = ['Price']
norm2 = ['bhp', 'rpm power', 'Nm', 'rpm torque']
columns_to_normalize = ['Kilometer', 'Engine', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']

scaler = preprocessing.StandardScaler()
price_scaler = preprocessing.StandardScaler()
full_df[norm1] = price_scaler.fit_transform(full_df[norm1])
full_df[norm2] = scaler.fit_transform(full_df[norm2])
full_df[columns_to_normalize] = scaler.fit_transform(full_df[columns_to_normalize])



analysis_df = full_df.drop(['Make', 'Model', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Drivetrain'], axis=1)

sns.heatmap(analysis_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.1, linecolor='black')




full_df.head()



def one_hot_encode(df, columns):
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1, inplace=True)
    return df
full_df = one_hot_encode(full_df, columns=['Make', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Drivetrain'])




y = full_df[['Price']].values
X = full_df[['Engine']].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=69)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2score = r2_score(y_test, y_pred)
print("R2 Score of trivial model: %.2f" % r2score)



y_adv = y
X_adv = full_df[['Year','Kilometer','Engine','Length','Width','Height','Seating Capacity','Fuel Tank Capacity']].values




model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2score = r2_score(y_test, y_pred)
print("R2 Score of sklearn model: %.2f" % r2score)



model.score(X_test, y_test)





y_fin = y
X_fin = full_df[['Year','Kilometer','Engine','Length','Width','Height','Seating Capacity','Fuel Tank Capacity']].values



X_train, X_test, y_train, y_test = train_test_split(X_fin, y_fin, test_size=0.4, random_state=69)





mlr = multipleLinearRegression()
W, train_loss, num_epochs = mlr.train(X_train, y_train)
test_pred, test_loss = mlr.test(X_test, y_test, W)





r2 = mlr.score(y_test, y_pred)
rmse = mlr.score_rmse(y_test, y_pred)
st.write("Multiple linear regression model R2 Score: ", r2)
st.write("Multiple linear regression model Root Mean Squared Error: ", rmse)



year = st.sidebar.number_input('Manufacturing year', min_value=1988, max_value=2022, value=2017)
kilometer = st.sidebar.number_input('Kilometers driven', min_value=0, max_value=1000000, value=80000)
engine = st.sidebar.number_input('Engine capacity, cc', min_value=0, max_value=10000, value=1200)
length = st.sidebar.number_input('Length, mm', min_value=0, max_value=10000, value=3992)
width = st.sidebar.number_input('Width, mm', min_value=0, max_value=10000, value=1687)
height = st.sidebar.number_input('Height, mm', min_value=0, max_value=10000, value=1525)
seating_capacity = st.sidebar.number_input('Seating Capacity', min_value=0, max_value=6, value=5)
fuel_tank_capacity = st.sidebar.number_input('Fuel Tank Capacity, l', min_value=0, max_value=500, value=39)



if st.button('Predict Price'):
    data = pd.DataFrame([[year, kilometer, engine, length, width, height, seating_capacity, fuel_tank_capacity]], columns=['Year','Kilometer','Engine','Length','Width','Height','Seating Capacity','Fuel Tank Capacity'])
    data['Year'] = data['Year'].map(year_mapping)
    cols = ['Kilometer','Engine','Length','Width','Height','Seating Capacity','Fuel Tank Capacity']
    data[cols] = scaler.transform(data[cols])
    


    input_data_as_numpy_array = np.asarray(data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print(input_data_reshaped)
    X_sample_reshaped = input_data_reshaped.flatten()
    print(X_sample_reshaped)
    price = mlr.predict(W, X_sample_reshaped)
    price = price_scaler.inverse_transform(np.array([price]).reshape(1, -1))
    st.write('The predicted price of the car is ', str(price).replace('[[', '').replace(']]', ''))

