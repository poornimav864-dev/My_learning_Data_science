import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Load the csv file
data = pd.read_csv("Food_Delivery_Times.csv")

#Print the columns name of the data
col_names = data.columns

#Print the summary statistics of the data
describe = data.describe().T
print(describe)
print("--------------------")

#Print the type of the data
d_types = data.dtypes
print(d_types)
print("_____________________")

#To return one columns
data_column = data['Time_of_Day']
#To return more than one columns
data_col = data[['Time_of_Day', 'Weather']]

#To find the information of the data
data_information = data.info()
print(data_information)
print("_______________________")

#To find the duplicate values
print(f'Number of duplicates in this dataset: {data.duplicated().sum()}')
print("<<<<<<<<<<<<<<<<<<<<<<<")

#Fill the categorical columns using Simple Imputer

imputer = SimpleImputer(strategy='most_frequent')
data[['Weather', 'Traffic_Level', 'Time_of_Day']] = (
    imputer.fit_transform(data[['Weather', 'Traffic_Level', 'Time_of_Day']]))

# Another method to Fill categorical columns using mode
# for col in ['Weather', 'Traffic_Level', 'Time_of_Day']:
#     data[col] = data[col].fillna(data[col].mode()[0])
"""mode selects the most frequent values in the columns, [0] selects the first mode"""
# Fill numerical column with median
data['Courier_Experience_yrs'] = data['Courier_Experience_yrs'].fillna(data['Courier_Experience_yrs'].median())

#Now checking the null datas or missing datas
null_check = data.isnull().sum()
print("Null entries:\n",null_check)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^")

#Converting categorical data into numerical data using OneHotEncoding method
encoded_data = pd.get_dummies(data,
                              columns=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'],
                              drop_first=True)

#Define features and target
x = encoded_data.drop(['Order_ID', 'Delivery_Time_min'], axis=1)
y = encoded_data ['Delivery_Time_min']

#Split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Define the model
model_1 = LinearRegression()
model_1.fit(x_train, y_train)

#Predict the value
y_predict = model_1.predict(x_test)
print("PREDICTED OUTPUT IS:", y_predict[0:10])

# results = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
# print(results.head())
# print("............................")

#Evaluating metrics
print("Linear Regression MAE:", mean_absolute_error(y_test, y_predict))
print("Linear Regression R2_score:", r2_score(y_test, y_predict))
print("****************************")

model_2 = DecisionTreeRegressor()
model_2.fit(x_train, y_train)

y_predict = model_2.predict(x_test)
print("PREDICTED OUTPUT IS:", y_predict[0:10])

print("Decision Tree Regression MAE:", mean_absolute_error(y_test, y_predict))
print("Decision Tree Regression R2_score:", r2_score(y_test, y_predict))
print("****************************")

model_3 = RandomForestRegressor()
model_3.fit(x_train, y_train)

y_predict = model_3.predict(x_test)
print("PREDICTED OUTPUT IS:", y_predict[0:10])

print("Random Forest Regression MAE:", mean_absolute_error(y_test, y_predict))
print("Random Forest Regression R2_score:", r2_score(y_test, y_predict))
print("****************************")

"""After evaluating multiple regression models — 
Linear Regression, Decision Tree Regression, and Random Forest Regression.
Based on these metrics, Linear Regression is selected as the final model for predicting delivery times."""