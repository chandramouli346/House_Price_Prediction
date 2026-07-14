import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
df=pd.read_csv("data/Housing.csv")
print(df.head(10))

#shape of the dataset
print(df.shape)

#columns of the dataset
print(df.columns)

#Data set information
print(df.info())

#statistical summary of the dataset
print(df.describe())

#missing values in the dataset
print("Null Values in the data set",df.isnull().sum())


#Print duplicate rows in the dataset
duplicates=df.duplicated().sum()
print("Duplicates rows in the dataset",duplicates)


#Price Distribution


plt.figure(figsize=(10,6))
sns.histplot(df["price"],bins=30,kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.show()

#Scatter plot 

plt.figure(figsize=(10,6))
sns.scatterplot(x="area",y="price",data=df)
plt.title("Area VS Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

#Box plot for price
plt.figure(figsize=(10,6))
sns.boxplot(y=df["price"])
plt.title("Box plot of House prices")
plt.ylabel("price")
plt.show()



#Count plot - Furnishing Status
plt.figure(figsize=(6,6))
sns.countplot(x="furnishingstatus",data=df)
plt.title("Furnishing Status Count")
plt.xlabel("Furnishing Status")
plt.ylabel("Count")
plt.show()


#Correlation Heatmap
plt.figure(figsize=(8,8))
correlation=df.corr(numeric_only=True)
sns.heatmap(correlation,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#Categorical Columns

categorical_columns=df.select_dtypes(include="object")
print("\nCategorical Columns in the dataset",categorical_columns.columns)


#LAbel Encoding for categorical columns

label_encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save all encoders
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("✅ Label Encoders Saved Successfully!")



#Machine learning model training and testing

#train test split
x_train,x_test,y_train,y_test=train_test_split(df.drop("price",axis=1),df["price"],test_size=0.2,random_state=42)
print("\nShape of x_train:",x_train.shape)
print("Testing Features shape : ",x_test.shape)

print("\nShape of y_train:",y_train.shape)
print("Testing Target shape : ",y_test.shape)

#Create a linear regression model

model=LinearRegression()
model.fit(x_train,y_train)#Fit means train the model with the training data
y_pred=model.predict(x_test)#Predict the target variable for the testing data
print("\nPredicted values for the testing data")
print(y_pred)

#calculate the metrics for the model
y_pred=model.predict(x_test)

#Model Evaluation Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


#Random Forest Regressor Model

rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)
y_pred_rf=rf_model.predict(x_test)          
rf_mae = mean_absolute_error(y_test, y_pred_rf)

# ----------------------------
# Save Feature Importance
# ----------------------------

feature_importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

joblib.dump(feature_importance, "model/feature_importance.pkl")

print("✅ Feature Importance Saved Successfully!")

#Evaluate the Random Forest model
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
print("\n===== Random Forest Results =====")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


#Save the best model to a file using joblib


# Save the Random Forest model inside the model folder
joblib.dump(rf_model, "model/best_model.pkl")

# Save feature names
feature_names = x_train.columns.tolist()
print(feature_names)
joblib.dump(feature_names, "model/feature_names.pkl")

print("✅ Model Saved Successfully!")
print("✅ Feature Names Saved Successfully!")
