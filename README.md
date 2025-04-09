# Chocolate Sales Prediction

Project Overview:

  *This project aims to predict chocolate sales using machine learning. It utilizes a Random Forest Regression model to analyze historical sales data and make predictions on future sales. 
  
  *The project provides insights into sales trends, feature importance, and overall sales performance.

Objectives:

 * Train a machine learning model to predict chocolate sales.
   
 * Make predictions on new or future sales data.
   
 * Visualize actual vs. predicted sales, residual distribution, feature importance, and total sales by country.

installation:

  *pip pandas numpy matplotlib seaborn scitkit-learn
  
Prepare Data:

  *chocolate sales data in a CSV file with the following columns: `Date`, `Country`, `Product`, `Sales Person`, and `Amount`.

feature:

   *Sales Predictions: The make predictions on new or future sales data.
   
   *Visualization: The visualizations to help you understand sales trends.
   
   *Actual vs. Predicted Sales: A scatter plot comparing the model's predictions with actual sales values.
   
   *Residual Distribution: A histogram showing the distribution of prediction errors (residuals).
   
   *Feature Importance: A bar chart highlighting the most influential features in the sales prediction model.
   
   *Total Sales by Country: A bar chart displaying the total sales for each country.


Libraries:


 *pandas
 *numpy
 *matplotlib
 *seaborn
 *scikit-learn
 
**command-line interface**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/Chocolate Sales.csv")
df
df.head()
df.tail()
df.isnull().sum()
df['Amount'] = df['Amount'].replace('[\$,]', '',regex=True).astype(int)
df
df.info()
df.describe()
df.columns
df.shape
df.dropna()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.head()
df.value_counts()
X = df[['Product', 'Day', 'Month', 'Year']] # Select features for X
y = df['Amount']  # Select target variable for y
label_encoder = LabelEncoder()
X.loc[:, 'Product'] = label_encoder.fit_transform(X['Product'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='Blue', alpha=0.9)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales ")
plt.ylabel("Predicted Sales ")
plt.title("Actual vs. Predicted Sales (Random Forest)")
plt.show()
# Calculate residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feature_importance, y=feature_importance.index, hue=feature_importance.index,palette="viridis",dodge=False, legend=False)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Sales Prediction Model")
plt.show()
plt.figure(figsize=(10,5))
df.groupby('Country')['Amount'].sum().sort_values().plot(kind='barh', color='orange')
plt.xlabel("Total Sales ")
plt.ylabel("Country")
plt.title("Total Sales by Country")
plt.show()
