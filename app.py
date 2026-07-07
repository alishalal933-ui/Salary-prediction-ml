import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Yahan apna CSV ka naam likho
df = pd.read_csv('salary_data.csv') 

X = df[['Experience', 'Age']]  # 2 features
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'salary_model.pkl')
print("Model ban gaya dono features ke saath")
