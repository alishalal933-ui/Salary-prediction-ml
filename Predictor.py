import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {'Experience': [1,2,3,4,5,6,7,8,9,10],
        'Salary': [30000,35000,40000,50000,55000,60000,70000,75000,85000,90000]}
df = pd.DataFrame(data)

X = df[['Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

experience = 5.5
predicted_salary = model.predict([[experience]])
print(f"Predicted salary for {experience} years experience: PKR {predicted_salary[0]:.0f}")
