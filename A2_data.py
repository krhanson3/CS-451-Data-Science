#CS 451/551
#Coding Practice Session 2
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu


#Q1
import pandas as pd
df = pd.read_csv("student_data - Sheet1.csv")
ages = df["age"]
mean_age = round(ages.mean())
#fill missing values in the age column 
df['age'].fillna(mean_age, inplace=True)

#Q2
import pandas as pd
df0 = pd.read_csv("student_data - Sheet1.csv")
df0_cleaned = df0.drop_duplicates(subset=["name", "age", "grade", "address"])
print(df0_cleaned)

#Q3
import pandas as pd
df1 = pd.read_csv("sales - Sheet1.csv")
df1['price'] = df1['price'].str.replace(' USD', '', regex=False).astype(float)
print(df1)

#Q4
import pandas as pd
df2= pd.read_csv('employee_data - Sheet1.csv')
df2 = df2.rename(columns={
    "name": "employee_name",
    "age": "employee_age",
    "salary": "employee_salary"
})

print(df2.columns)

#Q5
import pandas as pd
df3 = pd.read_csv('employee_data - Sheet1.csv')
df3['status'] = df3['status'].map({'Active': 1, 'Inactive': 0})

print(df3)

#Q6
import pandas as pd
df4 = pd.read_csv("student_data - Sheet1.csv")
df4 = df4.drop(columns=['address'])
print(df4)

#Q7
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = sns.load_dataset('titanic')

titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].median())
titanic_df['embarked'] = titanic_df['embarked'].fillna(titanic_df['embarked'].mode()[0])

corr_matrix = titanic_df.corr(numeric_only=True)
print("Correlation Matrix:")
print(corr_matrix)

# --- Heatmap Visualization ---
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap - Titanic Dataset")
plt.show()

#Q8
import pandas as pd
df5 = pd.read_csv('employee_data - Sheet1.csv')
salary_min = df5['salary'].min()
salary_max = df5['salary'].max()

df5['normalized_salary'] = (df5['salary'] - salary_min) / (salary_max - salary_min)
print(df5)

#Q9
import pandas as pd
df6 = pd.read_csv('employee_data - Sheet1.csv')
mean_salary = df6['salary'].mean()
std_salary = df6['salary'].std()

num1 = 3 * std_salary
lower_bound = mean_salary - num1
upper_bound = mean_salary + num1

df6_cleaned = df6[(df6['salary'] >= lower_bound) & (df6['salary'] <= upper_bound)]
print(df6_cleaned)

#Q10
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df_versicolor = df[df['species'] == 'versicolor']

df_versicolor = df_versicolor.rename(columns={
    'petal length (cm)': 'PetalLength',
    'petal width (cm)': 'PetalWidth'
})

model = smf.ols(formula='PetalWidth ~ PetalLength', data=df_versicolor).fit()
print(model.summary())

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PetalLength', y='PetalWidth', data=df_versicolor, color='blue', label='Data')
plt.plot(df_versicolor['PetalLength'], model.predict(df_versicolor), color='red', label='Regression Line')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear Regression: Petal Length vs Petal Width (Versicolor)')
plt.show()
