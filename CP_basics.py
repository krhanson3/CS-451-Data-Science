#CS 451/551
#Coding Practice Session 1
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#Q1
import pandas as pd
dataset = {
    'Movies': ['50 First Dates', 'How to train your dragon', 
                'how to lose a guy in ten days', 'deadpool', 'iron man'],
    'Likes' : ['romantic', 'fantastic story', 'good plot', 'action packed', 'great actors'], 
    'Dislikes' : ['slow! slow slow', 'n/a', 'took too long......', 'too much language', 'sad']
}
df = pd.DataFrame(dataset)
# print(df)

#Q2
import string
df["Movies"]= df['Movies'].str.replace(" ", "")
edited_likes=[]
for s in df['Likes']:
    edited_string = ""
    for char in s:
        if char not in string.punctuation:
            edited_string += char
    edited_likes.append(edited_string)
df['Likes'] = edited_likes
edited_dislikes=[]
for s in df['Dislikes']:
    edited_string = ""
    for char in s:
        if char not in string.punctuation:
            edited_string += char
    edited_dislikes.append(edited_string)
df['Dislikes'] = edited_dislikes
# print (df)

#Q3
Unique_Word_Count = []

for index, row in df.iterrows():
    likes_words = row['Likes'].split()
    dislikes_words = row['Dislikes'].split()
    unique_words = len(set(likes_words + dislikes_words))
    Unique_Word_Count.append(unique_words)

df['Unique_Word_Count'] = Unique_Word_Count

print(df)

#Q4 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.bar(df['Movies'], df['Unique_Word_Count'], color='red')
plt.xlabel("Movies")
plt.ylabel("Unique Word Count")
plt.title("Unique Word Count of Reviews per Movie ")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Q5
from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
iris_df = pd.concat([X, y], axis=1)
iris_df.columns = [col.replace(" (cm)", "").replace(" ", "_").lower() for col in iris_df.columns]
print(iris_df.head())


#Q6 
import matplotlib.pyplot as plt
plt.scatter(data=iris_df, x="sepal_length", y="petal_length", color="red")
plt.title("Sepal Length vs. Petal Length by Class")
plt.show()

#Q7
summary = iris_df.describe().T[["mean", "std"]]
summary["median"] = iris_df.select_dtypes(include='number').median()

print(summary)

#Q8
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_df.iloc[:, 0:4])
scaled_df = pd.DataFrame(scaled_features, columns=iris.feature_names)

print(scaled_df.head())

#Q9

sepal_petal_corr = iris_df["sepal_length"].corr(iris_df["petal_length"])
print(f"\nCorrelation between Sepal and Petal Length: {sepal_petal_corr:.2f}")
correlation_matrix = iris_df.iloc[:, 0:4].corr()

print(correlation_matrix)


