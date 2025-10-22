#CS 451/551
#Assignment 2
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#region Q1: Definitions
# A: Ranking = d: ordering items based on scores from a scoring function
print('A, d')
# B: Scraping = b: downloading the right set of pages for analysis
print('B, b')
# C: Spidering = c: stripping downloaded webpages for useful content 
print('C, c')
# D: Scoring Function = a: reducing data to a single value to be more useful
print('D, a')
print('\n')
#endregion

#region Q2: Definitions
# A: Z-Score = b: A scoring function that measures how distant in standard deviations a variable is from the mean of said variable
print('A, b')
# B: Elo Rankings = e: A scoring function that consolidates sequences of binary comparisons such as in sports matches to compute probability of win or player skill level
print('B, e')
# C: Page Rank = c: A scoring function that ranks the importance of webpages based on received links from other webpages
print('C, c')
# D: Logit Function = a: A scoring function that converts a continuous value into a probability
print('D, a')
# E: BMI = d: A scoring function that consolidates an individual’s height and weight into a single measure of how overweight, normal or underweight they are
print('E, d')
print('\n')
#endregion

#region Q3: RottenTomatoes Data Scraping
from bs4 import BeautifulSoup
import pandas as pd
import re #regular expression to parse the date seciton in the div 

with open("data-sheets/rotten_tomatoes_best_movies.html", "r", encoding="utf-8") as f: #doc, mode (read), encoding type
    soup = BeautifulSoup(f, "html.parser")

months_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
titles, critics, users, months, days, years = [], [], [], [], [], []

for div in soup.find_all("div", class_="flex-container"): #all the data is together at the bottom of the html file
    try:
        title_tag = div.find("span", {"data-qa": "discovery-media-list-item-title"})
        title = title_tag.get_text(strip=True) if title_tag else None

        critic_tag = div.find("rt-text", {"slot": "criticsScore"}) #the specific spot where the numbers are mentioned
        user_tag   = div.find("rt-text", {"slot": "audienceScore"})
        critic = int(critic_tag.get_text(strip=True).replace("%", "")) if critic_tag else None #turning the string into the specific int value 
        user   = int(user_tag.get_text(strip=True).replace("%", "")) if user_tag else None

        date = div.find("span", {"data-qa": "discovery-media-list-item-start-date"})
        month = day = year = None
        if date:
            match = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", date.text)
            if match:
                month = months_map.get(match.group(1)[:3])
                day, year = int(match.group(2)), int(match.group(3))

        if all([title, critic, user, month, day, year]):
            titles.append(title)
            critics.append(critic)
            users.append(user)
            months.append(month)
            days.append(day)
            years.append(year)

        if len(titles) >= 300:  
            break

    except Exception:
        continue

movies_lowdim = pd.DataFrame({
    "title": titles, "critic_rating": critics, "user_rating": users,
    "month": months, "day": days, "year": years
    }).astype({
        "title": "string", "critic_rating": "int", "user_rating": "int",
        "month": "int", "day": "int", "year": "int"
})

print(movies_lowdim.head(10))
print("\nShape of dataset:", movies_lowdim.shape)
print('\n')
#endregion

#region Q4: Frequency Distributions 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#copy dataset 
df = movies_lowdim.copy()

# Helper function for statistics
def describe(series):
    return series.mean(), series.std(), np.percentile(series, 25), np.percentile(series, 75)

user_stats = describe(df["user_rating"])
critic_stats = describe(df["critic_rating"])

print("User ratings (mean, std, 25th, 75th):", user_stats)
print("Critic ratings (mean, std, 25th, 75th):", critic_stats)

# Plot histograms
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.hist(df["user_rating"], bins=15, color='skyblue', edgecolor='black')
plt.title("User Ratings")
plt.subplot(2, 1, 2)
plt.hist(df["critic_rating"], bins=15, color='lightgreen', edgecolor='black')
plt.title("Critic Ratings")
plt.tight_layout()
plt.show()

# Boxplot comparison
plt.figure(figsize=(8, 6))
plt.boxplot([df["user_rating"], df["critic_rating"]],
            labels=["User Ratings", "Critic Ratings"])
plt.title("User Ratings vs Critic Ratings")
plt.show()

#convert to datetime and extract week
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
df["week"] = df["date"].dt.isocalendar().week.astype(int)

#weekly averages
weekly_avg = df.groupby("week")[["user_rating", "critic_rating"]].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(weekly_avg["week"], weekly_avg["user_rating"], color="red", label="User Ratings")
plt.plot(weekly_avg["week"], weekly_avg["critic_rating"], color="green", label="Critic Ratings")
plt.xlabel("Week of Year")
plt.ylabel("Average Rating (%)")
plt.title("Weekly Timeline of Ratings")
plt.legend()
plt.show()

# Correlation
corr = df["user_rating"].corr(df["critic_rating"])
print(f"Correlation between user and critic ratings: {corr:.3f}")

# Bucket function
def bucket_string(x):
    if x > 95: return "super_fresh"
    elif x > 90: return "somewhat_fresh"
    elif x > 80: return "okay"
    elif x > 75: return "somewhat_rotten"
    else: return "very_rotten"

df["user_bucket"] = df["user_rating"].apply(lambda x: bucket_string(x) if pd.notna(x) else "very_rotten")
df["critic_bucket"] = df["critic_rating"].apply(lambda x: bucket_string(x) if pd.notna(x) else "very_rotten")

# Bar plots with fill_value=0
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
df["user_bucket"].value_counts().reindex(
    ["super_fresh", "somewhat_fresh", "okay", "somewhat_rotten", "very_rotten"],
    fill_value=0
).plot(kind="bar", color='skyblue')
plt.title("User Rating Buckets")

plt.subplot(2, 1, 2)
df["critic_bucket"].value_counts().reindex(
    ["super_fresh", "somewhat_fresh", "okay", "somewhat_rotten", "very_rotten"],
    fill_value=0
).plot(kind="bar", color='lightgreen')
plt.title("Critic Rating Buckets")
plt.tight_layout()
plt.show()

print("Most frequent user bucket:", df["user_bucket"].mode()[0])
print("Most frequent critic bucket:", df["critic_bucket"].mode()[0])

# Numeric buckets
def bucket_number(x):
    if x > 90: return 3
    elif x > 80: return 2
    else: return 1

df["user_bucket_number"] = df["user_rating"].apply(lambda x: bucket_number(x) if pd.notna(x) else 1)
df["critic_bucket_number"] = df["critic_rating"].apply(lambda x: bucket_number(x) if pd.notna(x) else 1)

print("\nAdded columns: user_bucket_number, critic_bucket_number")
print(df.head(10))
#endregion

#region Q5: DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import numpy as np

df = df.copy()

X1 = df[["critic_bucket_number"]]
X2 = df[["critic_rating"]]
X3 = df[["month", "critic_bucket_number"]]
X4 = df[["month", "year", "day", "critic_bucket_number", "critic_rating"]]
y = df["user_bucket_number"]

def evaluate_model(model, X, y, name, scenario_label):
    cv = cross_validate(model, X, y, cv=5, return_train_score=True)
    train_acc = np.mean(cv["train_score"])
    test_acc = np.mean(cv["test_score"])
    print(f"Scenario {scenario_label}: {train_acc:.3f} {test_acc:.3f}")
    return (name, scenario_label, test_acc)

results = []

print("Algorithm: DecisionTreeClassifier")
for i, X in enumerate([X1, X2, X3, X4], start=1):
    model = DecisionTreeClassifier(random_state=42)
    results.append(evaluate_model(model, X, y, "DecisionTreeClassifier", i))

print("\nAlgorithm: RandomForestClassifier")
for i, X in enumerate([X1, X2, X3, X4], start=1):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    results.append(evaluate_model(model, X, y, "RandomForestClassifier", i))

best_model, best_scenario, best_score = max(results, key=lambda x: x[2])
print(f"\nBest Test Performance: {best_model}, Scenario {best_scenario}")

print('\n')
#endregion

#region Q6: Encoding Definitions
# A: one-hot encoding = b: encoding the unique values of a feature into multiple binary flag columns
print('A, b')
# B: frequency encoding = a: encoding of categorical levels of feature to values between 0 and 1 based on their relative frequency
print('B, a')
# C: target-mean encoding = c: encoding the feature values with mean values of an outcome variable
print('C, c')
print('\n')
#endregion

#region Q7: High-dimensional Movie Metadata and Reviews
import requests
from bs4 import BeautifulSoup
import random
import datetime as dt
import re

sample_movies = df.sample(15, random_state=42)[["title", "user_bucket", "critic_bucket"]]

print("Random 15 Movies:")
for _, row in sample_movies.iterrows():
    print(f"{row['title']} | User: {row['user_bucket']} | Critic: {row['critic_bucket']}")

movie_raw = {}

def fetch_reviews(movie_slug):
    url = f"https://www.rottentomatoes.com/m/{movie_slug}/reviews"
    reviews_list = []
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        review_blocks = soup.find_all("review-speech-balloon") or soup.find_all("div", class_="review_table_row")
        for rb in review_blocks[:5]:
            try:
                author = rb.find("a", class_="display-name")
                author = author.get_text(strip=True).lower() if author else "not_available"
                
                publication = rb.find("em", class_="subtle")
                publication = publication.get_text(strip=True).lower() if publication else "not_available"
                
                text = rb.find("div", class_="the_review")
                text = text.get_text(strip=True).lower() if text else "not_available"
                
                score_tag = rb.find("div", class_="small subtle review-link")
                score = score_tag.get_text(strip=True).lower() if score_tag else "not_available"
                
                fresh = 1 if rb.find("span", class_="fresh") else 0
                
                reviews_list.append({
                    "review_author": author,
                    "review_publication": publication,
                    "review_text": text,
                    "review_fresh": fresh,
                    "review_score": score
                })
            except Exception:
                continue
    except Exception as e:
        print(f"Could not fetch {url}: {e}")
    
    while len(reviews_list) < 5:
        reviews_list.append({
            "review_author": "not_available",
            "review_publication": "not_available",
            "review_text": "not_available",
            "review_fresh": 0,
            "review_score": "not_available"
        })
    
    return reviews_list

for _, row in sample_movies.iterrows():
    slug = re.sub(r'[^a-z0-9]+', '-', row["title"].lower()).strip('-')
    
    movie_raw[slug] = {
        "movie_title": row["title"].lower(),
        "designation": "not_available",
        "genres": ["not_available"],
        "director_name": "not_available",
        "in_theatres": dt.datetime(2025, 1, 1),
        "streaming": dt.datetime(2025, 10, 1),
        "reviews": fetch_reviews(slug),
        "user_bucket": row["user_bucket"],
        "critic_bucket": row["critic_bucket"]
    }
first_movie = next(iter(movie_raw.keys()))
print("\nSample Movie Entry:")
print(movie_raw[first_movie])

print('\n')
#endregion

#region Q8: High-dimensional movies_highdim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

movies_list = []

for movie_key, movie in movie_raw.items():
    row = {}
    row['title'] = movie['movie_title']
    row['user_bucket_number'] = bucket_number(movie.get('user_bucket', 'not_available'))
    row['critic_bucket_number'] = bucket_number(movie.get('critic_bucket', 'not_available'))
    
    row['designation'] = movie['designation']
    
    for genre in movie.get('genres', ['not_available']):
        if genre != 'not_available':
            row[genre] = 1
    
    row['director_name'] = movie['director_name']
    
    row['in_theatres_week'] = movie['in_theatres'].isocalendar()[1]
    row['in_theatres_month'] = movie['in_theatres'].month
    row['in_theatres_year'] = movie['in_theatres'].year
    
    row['streaming_week'] = movie['streaming'].isocalendar()[1]
    row['streaming_month'] = movie['streaming'].month
    row['streaming_year'] = movie['streaming'].year
    
    for i, review in enumerate(movie['reviews'], start=1):
        row[f'review{i}_author'] = review['review_author']
        row[f'review{i}_publication'] = review['review_publication']
        
        score = review['review_score'].lower()
        if score in ['a', 'b', 'c']:
            score_map = {'a': 1, 'b': 2, 'c': 3}
            row[f'review{i}_score'] = score_map.get(score, 4)
        elif '/' in score:
            try:
                val, scale = map(float, score.split('/'))
                scaled = int(np.floor(val * 5 / scale))
                row[f'review{i}_score'] = max(1, min(5, scaled))
            except:
                row[f'review{i}_score'] = 3
        elif score.replace('.', '', 1).isdigit():
            row[f'review{i}_score'] = min(5, max(1, int(np.floor(float(score)))))
        else:
            row[f'review{i}_score'] = 3
    
    review_texts = [r['review_text'] for r in movie['reviews']]
    row['review_texts_combined'] = " ".join(review_texts)
    
    movies_list.append(row)

movies_df = pd.DataFrame(movies_list)

unique_designations = {val: i+1 for i, val in enumerate(movies_df['designation'].unique())}
movies_df['designation'] = movies_df['designation'].map(unique_designations)

unique_directors = {val: i+1 for i, val in enumerate(movies_df['director_name'].unique())}
movies_df['director_name'] = movies_df['director_name'].map(unique_directors)

for i in range(1, 6):
    col_author = f'review{i}_author'
    col_pub = f'review{i}_publication'
    
    unique_authors = {val: j+1 for j, val in enumerate(movies_df[col_author].unique())}
    unique_pubs = {val: j+1 for j, val in enumerate(movies_df[col_pub].unique())}
    
    movies_df[col_author] = movies_df[col_author].map(unique_authors)
    movies_df[col_pub] = movies_df[col_pub].map(unique_pubs)

all_genres = set()
for movie in movie_raw.values():
    all_genres.update(movie['genres'])
all_genres.discard('not_available')
for genre in all_genres:
    if genre not in movies_df.columns:
        movies_df[genre] = 0

vectorizer = CountVectorizer(binary=True, stop_words='english')
bow_matrix = vectorizer.fit_transform(movies_df['review_texts_combined'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=[f'BOW{i+1}' for i in range(bow_matrix.shape[1])])

movies_highdim = pd.concat([movies_df.drop(columns=['review_texts_combined']), bow_df], axis=1)

print("High-dimensional DataFrame shape:", movies_highdim.shape)
print(movies_highdim.head())

print('\n')
#endregion

#region Q9: Methods Definitions
# A: Feature Selection = c: finding a subset of candidate features to produce a better ML model 
print('A, c')
# B: Wrapper Methods = d: iterative search for a subset of optimal features through shrinking or growing the feature set 
print('B, d')
# C: Filter Methods = a: assessing feature importance using model-agnostic criteria to select the best features
print('C, a')
# D: Embedded Methods = b: reducing the size of the feature set through model learning 
print('D, b')

print('\n')
#endregion

#region Q10: Analysis Definitions
# A: PCA = a: transform features set into orthogonal features to maximize variance captured
print('A, a')
# B: LDA = b: creating a projection that maximizes the seperation between members of different classes
print('B, b')

print('\n')
#endregion

#region Q11
#endregion

#region Q12
#endregion

#region Q13
#endregion

#region Q14
#endregion

#region Q15
#endregion

#region Q16
#endregion

#region Q17
#endregion

#region Q18
#endregion

#region Q19
#endregion

#region Q20

#endregion

#region Q21
#endregion

#region Q22
#endregion

#region Q23
#endregion

#region Q24
#endregion

#region Q25
#endregion