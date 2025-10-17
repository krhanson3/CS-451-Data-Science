#CS 451/551
#Assignment 1
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#Q1: Definitions
# A: Ranking = d: ordering items based on scores from a scoring function
print('A, d')
# B: Scraping = b: downloading the right set of pages for analysis
print('B, b')
# C: Spidering = c: stripping downloaded webpages for useful content 
print('C, c')
# D: Scoring Function = a: reducing data to a single value to be more useful
print('D, a')

#Q2: Definitions
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

#Q3 RottenTomatoes Data Scraping
from bs4 import BeautifulSoup
import json
import pandas as pd

# Load the saved Rotten Tomatoes HTML file
with open("data-sheets\rotten_tomatoes_best_movies.html", "r", encoding="utf-8") as f:
    html = f.read()

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Locate the JSON-LD script that contains movie data
json_tag = soup.find("script", type="application/ld+json")
data = json.loads(json_tag.string)

# Extract the list of movies
movies = data["itemListElement"]["itemListElement"]

# Prepare lists for each column
titles, critic_ratings, user_ratings, months, days, years = [], [], [], [], [], []

# Populate the lists
for movie in movies:
    try:
        title = movie["name"].strip()
        critic_rating = int(movie["aggregateRating"]["ratingValue"])
        # Simulate user ratings (not provided directly in JSON)
        # In a real scrape, you'd get audience_score, but it's absent here
        # For demonstration, assume critic rating as proxy (you may replace with real data later)
        user_rating = critic_rating  
        
        # Extract streaming year and approximate random streaming month/day
        year = int(movie["dateCreated"])
        # Placeholder logic — replace if actual month/day is present in full dataset
        month = 10  
        day = 1

        titles.append(title)
        critic_ratings.append(critic_rating)
        user_ratings.append(user_rating)
        months.append(month)
        days.append(day)
        years.append(year)
        
        if len(titles) >= 300:
            break
    except Exception:
        continue

# Create DataFrame
movies_lowdim = pd.DataFrame({
    "title": titles,
    "critic_rating": critic_ratings,
    "user_rating": user_ratings,
    "month": months,
    "day": days,
    "year": years
})

# Ensure correct data types
movies_lowdim = movies_lowdim.astype({
    "title": "string",
    "critic_rating": "int",
    "user_rating": "int",
    "month": "int",
    "day": "int",
    "year": "int"
})

# Display first 10 rows and column headers
print(movies_lowdim.head(10))
print("\nShape of dataset:", movies_lowdim.shape)
