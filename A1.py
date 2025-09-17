#CS 451/551
#Assignment 1
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

# Q1 Matching Terms/Definitions
print('#Q1')

print('(A, d)') 
# A. Quantitaive Data: 
# d. Numerical values, like height and weight; these values can be incorporated directly into algebraic formulas 
#    and displayed in graphs and charts.

print('(B, a)')     
# B. Categorical Data: a. Labels describing the properties of the objects under investigation, 
#                         like gender, hair color, and occupation; these labels can usually be coded numerically

print('(C, c)')     
# C. Structured Data: 
# c. Data with a well-defined, predictable schema 
   
print('(D, f)')     
# D. Unstructured Data: 
# f. Data with no well-defined or predictable schema

print('(E,b)')      
# E. Classification: 
# b. Assigning a label to an item from a discrete set of possibilities

print('(F, e)')     
# F. Regression: 
# e. Forecasting a numerical quantity

#Q2 Data Science Problems: (A) Classification vs. (B) Regression
print('#Q2')

print('(B, a)')
# a. What will the price of an S&P 500 index fund tomorrow?

print('(A, b)')
# b. Is this email spam or not?

print('(B, c)')
# c. What was the average weather in Montgomery County, VA on July 4th, 2024?

print('(A, d)')
# d. Based on a given scan, is the tumor malignant or benign?

print('(A, e)')
# e. Based on online reviews, is the difficulty level of the Diamond Head hiking trail in 
#    Oahu, Hawaii low, medium or high?

print('(B, f)')
# f. What is a person’s blood pressure and heart rate during a neighborhood run, based on 
#    their resting blood pressure and heart rate?

#Q3 (A) Classification vs. (B) Regression
# if A, identify the number of classes
print('#Q3')

print('B, a, NA')
# a. How many Google reviews will a new restaurant receive in the month of October 2025?

print('B, b, NA')
# b. How many Yelp reviews did a new restaurant receive in the month of February 2025?

print('A, c, N=2')
# c. Based on the text of a restaurant review, did it come from Yelp or Google?

print('B, d, NA')
# d. What will be a restaurant’s star rating (a whole number between and including 1 and 5) next month? 

print('A, e, N=3')
# e. Is the sentiment of a given restaurant review positive, negative or neutral?

print('A, f, N=2')
# f. Based on all of a restaurant’s reviews and other information available online 
#    (menu, timings, type of cuisine), would a friend of yours like or dislike this restaurant?

#Q4 Matching Terms with examples: (A) Centrality Measures vs. (B)Variability Measures
print('#Q4')

print('B, a')
# a. Variance 

print('A, b')
# b. Geometric Mean 

print('A, c')
# c. Median 

print('B, d')
# d. Standard Deviation

print('A, e')
# e. Mean 

print('A, f')
# f. Mode

#Q5 Conditional Probability 
print('#Q5')
# P(e1) = 0.80  *chicken sandwhich
# P(e2) = 0.89  *waffle fries
# P(e1 and e2) = 0.78   *sandwhich and fries

#What is P(e2 given e1)? *given someone likes the sandwhich, 
#                         whats the probability they like the fries?
# P(e2 given e1) = P(e1 and e2)/P(e1)
print('numerator = P(sandwhich AND fries) = 0.78')
print('denominator = P(sandwhich) = 0.80')
probability = 0.78 / 0.80
print(f'final probability = P(fries given sandwhich) = {probability:.3f} or {probability * 100}%')

#Q6 True/False: 
print('#Q6')

print('a, true')
# a. If two random variables are statistically independent, their correlation coefficient is zero

print('b, false')
# b. Correlation coefficient ranges from -10 to +10

print('c, false')
# c. Correlation captures the complex non-linear relationships between two variables

print('d, true')
# d. A skewed distribution looks asymmetrical

print('e, false')
# e. Two variables are correlated when the value of one has no predictive power on the other

print('f, false')
# f. The correlation between household income and prevalence of heart disease is r = - 0.7,
#        so the likelihood of heart disease increases when income increases.

print('g, false')
# g. The correlation between X and Y is r = 0.1, X and Y have a strong linear relationship. 

#Q7 True/False 
print('#Q7')

print('(a, false)')
# a. Correlation implies causation

print('(b, false)')
# b. The amount of medicine people take is correlated with the likelihood they are sick, so this tells us the medicine they take is responsible for their sickness

print('(c, true)')
# c. The square of the correlation coefficient estimates the fraction of variance in Y explained by X in a simple linear regression

print('(d, false)')
# d. A correlation of 0.5 tells us that 50% of the variance in Y is explained by X

print('(e, true)')
# e. The statistical significance of a correlation depends on the coefficient r as well as the sample size n

print('(f, true)')
# f. Logarithms undo exponentiation in the way that division undoes multiplication

print('(g, true)')
# g. Transforming with logarithms can bring a frequency distribution closer to a symmetric bell shape

#Q8 Pearson Correlation Coefficient
print('#Q8')
import numpy as np
from scipy.stats import pearsonr 

X = [1, 2, 3, 4] 
Y = [3, 1, 2, 4] 
corr, p_value = pearsonr(X, Y)
print(f'Pearson correlation coefficient: {corr:.3f}')
print(f'P-value: {p_value:.3f}')

#Q9 Dataframes
print('#Q9')
import pandas as pd
dataset = {
    'Movies': 
    [   '50 First Dates', 
        'How to train your dragon', 
        'how to lose a guy in ten days', 
        'deadpool', 
        'iron man', 
        'Legally Blonde', 
        'The Big Lebowski', 
        'Rocket Man', 
        'Kings Man Secret Service', 
        'Zootopia'
    ],
    'Likes' : 
    [   'romantic, love the actors!', 
        'fantastic story, epic', 
        'good plot, fun ending', 
        'action packed, funny', 
        'great actors, start of a great series', 
        'an amazing story of overcoming stereotypes',
        'well paced interesting story.',
        'love a good story about elton john',
        'so funny and exciting!!!',
        'good story for young kids and adults alike'
    ], 
    'Dislikes' : 
    [   'Too Repetitive, story felt stuck in a loop',
        'Animation needed more Work',
        'predictable with Overdone cliches',
        'tries TOO hard to be edgy, humor exhausting',
        'just a billionaire in a robot suit',
        'too bubbly and shallow',
        'dragged on and ON AND ON, didnt see the hype',
        'too many musical numbers, little story',
        'cool action but way too ridiculous',
        'kids movie trying too hard to teach lessons'
    ],
    'Ratings' :
    [   '3.4',
        '4.05',
        '3.25',
        '3.75',
        '3.95',
        '3.15',
        '4.05',
        '3.15',
        '3.85',
        '4.0'
    ]
}

df = pd.DataFrame(dataset)
#print(df.columns.tolist())
#print(df)

#Q10
print('#Q10')
import math
df2 = pd.DataFrame(dataset)
df2["Movies"]= df2['Movies'].str.replace(" ", "")
df2["Likes"]= df2['Likes'].str.replace(" ", "")
df2["Dislikes"] = df2['Dislikes'].str.lower()
df2['Ratings'] = df2['Ratings'].astype(float).apply(math.floor).astype(str)

print(df2)

#Q11
print('#Q11')
import matplotlib.pyplot as plt
df['Ratings'] = df['Ratings'].astype(float)

plt.hist(df['Ratings'], )
