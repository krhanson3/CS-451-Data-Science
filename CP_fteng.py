#CS 451/551
#Coding Practice Session 3
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#P1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
x, y= housing.data, housing.target