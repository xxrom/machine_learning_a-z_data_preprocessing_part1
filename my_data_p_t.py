# inporting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importin the database
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values