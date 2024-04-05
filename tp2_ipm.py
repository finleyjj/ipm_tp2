import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

file_path =r"C:\Users\Client\OneDrive - HEC Montréal\International Porfolio Management\TP2"

# Import spots and forward data
df_spot = pd.read_csv(rf"{file_path}\spot.csv")
df_forward = pd.read_csv(rf"{file_path}\forward.csv")




