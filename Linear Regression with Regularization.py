# Loading the Required Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Read the hour.csv file
df_hour = pd.read_csv("/content/hour.csv")
df_hour.head(5)
df_hour.dtypes


fig = plt.figure(figsize=(10,5))
plt.bar(df_hour['hr'],df_hour['cnt'])
plt.xlabel("Hour")
plt.ylabel("Number of bikes rented")
plt.title("Bikes Rental Distribution");

fig = plt.figure(figsize=(15,6))
sns.histplot(data=df_hour, x='cnt');
plt.xlabel('Bikes Rented')
plt.ylabel('Frequency');

fig = plt.figure(figsize=(15,6))
sns.histplot(data=df_hour, x='casual');
plt.xlabel('casual')
plt.ylabel('Frequency');

fig = plt.figure(figsize=(15,6))
sns.histplot(data=df_hour, x='registered');
plt.xlabel('Bikes Registered')
plt.ylabel('Frequency');
