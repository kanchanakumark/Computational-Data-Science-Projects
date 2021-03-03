#importing necessary libraries
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

#Downloading Dataset
playstore = pd.read_csv("https://cdn.iisc.talentsprint.com/CDS/Datasets/googleplaystore.csv")\

playstore.head()
playstore.info()
playstore.isnull().sum()
playstore.isnull().sum()
playstore.shape

#Creating a copy
playstore_fill=playstore

#Replacing NAN with Mean
playstore_fill['Rating'].fillna((playstore['Rating'].mean()), inplace = True)
playstore_fill.isnull().sum()

#Droping remaining Null values
playstore_fill.dropna(inplace=True)
playstore_fill.shape

playstoredata=playstore_fill.sort_values(by='Last Updated', ascending= True)

#Removing duplicate rows
playstoredata.drop_duplicates(keep = 'last', inplace=True)
playstoredata = playstoredata.reset_index()

# Identifying English Apps
engapplist=[]
for j in range(0,len(playstoredata)):
  for i in playstoredata['App'][j]:
    c=i
    if(ord(c) > 31 & ord(c) < 128):
      eng = True
      continue
    else:
      eng=False
      break
  if(eng==True):
    engapplist.append(playstoredata.iloc[j])

engapp = pd.DataFrame(engapplist)
engapp.shape
playstoredata["Size"].value_counts()


#Replacing the the Size data with 0
playstoredata.loc[(playstoredata.Size == 'Varies with device'), ['Size']] =0
playstoredata['Size'].value_counts()

#In the size column, multiply 10,000,000 with entries having M and multiply by 10,000 if we have K in the cell.
for i in range (0, len(playstoredata.Size)):
  if(playstoredata['Size'][i]!=0):
    if(playstoredata['Size'][i][-1]=='M'):
      playstoredata.loc[i,'Size'] = float((playstoredata['Size'][i][:-1]))*1000000
    else:
      playstoredata.loc[i,'Size']  = float((playstoredata['Size'][i][:-1]))*1000

mean_size=int(playstoredata['Size'].mean())

#Replacing the Size value 0 with the mean
for i in range (0,len(playstoredata)):
  if playstoredata["Size"][i] == 0:
    playstoredata.loc[i,'Size']=mean_size

playstoredata.Size.value_counts()
playstoredata.drop(['index'], axis=1, inplace=True)

#number of apps in various categories by using an appropriate plot.

fig, ax = plt.subplots(figsize=(20,5))
catcountplt = sns.countplot(x="Category", data = playstoredata)
catcountplt.set_xticklabels(catcountplt.get_xticklabels(), rotation=90);

#distribution of free and paid apps across different categories

uniquecateg = playstoredata['Category'].unique()
uniquecateg = list(uniquecateg)
paid=[]
free=[]
for i in uniquecateg:
  f=0
  p=0
  for j in range(0,len(playstoredata)):
    if playstoredata['Category'][j] == i:
      if playstoredata['Type'][j] =='Free':
        f+=1
      if playstoredata['Type'][j]=="Paid":
        p+=1
  paid.append(p)
  free.append(f)
uniquecat=pd.DataFrame(paid,columns=['Paid'])
uniquecat['Free']=free
uniquecat['Category']=uniquecateg
uniquecat.set_index('Category')
uniquecat.plot(kind='bar', stacked=True, figsize=(23,8));
plt.xticks(np.arange(0,33), uniquecateg);

#distribution of app rating on a scale of 1-5 using an appropriate plot
fig, ax = plt.subplots(1,figsize=(15,7))
plt.hist(x='Rating', bins =np.arange(1,5,0.2),data= playstoredata, histtype="bar");

fig, ax = plt.subplots(1,figsize=(15,9))
sns.histplot(data=playstoredata, x='Rating', kde=True, bins=np.arange(1,5,0.1), hue='Type');

#outliers of the rating column by plotting the boxplot category wise and handle them.

from scipy import stats
for i in range(0,len(uniquecateg)):
  temp=[]
  for j in range(1,len(playstoredata)):
    if playstoredata['Category'][j] == uniquecateg[i]:
      temp.append(playstoredata.loc[j,'Rating'])
  rdf = pd.DataFrame(temp)
  fig,ax=plt.subplots(1, figsize=(19,10))
  plot = plt.boxplot(x=rdf[0])
  rdf['zs'] = stats.zscore(rdf)
  rdf = rdf.loc[rdf['zs'].abs()<=3]
  fig,ax=plt.subplots(1, figsize=(19,10))
  plot = plt.boxplot(x=rdf[0])

#barplot of all the categories indicating no. of installs
#Removes +
for i in range(0,len(playstoredata)):
  playstoredata.loc[i,'Installs']= playstoredata['Installs'][i][:-1]
#Removes ,
for i in range(0,len(playstoredata)):
  playstoredata.loc[i,'Installs']=playstoredata['Installs'][i].replace(",","")

playstoredata['Installs']=playstoredata['Installs'].astype(int)

fig, ax = plt.subplots(1,figsize=(20,8))
insplot = sns.barplot(x='Category', y='Installs', data=playstoredata);
insplot.set_xticklabels(insplot.get_xticklabels(),rotation=90);

#the price correlate with the size of the app

fig,ax=plt.subplots(1, figsize=(18,8))
scat = sns.scatterplot(x='Price', y='Size', data = playstoredata);
scat.set_xticklabels(scat.get_xticklabels(),rotation=30);

# popular app categories based on rating and no. of installs

#Groups category and applies mean function on Rating
popapps=playstoredata.groupby('Category').agg({'Rating': 'mean'})
popapps.sort_values(by='Rating', ascending=False)

#apps are produced in each year category-wise ?
#Extracts year from Last Updated
for i in range(0, len(playstoredata)):
  playstoredata.loc[i, 'Year'] = playstoredata['Last Updated'][i][-4:]

playstoredata.head(10)

#Gives the Year ans its corresponding number of app updated in that year
yrapp=playstoredata.groupby('Year').agg({'App': 'count'})
yrapp.sort_values(by='App', ascending = False)

#Gives avg rating Category and Year wise
appyr_rat= playstoredata.groupby(['Category', 'Year']).agg({ 'Rating': 'mean'})
appyr_rat

appyr_rat.reset_index(inplace=True)

#Plots the trend of avg rating category-wise across years (Need to work for good clarity)
fig,ax=plt.subplots(1, figsize=(18,15))
sns.lineplot(x='Year', y='Rating', data=appyr_rat, hue='Category', legend = False );

playstoredata.head()

#highest paid apps with a good rating

for i in range(0,len(playstoredata)):
  if (playstoredata['Type'][i]=='Paid'):
    playstoredata.loc[i,'Price']= playstoredata['Price'][i][1:]

playstoredata['Price']=playstoredata['Price'].astype(float)
playstoredata['Installs']=playstoredata['Installs'].astype(int)

playstoredata['Price'].dtype
playstoredata['Installs'].dtype

#Gives total price for paid apps with rating more than 4.5
for i in range(0,len(playstoredata)):
  if((playstoredata['Rating'][i]>4.5) & (playstoredata['Price'][i]!=0) ):
    playstoredata.loc[i,'TotalPrice'] = (playstoredata['Price'][i])*(playstoredata['Installs'][i])
  else:
    playstoredata.loc[i,'TotalPrice'] = 0

#Top 5 paid apps
highestpaidapp = playstoredata.sort_values(by='TotalPrice', ascending=False)
highestpaidapp.App.head(5)

#checking te top-rated apps
playstoredata.sort_values(by='Rating', ascending=False)

playstoredata['Reviews']=playstoredata['Reviews'].astype(int)

playstoredata['Reviews'].describe()

# Considering only the apps which was reviewed more than 400000 
playstoredata[playstoredata['Reviews']>400000].sort_values(by="Rating", ascending=False)

#number of reviews of an app is very low
playstoredata[(playstoredata['Reviews']<400000) &(playstoredata['Reviews']>0)].sort_values(by="Rating", ascending=False).head(15)

