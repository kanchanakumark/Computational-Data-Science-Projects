import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Loading the dataset
orders= pd.read_csv('Instacart//orders.csv')
orderproducts= pd.read_csv('Instacart//order_products__train.csv')
products= pd.read_csv('Instacart//products.csv')
aisles= pd.read_csv('Instacart//aisles.csv')
departments= pd.read_csv('Instacart//departments.csv')

#Displays the column of orders dataset
orders.columns
#Displays the column of orders_products dataset
orderproducts.columns
#Displays the column of products dataset
products.columns
#Displays the column of aisles dataset
aisles.columns
#Displays the column of departments dataset
departments.columns

#Merges Orders dataset with orderproducts using order_id column
data = orders.merge(orderproducts, how='inner', on='order_id')
data.shape
#Merges new dataset dataset with products using product id
data = data.merge(products,how='inner', on='product_id')
data.shape
#Merges new dataset with aisles usimg aisle id
data = data.merge(aisles, how='inner', on='aisle_id')
data.shape
#Merges new dataset with departments using departent id
data = data.merge(departments, how='inner', on='department_id')
#Shape and First Five rows of merged dataframe
data.shape
data.head()

#Null Values
data.isnull().sum()


#grouping the dataset based on product id and  product name and aggregates with count
product_freq=data.groupby(['product_id','product_name']).agg({'product_name':'count'})
#Displays the number of time each product is ordered
product_freq
#Grouping the data by department id and counts the orderid in each department
dept_order=data.groupby('department_id').agg({'order_id':'count'})
dept_order.head(5)

#bar plot
plt.figure(figsize=(18,8))
sns.barplot(dept_order.index,dept_order.order_id);
plt.xlabel("Department ID")
plt.ylabel('Number of Orders')
plt.show();

#Groups the data by order day and counts the order id 
weekday=data.groupby('order_dow').agg({'order_id':'count'})

#bar plot
plt.figure(figsize=(15,8))
sns.barplot(weekday.index,weekday.order_id);
plt.xlabel('Day of Week')
plt.ylabel('Number of Orders')
plt.show()

#Groups the data by order hour and counts the order id 
hour_order=data.groupby('order_hour_of_day').agg({'order_id':'count'})

#bar plot
plt.figure(figsize=(15,8))
sns.barplot(hour_order.index,hour_order.order_id);
plt.xlabel('Hour')
plt.ylabel('Number of Orders')
plt.show()

order_reorder = data.groupby('reordered')['reordered'].aggregate('count')
order_reorder

plt.pie(order_reorder, labels=order_reorder.index,autopct='%1.1f%%', startangle=90);
plt.title('Reordered vs Not-Reordered');

hmdata=heatmapp.pivot('order_dow', 'order_hour_of_day', 'reordered')
plt.figure(figsize=(18,8))
sns.heatmap(hmdata);
plt.title('Order Day vs Order Hour vs Reorder')
plt.xlabel('Hour of the day')
plt.ylabel('Day of the week');

#Creating Basket
order_freq = data.drop(['user_id', 'eval_set', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order', 'reordered','aisle_id', 'department_id', 'aisle', 'department'], axis=1)
mostfreq_products = order_freq.groupby('product_name').agg({'product_id':'count'}).sort_values(by='product_id', ascending=False).head(100).reset_index()
mostfreq_dataset = data[data['product_name'].isin(mostfreq_products['product_name'])]
basket = mostfreq_dataset.pivot_table(index='order_id',columns='product_name',values='reordered', aggfunc=np.sum, fill_value=0)
basket

def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1

basket = basket.applymap(hot_encode)
basket

freq_items = apriori(basket, min_support = 0.1, use_colnames = True) 
freq_items
mba_rules = association_rules(freq_items, metric ="lift", min_threshold = 1.0) 
mba_rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
