#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
apps = pd.read_csv("datasets/apps.csv")


# In[11]:


for char in chars_to_remove:
    apps['Installs'] = apps['Installs'].apply(lambda x: x.replace(char, ''))
apps['Installs'] = apps['Installs'].astype(int)
apps.info()    
apps.head()


# In[8]:


app_category_info = apps.groupby('Category').agg({'App': 'count', 'Price': 'mean', 'Rating': 'mean'})
app_category_info.head()


# In[7]:


app_category_info = app_category_info.rename(columns={"App": "Number of apps", "Price": "Average price", "Rating": "Average rating"})
app_category_info.head()


# In[12]:


reviews = pd.read_csv('datasets/user_reviews.csv')

finance_apps = apps[apps['Category'] == 'FINANCE']

free_finance_apps = finance_apps[finance_apps['Type'] == 'Free']

merged_df = pd.merge(finance_apps, reviews, on = "App", how = "inner")


app_sentiment_score = merged_df.groupby('App').agg({'Sentiment Score' :'mean'})


user_feedback = app_sentiment_score.sort_values(by = 'Sentiment Score', ascending = False)

top_10_user_feedback = user_feedback[:10]
top_10_user_feedback


# In[5]:









