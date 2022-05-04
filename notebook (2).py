# ## 1. Google Play Store apps and reviews
# <p>Mobile apps are everywhere. They are easy to create and can be lucrative. Because of these two factors, more and more apps are being developed. In this notebook, we will do a comprehensive analysis of the Android app market by comparing over ten thousand apps in Google Play across different categories. We'll look for insights in the data to devise strategies to drive growth and retention.</p>
# <p><img src="https://assets.datacamp.com/production/project_619/img/google_play_store.png" alt="Google Play logo"></p>
# <p>Let's take a look at the data, which consists of two files:</p>
# <ul>
# <li><code>apps.csv</code>: contains all the details of the applications on Google Play. There are 13 features that describe a given app.</li>
# <li><code>user_reviews.csv</code>: contains 100 reviews for each app, <a href="https://www.androidpolice.com/2019/01/21/google-play-stores-redesigned-ratings-and-reviews-section-lets-you-easily-filter-by-star-rating/">most helpful first</a>. The text in each review has been pre-processed and attributed with three new features: Sentiment (Positive, Negative or Neutral), Sentiment Polarity and Sentiment Subjectivity.</li>
# </ul>


import pandas as pd
apps_with_duplicates = pd.read_csv("datasets/apps.csv")

apps = apps_with_duplicates.drop_duplicates()
print('Total number of apps in the dataset = ', len(apps))

n = 5
apps.sample(n)


# In[15]:


get_ipython().run_cell_magic('nose', '', '\ncorrect_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n\ndef test_pandas_loaded():\n    assert (\'pd\' in globals()), "pandas is not imported and aliased as specified in the instructions."\n\ndef test_apps_with_duplicates_loaded():\n#     correct_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n    assert (correct_apps_with_duplicates.equals(apps_with_duplicates)), "The data was not correctly read into apps_with_duplicates."\n    \ndef test_duplicates_dropped():\n#     correct_apps_with_duplicates = pd.read_csv(\'datasets/apps.csv\')\n    correct_apps = correct_apps_with_duplicates.drop_duplicates()\n    assert (correct_apps.equals(apps)), "The duplicates were not correctly dropped from apps_with_duplicates."\n    \ndef test_total_apps():\n    correct_total_apps = len(correct_apps_with_duplicates.drop_duplicates())\n    assert (correct_total_apps == len(apps)), "The total number of apps is incorrect. It should equal 9659."\n    ')


# ## 2. Data cleaning

# In[16]:

chars_to_remove = ['+', ',', '$']
cols_to_clean = ['Installs', 'Price']

for col in cols_to_clean:

 for char in chars_to_remove:

        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))


print(apps.info())


# In[17]:


get_ipython().run_cell_magic('nose', '', 'import numpy as np\n\ndef test_installs_plus():\n    installs = apps[\'Installs\'].values\n    plus_removed_correctly = all(\'+\' not in val for val in installs)\n    assert plus_removed_correctly, \\\n    \'Some of the "+" characters still remain in the Installs column.\' \n    \ndef test_installs_comma():\n    installs = apps[\'Installs\'].values\n    comma_removed_correctly = all(\',\' not in val for val in installs)\n    assert comma_removed_correctly, \\\n    \'Some of the "," characters still remain in the Installs column.\'\n    \ndef test_price_dollar():\n    prices = apps[\'Price\'].values\n    dollar_removed_correctly = all(\'$\' not in val for val in prices)\n    assert dollar_removed_correctly, \\\n    \'Some of the "$" characters still remain in the Price column.\'')


# ## 3. Correcting data types

# In[18]:

import numpy as np

# Convert Installs to float data type
apps['Installs'] = apps['Installs'].astype(float)

# Convert Price to float data type
apps['Price'] = apps['Price'].astype(float)

# Checking dtypes of the apps dataframe
print(apps.dtypes)


# In[19]:


get_ipython().run_cell_magic('nose', '', "import numpy as np\n\ndef test_installs_numeric():\n    assert isinstance(apps['Installs'][0], np.float64), \\\n    'The Installs column is not of numeric data type (float).'\n\ndef test_price_numeric():\n    assert isinstance(apps['Price'][0], np.float64), \\\n    'The Price column is not of numeric data type (float).'")


# ## 4. Exploring app categories


import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)

num_apps_in_category = apps['Category'].value_counts()

sorted_num_apps_in_category = num_apps_in_category.sort_values(ascending = False)

data = [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
)]

plotly.offline.iplot(data)


# In[21]:


get_ipython().run_cell_magic('nose', '', '\ndef test_num_categories():\n    assert num_categories == 33, "The number of app categories is incorrect. It should equal 33."\n    \ndef test_num_apps_in_category():\n    correct_sorted_num_apps_in_category = apps[\'Category\'].value_counts().sort_values(ascending=False)\n    assert (correct_sorted_num_apps_in_category == sorted_num_apps_in_category).all(), "sorted_num_apps_in_category is not what we expected. Please inspect the hint."')


# ## 5. Distribution of app ratings

# In[22]:

avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

data = [go.Histogram(
        x = apps['Rating']
)]

layout = {'shapes': [{
              'type' :'line',
              'x0': avg_app_rating,
              'y0': 0,
              'x1': avg_app_rating,
              'y1': 1000,
              'line': { 'dash': 'dashdot'}
          }]
          }

plotly.offline.iplot({'data': data, 'layout': layout})


# In[23]:


get_ipython().run_cell_magic('nose', '', '\ndef test_app_avg_rating():\n    assert round(avg_app_rating, 5) == 4.17324, \\\n    "The average app rating rounded to five digits should be 4.17324."\n    \n# def test_x_histogram():\n#     correct_x_histogram = apps[\'Rating\']\n#     assert correct_x_histogram.all() == data[0][\'x\'].all(), \\\n#     \'x should equal Rating column\'')


# ## 6. Size and price of an app

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

apps_with_size_and_rating_present = apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())]

large_categories = apps_with_size_and_rating_present.groupby(['Category']).filter(lambda x: len(x) >= 250)

plt1 = sns.jointplot(x = large_categories['Size'], y = large_categories['Rating'])

paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type'] == 'Paid']

plt2 = sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'])


# In[25]:


get_ipython().run_cell_magic('nose', '', '\ncorrect_apps_with_size_and_rating_present = apps[(~apps[\'Rating\'].isnull()) & (~apps[\'Size\'].isnull())]\n \ndef test_apps_with_size_and_rating_present():\n    global correct_apps_with_size_and_rating_present\n    assert correct_apps_with_size_and_rating_present.equals(apps_with_size_and_rating_present)\n    "The correct_apps_with_size_and_rating_present is not what we expected. Please review the instructions and check the hint if necessary."\n    \ndef test_large_categories():\n    global correct_apps_with_size_and_rating_present\n    correct_large_categories = correct_apps_with_size_and_rating_present.groupby([\'Category\']).filter(lambda x: len(x) >= 250)\n    assert correct_large_categories.equals(large_categories), \\\n    "The large_categories DataFrame is not what we expected. Please review the instructions and check the hint if necessary."\n\ndef test_size_vs_rating():\n    global correct_apps_with_size_and_rating_present\n    correct_large_categories = correct_apps_with_size_and_rating_present.groupby(\'Category\').filter(lambda x: len(x) >= 250)\n#     correct_large_categories = correct_large_categories[correct_large_categories[\'Size\'].notnull()]\n#     correct_large_categories = correct_large_categories[correct_large_categories[\'Rating\'].notnull()]\n    assert plt1.x.tolist() == large_categories[\'Size\'].values.tolist() and plt1.y.tolist() == large_categories[\'Rating\'].values.tolist(), \\\n    "The size vs. rating jointplot is not what we expected. Please review the instructions and check the hint if necessary."\n    \ndef test_paid_apps():\n    global correct_apps_with_size_and_rating_present\n    correct_paid_apps = correct_apps_with_size_and_rating_present[correct_apps_with_size_and_rating_present[\'Type\'] == \'Paid\']\n    assert correct_paid_apps.equals(paid_apps), \\\n    "The paid_apps DataFrame is not what we expected. Please review the instructions and check the hint if necessary."\n    \ndef test_price_vs_rating():\n    global correct_apps_with_size_and_rating_present\n    correct_paid_apps = correct_apps_with_size_and_rating_present[correct_apps_with_size_and_rating_present[\'Type\'] == \'Paid\']\n#     correct_paid_apps = correct_paid_apps[correct_paid_apps[\'Price\'].notnull()]\n#     correct_paid_apps = correct_paid_apps[correct_paid_apps[\'Rating\'].notnull()]\n    assert plt2.x.tolist() == correct_paid_apps[\'Price\'].values.tolist() and plt2.y.tolist() == correct_paid_apps[\'Rating\'].values.tolist(), \\\n    "The price vs. rating jointplot is not what we expected. Please review the instructions and check the hint if necessary."\n')


# ## 7. Relation between app category and app price

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

ax = sns.stripplot(x = popular_app_cats['Price'], y = popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')

apps_above_200 = apps_above_200 = popular_app_cats[popular_app_cats['Price'] > 200]
apps_above_200[['Category', 'App', 'Price']]

#  in[25]:


get_ipython().run_cell_magic('nose', '', '\nlast_output = _\n\ndef test_apps_above_200():\n    assert len(apps_above_200) == 17, "There should be 17 apps priced above 200 in apps_above_200."')


# ## 8. Filter out "junk" apps

# In[28]:


apps_under_100 = popular_app_cats[popular_app_cats['Price'] < 100]

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

ax = sns.stripplot(x = 'Price', y = 'Category', data = apps_under_100, jitter = True, linewidth = 1)
ax.set_title('App pricing trend across categories after filtering for junk apps')


# In[29]:


get_ipython().run_cell_magic('nose', '', '\ndef test_apps_under_100():\n    correct_apps_under_100 = popular_app_cats[popular_app_cats[\'Price\'] < 100]\n    assert correct_apps_under_100.equals(apps_under_100), \\\n    "The apps_under_100 DataFrame is not what we expected. Please review the instructions and check the hint if necessary."')


# ## 9. Popularity of paid apps vs free apps

# In[30]:


trace0 = go.Box(

    y = apps[apps['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

trace1 = go.Box(

    y = apps[apps['Type'] == 'Free']['Installs'],
    name = 'Free'
)



data = [trace0, trace1]
plotly.offline.iplot({'data': data, 'layout': layout})


# In[31]:


get_ipython().run_cell_magic('nose', '', '\ndef test_trace0_y():\n    correct_y = apps[\'Installs\'][apps[\'Type\'] == \'Paid\']\n    assert all(trace0[\'y\'] == correct_y.values), \\\n    "The y data for trace0 appears incorrect. Please review the instructions and check the hint if necessary."\n\ndef test_trace1_y():\n    correct_y_1 = apps[\'Installs\'][apps[\'Type\'] == \'Free\']\n    correct_y_2 = apps[\'Installs\'][apps[\'Price\'] == 0]\n    try:\n        check_1 = all(trace1[\'y\'] == correct_y_1.values)\n    except:\n        check_1 = False\n    try:\n        check_2 = all(trace1[\'y\'] == correct_y_2.values)\n    except:\n        check_2 = False\n        \n    assert check_1 or check_2, \\\n    "The y data for trace1 appears incorrect. Please review the instructions and check the hint if necessary."')


# ## 10. Sentiment analysis of user reviews

# In[32]:



reviews_df = pd.read_csv('datasets/user_reviews.csv')

merged_df = pd.merge(apps, reviews_df, on = "App")

merged_df = merged_df.dropna(subset = ['Sentiment', 'Review'])

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

ax = sns.boxplot(x = 'Type', y = 'Sentiment_Polarity', data = merged_df)
ax.set_title('Sentiment Polarity Distribution')


# In[33]:


get_ipython().run_cell_magic('nose', '', '\ndef test_user_reviews_loaded():\n    correct_user_reviews = pd.read_csv(\'datasets/user_reviews.csv\')\n    assert (correct_user_reviews.equals(reviews_df)), "The user_reviews.csv file was not correctly loaded. Please review the instructions and inspect the hint if necessary."\n    \ndef test_user_reviews_merged():\n    user_reviews = pd.read_csv(\'datasets/user_reviews.csv\')\n    correct_merged = pd.merge(apps, user_reviews, on = "App")\n    correct_merged = correct_merged.dropna(subset=[\'Sentiment\', \'Review\'])\n    assert (correct_merged.equals(merged_df)), "The merging of user_reviews and apps is incorrect. Please review the instructions and inspect the hint if necessary."\n    \ndef test_project_reset():\n    user_reviews = pd.read_csv(\'datasets/user_reviews.csv\')\n    assert (\'Translated_Reviews\' not in user_reviews.columns), "There is an update in the project and some column names have been changed. Please choose the \\"Reset Project\\" option to fetch the updated copy of the project."\n    ')

