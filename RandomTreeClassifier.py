#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV 

import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/justy/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


# In[2]:


# data loading and cleaning 
data = pd.read_csv('data494.csv')

data_df = pd.DataFrame(data)
data_df.head(5)


# In[3]:


# Columns - drop / change names

data_df.columns = [
    'Timestamp', 
    'NEP1', 'NEP2', 'NEP3', 'NEP4', 'NEP5', 'NEP6', 'NEP7', 'NEP8', 'NEP9', 'NEP10', 'NEP11', 'NEP12', 'NEP13', 'NEP14', 'NEP15', 
    'GEK1', 'GEK2', 'GEK3', 'GEK4', 'GEK5', 'GEK6', 'GEK7', 'GEK8', 'GEK9', 'GEK10', 'GEK11', 'GEK12', 'GEK13', 'GEK14', 'GEK15', 
    'SIBS1', 'SIBS2', 'SIBS3', 'SIBS4', 'SIBS5', 'SIBS6', 'SIBS7', 'SIBS8', 'SIBS9', 'SIBS10', 'SIBS11', 'SIBS12', 'SIBS13', 'SIBS14', 'SIBS15', 
    'Sex', 'Age', 'Marital_status', 'Place_of_residence', 'Education_level', 'Current_occupation', 'Monthly_net_income', 
    'Household_size', 'Financial_situation', 'Satisfaction_with_life', 'Diet', 'Social_media', 'Religious_practices', 
    'Next_election', 
    'E-mail']
print(data_df.columns)

# Remove unnecessary columns
data_df = data_df.drop(labels = ['Timestamp', 'E-mail'], axis=1)
data_df


# In[4]:


print(data_df.info())


# In[5]:


# NEP - convert answers to points
print(data_df.filter(regex = "^NEP").head())

# Get old labels
nep_old_cat = list(set(sum(data_df.filter(regex = "^NEP").values.tolist(), [])))
nep_old_cat.sort()
print(nep_old_cat)

# zdecydowanie zgadzam się = 2|0
# raczej zgadzam się = 1.5|0.5
# nie mam zdania = 1.0|1.0
# raczej nie zgadzam się = 0.5|1.5
# zdecydowanie nie zgadzam się = 0|2

# Define new labels (mind the order)
nep_new_cat_pos = [1.0, 1.0, 0.5, 1.5, 0.0, 2.0]
nep_new_cat_neg = [1.0, 1.0, 1.5, 0.5, 2.0, 0.0]

nep_dict_pos = dict(zip(nep_old_cat, nep_new_cat_pos))
nep_dict_neg = dict(zip(nep_old_cat, nep_new_cat_neg))
print(nep_dict_pos)
print(nep_dict_neg)

# change labels
for i, col in enumerate(data_df.filter(regex = "^NEP").columns):
    if i % 2 == 0:
        # positive questions
        data_df[col] = data_df[col].replace(nep_dict_pos)
    else:
        # negative questions
        data_df[col] = data_df[col].replace(nep_dict_neg)
    
print(data_df.filter(regex = "^NEP").head())


# In[6]:


# GEK - match positive answers

# 1c 2e 3a 4c 5d 6c 7b 8d 9d 10d 11b 12e 13d 14e 15d

print(data_df.filter(regex = "^GEK").head())

# Dictionary with positive answers
positive_answers = ['zwiększona emisja gazów cieplarnianych, tzw. efekt cieplarniany',
                   'energia atomowa',
                   'całkowita emisja wszystkich gazów cieplarnianych spowodowana podczas pełnego cyklu życia produktu',
                   '10 miliardów',
                   '130',
                   'korzystanie z samochodów z katalizatorem',
                    'przejście na dietę wegetariańską',
                    'przełączanie sprzętu elektronicznego w tryb gotowości, gdy nie jest używany',
                    'kości',
                    'oczyszczenie z tych substancji w oczyszczalni jest trudne',
                    'autokar',
                    'wołowina',
                    '15 razy więcej',
                    'nigdy',
                    '200 litrów']

positive_answers = dict(zip([positive_answers.index(x) for x in positive_answers], positive_answers))
print(positive_answers)

# Set positive answers to 1, 0 otherwise
for i, col in enumerate(data_df.filter(regex = "^GEK").columns):
    data_df[col] = np.where(data_df[col] == positive_answers.get(i), 2, 0)

print(data_df.filter(regex = "^GEK").head())


# In[7]:


# SIB - convert answers to points
print(data_df.filter(regex = "^SIBS").head())

# Get old labels
sib_old_cat = list(set(sum(data_df.filter(regex = "^SIBS").values.tolist(), [])))
sib_old_cat.sort()
print(sib_old_cat)

# Define new labels
sib_new_cat_pos = [0, 0.5, 1, 1.5, 2]
sib_new_cat_neg = sib_new_cat_pos[::-1]

sib_dict_pos = dict(zip(sib_old_cat, sib_new_cat_pos))
sib_dict_neg = dict(zip(sib_old_cat, sib_new_cat_neg))
print(sib_dict_pos)
print(sib_dict_neg)

# change labels
for i, col in enumerate(data_df.filter(regex = "^SIBS").columns):
    if i in (0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14):
        # positive questions
        data_df[col] = data_df[col].replace(sib_dict_pos)
    else:
        # negative questions
        data_df[col] = data_df[col].replace(sib_dict_neg)
    
print(data_df.filter(regex = "^SIBS").head())


# In[8]:


# Add columns with scores
data_df['NEP_score'] = data_df.loc[:, data_df.filter(regex = "^(NEP)").columns].sum(axis=1)
data_df['GEK_score'] = data_df.loc[:, data_df.filter(regex = "^(GEK)").columns].sum(axis=1)
data_df['SIBS_score'] = data_df.loc[:, data_df.filter(regex = "^(SIBS)").columns].sum(axis=1)

data_df.head()


# In[9]:


print(data_df.info())


# In[10]:


### Answers' translations


#sex
sex_old_cat = data_df['Sex'].unique().tolist()
print(sex_old_cat)
sex_new_cat = ['male', 'female']
print(sex_new_cat)
sex_dict = dict(zip(sex_old_cat, sex_new_cat))
print(sex_dict)

data_df['Sex'] = data_df['Sex'].replace(sex_dict)   
print(data_df['Sex'].head())


#age
age_old_cat = data_df['Age'].unique().tolist()
print(age_old_cat)
age_new_cat = ['20-29 years old', '30-39 years old', 'under 20 years old', '50-59 years old', '40-49 years old', 
               '60 years old or above']
print(age_new_cat)
age_dict = dict(zip(age_old_cat, age_new_cat))
print(age_dict)

data_df['Age'] = data_df['Age'].replace(age_dict) 
print(data_df['Age'].head())


#marital status
marital_old_cat = data_df['Marital_status'].unique().tolist()
print(marital_old_cat)
marital_new_cat = ['single', 'informal relationship', 'married', 'widow(er)/divorced', 'widow(er)/divorced']
print(marital_new_cat)
marital_dict = dict(zip(marital_old_cat, marital_new_cat))
print(marital_dict)

data_df['Marital_status'] = data_df['Marital_status'].replace(marital_dict) 
print(data_df['Marital_status'].head())


#place of residence
place_old_cat = data_df['Place_of_residence'].unique().tolist()
print(place_old_cat)
place_new_cat = ['over 500 000 residents', 'village', 'below 50 000 residents', 'from 150 to 500 000 residents', 
                 'from 50 to 150 000 residents']
print(place_new_cat)
place_dict = dict(zip(place_old_cat, place_new_cat))
print(place_dict)

data_df['Place_of_residence'] = data_df['Place_of_residence'].replace(place_dict)
print(data_df['Place_of_residence'].head())


#education
edu_old_cat = data_df['Education_level'].unique().tolist()
print(edu_old_cat)
edu_new_cat = ['higher', 'secondary', 'vocational', 'primary']
print(edu_new_cat)
edu_dict = dict(zip(edu_old_cat, edu_new_cat))
print(edu_dict)

data_df['Education_level'] = data_df['Education_level'].replace(edu_dict)  
print(data_df['Education_level'].head())


#occupation
occup_old_cat = data_df['Current_occupation'].unique().tolist()
print(occup_old_cat)
occup_new_cat = ['white-collar', 'student', 'mid-level manager', 'other', 'blue-collar', 'service and trade', 
                 'other', 'other', 'other', 'other']
print(occup_new_cat)
occup_dict = dict(zip(occup_old_cat, occup_new_cat))
print(occup_dict)

data_df['Current_occupation'] = data_df['Current_occupation'].replace(occup_dict)  
print(data_df['Current_occupation'].head())


#income
income_old_cat = data_df['Monthly_net_income'].unique().tolist()
print(income_old_cat)
income_new_cat = ['4001-5000 PLN', '1000 PLN or below', 'above 5000 PLN', '3001-4000 PLN', '1001-2000 PLN', '2001-3000 PLN']
print(income_new_cat)
income_dict = dict(zip(income_old_cat, income_new_cat))
print(income_dict)

data_df['Monthly_net_income'] = data_df['Monthly_net_income'].replace(income_dict)
print(data_df['Monthly_net_income'].head())


#household size
household_old_cat = data_df['Household_size'].unique().tolist()
print(household_old_cat)
household_new_cat = ['1 (only me)', '3', '2', '5 or more', '4']
print(household_new_cat)
household_dict = dict(zip(household_old_cat, household_new_cat))
print(household_dict)

data_df['Household_size'] = data_df['Household_size'].replace(household_dict)
print(data_df['Household_size'].head())


#financial situation
fin_old_cat = data_df['Financial_situation'].unique().tolist()
print(fin_old_cat)
fin_new_cat = ['average', 'good', 'very good', 'poor/very poor', 'poor/very poor']
print(fin_new_cat)
fin_dict = dict(zip(fin_old_cat, fin_new_cat))
print(fin_dict)

data_df['Financial_situation'] = data_df['Financial_situation'].replace(fin_dict)
print(data_df['Financial_situation'].head())


#life satisfaction
satisf_old_cat = data_df['Satisfaction_with_life'].unique().tolist()
print(satisf_old_cat)
satisf_new_cat = ['hard to say', 'rather satisfied', 'very satisfied', 'rather dissatisfied', 'very dissatisfied']
print(satisf_new_cat)
satisf_dict = dict(zip(satisf_old_cat, satisf_new_cat))
print(satisf_dict)

data_df['Satisfaction_with_life'] = data_df['Satisfaction_with_life'].replace(satisf_dict) 
print(data_df['Satisfaction_with_life'].head())


#diet
diet_old_cat = data_df['Diet'].unique().tolist()
print(diet_old_cat)
diet_new_cat = ['omnivore diet', 'semivegetarianism', 'vegetarianism', 'others', 'veganism', 'others']
print(diet_new_cat)
diet_dict = dict(zip(diet_old_cat, diet_new_cat))
print(diet_dict)

data_df['Diet'] = data_df['Diet'].replace(diet_dict) 
print(data_df['Diet'].head())


#social media
social_old_cat = data_df['Social_media'].unique().tolist()
print(social_old_cat)
social_new_cat = ['few times a day', '1-2 times a day', 'few times a week', 'once a week/two weeks', 'never', 
                  'once a week/two weeks']
print(social_new_cat)
social_dict = dict(zip(social_old_cat, social_new_cat))
print(social_dict)

data_df['Social_media'] = data_df['Social_media'].replace(social_dict)
print(data_df['Social_media'].head())


#religious_practices
religion_old_cat = data_df['Religious_practices'].unique().tolist()
print(religion_old_cat)
religion_new_cat = ['not practice, not believer', 'not practice, believer', 'practice irregularly', 'practice rarely', 
                    'practice regularly']
print(religion_new_cat)
religion_dict = dict(zip(religion_old_cat, religion_new_cat))
print(religion_dict)

data_df['Religious_practices'] = data_df['Religious_practices'].replace(religion_dict)
print(data_df['Religious_practices'].head())


#next election
election_old_cat = data_df['Next_election'].unique().tolist()
print(election_old_cat)
election_new_cat = ['Koalicja Obywatelska', 'not going to vote/not know yet', 'PSL, Kukiz’15', 'Konfederacja', 
                    'Lewica', 'Bezpartyjni Samorządowcy', 'Prawo i Sprawiedliwość']
print(election_new_cat)
election_dict = dict(zip(election_old_cat, election_new_cat))
print(election_dict)

data_df['Next_election'] = data_df['Next_election'].replace(election_dict)
print(data_df['Next_election'].head())


# In[11]:


# Pie charts 

colors_02 = ['yellowgreen', 'gold']
colors_04 = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
colors_05 = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'silver']
colors_06 = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'silver', 'thistle']
colors_07 = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'silver', 'thistle', 'navajowhite']

#sex
data_df['Sex'].value_counts().plot.pie(colors=colors_02, wedgeprops = None, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('sex.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#age
data_df['Age'].value_counts().plot.pie(colors=colors_06, autopct='%1.1f%%', shadow=False, startangle=170,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('age.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#marital status
data_df['Marital_status'].value_counts().plot.pie(colors=colors_04, autopct='%1.1f%%', shadow=False, startangle=0,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('marital.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#place of residence
data_df['Place_of_residence'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=90,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('place.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#education
data_df['Education_level'].value_counts().plot.pie(colors=colors_04, autopct='%1.1f%%', shadow=False, startangle=170,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('edu.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#occupation
data_df['Current_occupation'].value_counts().plot.pie(colors=colors_06, autopct='%1.1f%%', shadow=False, startangle=0,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('occup.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#income
data_df['Monthly_net_income'].value_counts().plot.pie(colors=colors_06, autopct='%1.1f%%', shadow=False, startangle=90,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('income.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#household size
data_df['Household_size'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=90,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('house.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#financial situation
data_df['Financial_situation'].value_counts().plot.pie(colors=colors_04, autopct='%1.1f%%', shadow=False, startangle=0,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('finance.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#life satisfaction
data_df['Satisfaction_with_life'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=150,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('satisf.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#diet
data_df['Diet'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=170,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('diet.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#social media
data_df['Social_media'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=0,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('social.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#religious_practices
data_df['Religious_practices'].value_counts().plot.pie(colors=colors_06, autopct='%1.1f%%', shadow=False, startangle=90,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('religion.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

#next election
data_df['Next_election'].value_counts().plot.pie(colors=colors_07, autopct='%1.1f%%', shadow=False, startangle=150,  pctdistance = 1.0, labeldistance = 1.3)
plt.axis('equal')
plt.axes().set_ylabel('')
plt.tight_layout()
plt.savefig('election.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1,
           bbox_inches = 'tight')
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


###### scores analysis

#NEP_score
print(data_df['NEP_score'].describe())

data_df['NEP_score'].plot(kind = 'hist')
axes = plt.gca()
axes.set_xlim([0, 30])
plt.xticks(np.arange(0, 30+1, 5))
plt.savefig('NEP_score.png')
plt.show()

#GEK_score
print(data_df['GEK_score'].describe())

data_df['GEK_score'].plot(kind = 'hist')
axes = plt.gca()
axes.set_xlim([0, 30])
plt.xticks(np.arange(0, 30+1, 5))
plt.savefig('GEK_score.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)
plt.show()

#SIBS_score
print(data_df['SIBS_score'].describe())

data_df['SIBS_score'].plot(kind = 'hist')
axes = plt.gca()
axes.set_xlim([0, 30])
plt.xticks(np.arange(0, 30+1, 5))
plt.savefig('SIBS_score.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)
plt.show()


# In[13]:


#Spearman correlation for continuous variables
NG_cor = scipy.stats.spearmanr(data_df['NEP_score'],data_df['GEK_score'])
NS_cor = scipy.stats.spearmanr(data_df['NEP_score'],data_df['SIBS_score'])
SG_cor = scipy.stats.spearmanr(data_df['SIBS_score'],data_df['GEK_score'])
NG_cor, NS_cor, SG_cor


# In[14]:


#output variable discretization - three categories, whole dataset

#print(pd.qcut(data_df['SIBS_score'], q = 3).value_counts())
#data_df['SIBS_label'] = pd.qcut(data_df['SIBS_score'], q = 3, labels = ['low', 'medium', 'high']).astype(str)
#data_df.head()


# In[15]:


#output variable discretization - two categories, without middle part of dataset

#print(pd.qcut(data_df['SIBS_score'], q = 10).value_counts())
#(17.0-18.5] - if 20% deleted
#(15.5-19.5] - if 40% deleted


data_df = data_df.loc[(data_df['SIBS_score'] <= 17.0) | (data_df['SIBS_score'] > 18.5), :]
print(pd.qcut(data_df['SIBS_score'], q = 2).value_counts())
data_df['SIBS_label'] = pd.qcut(data_df['SIBS_score'], q = 2, labels = ['low', 'high']).astype(str)
data_df.head()


# In[16]:


#input variables encoding

# float variables: NEP_score, GEK_score
# categorical variables: Sex, Age, Marital_status, Place_of_residence, Education_level, Current_occupation, Monthly_net_income, 
# Household_size, Financial_situation, Satisfacion_with_life, Diet, Social_media, Religious_practices, Next_election 


# numeric features - standarization
numeric_features = ['NEP_score', 'GEK_score']
#change integer to float
for f in numeric_features:
    data_df[f] = data_df[f].astype(np.float64)

scaler = StandardScaler()
ordinal = OrdinalEncoder()
onehot = OneHotEncoder()

################zamienić tę część jakoś tak, żeby nie trzeba było tworzyć Pipeline z modelem (tj. Pipeline może co najwyżej przerobić
###############zmienne) albo żeby w pętlach się zamieniały czy coś

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])


# categorical features - transformation into binary
categorical_features = ['Sex', 'Marital_status', 'Current_occupation', 'Diet', 'Religious_practices', 'Next_election']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# ordinal features - transformation  an integer array
ordinal_features = ['Age', 'Place_of_residence', 'Education_level', 'Monthly_net_income', 'Household_size', 
                       'Financial_situation', 'Satisfaction_with_life',  'Social_media']

ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories = [
        ['under 20 years old', '20-29 years old', '30-39 years old', '40-49 years old', 
                                          '50-59 years old', '60 years old or above'],
        ['village', 'below 50 000 residents', 'from 50 to 150 000 residents', 
                                          'from 150 to 500 000 residents', 'over 500 000 residents'],
        ["primary", "vocational", "secondary", "higher"],
        ['1000 PLN or below', '1001-2000 PLN', '2001-3000 PLN', '3001-4000 PLN', '4001-5000 PLN', 
                                          'above 5000 PLN'],
        ['1 (only me)', '2', '3', '4', '5 or more'],
        ['poor/very poor', 'average', 'good', 'very good'],
        ['very dissatisfied', 'rather dissatisfied', 'hard to say', 
                                          'rather satisfied', 'very satisfied'],
        ['never', 'once a week/two weeks', 'few times a week', 
                                          '1-2 times a day', 'few times a day']
    ]))])


               
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)])


# In[29]:


# model pipeline (for given random seed)

rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('randfor', RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=10, random_state = 21))])

#print(rf.named_steps['preprocessor'].get_params()['ord'].named_steps['ordinal'].categories)


# In[30]:


# dataset split

X = data_df[numeric_features+categorical_features+ordinal_features]
print(X.head())
print(X.info())
y = data_df['SIBS_label']
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  
X_train.head()


# In[31]:


# model fitting

rf = rf.fit(X_train,y_train)


# In[32]:


# model evaluation

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[33]:


# variable labels

encoder = OneHotEncoder()
train_X_encoded = encoder.fit_transform(X_train[categorical_features])
labels_cat = encoder.get_feature_names(X_train[categorical_features].columns)

labels = numeric_features + list(labels_cat) + ordinal_features 


# In[41]:


# features importance

importance = pd.DataFrame({'feature':labels,
                           'importance':np.round(rf.named_steps['randfor'].feature_importances_,3)})
importance = importance.sort_values('importance',ascending=True)
importance = importance.loc[importance['importance'] >0, :]
print(importance)

ax = importance.plot.barh(x = 'feature', color='lightgreen', fontsize=7)
plt.savefig('importance_rf.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1, bbox_inches = 'tight')
plt.show


# In[35]:


# inspect rf's parameters

rf.named_steps['randfor'].get_params()


# In[36]:


# define a grid of hyperparameter and instantiate 'grid_rf'

#params_rf = {
#    'n_estimators': [300, 400, 500],
#    'max_depth': [4, 6, 8],
#    'min_samples_leaf': [10, 20, 30]
#}

#grid_rf = GridSearchCV(estimator = rf,
#                      param_grid = params_rf,
#                      cv = 3, 
#                      scoring = 'accuracy',
#                      verbose = 1, 
#                      n_jobs = -1)


# In[37]:


#grid fitting

#rf_detector = grid_rf.fit(X_train, y_train)
#print(grid_rf.grid_scores_)


# In[ ]:


# extract best hyperparameters and model

#best_hyperparams = grid_rf.best_params_
#print(best_hyperparams)

#best_model = grid_rf.best_estimator_


# In[ ]:


# best model evaluation

#y_pred_grid = best_model.predict(X_test)
#accuracy_grid = accuracy_score(y_test, y_pred)
#print(accuracy_grid)

