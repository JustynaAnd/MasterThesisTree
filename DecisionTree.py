#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import tree
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/justy/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


# data loading and cleaning 
data = pd.read_csv('data494.csv')

data_df = pd.DataFrame(data)
data_df.head(5)


# Columns - drop / change names

data_df.columns = [
    'Timestamp', 
    'NEP1', 'NEP2', 'NEP3', 'NEP4', 'NEP5', 'NEP6', 'NEP7', 'NEP8', 'NEP9', 'NEP10', 'NEP11', 'NEP12', 'NEP13', 'NEP14', 'NEP15', 
    'GEK1', 'GEK2', 'GEK3', 'GEK4', 'GEK5', 'GEK6', 'GEK7', 'GEK8', 'GEK9', 'GEK10', 'GEK11', 'GEK12', 'GEK13', 'GEK14', 'GEK15', 
    'SIBS1', 'SIBS2', 'SIBS3', 'SIBS4', 'SIBS5', 'SIBS6', 'SIBS7', 'SIBS8', 'SIBS9', 'SIBS10', 'SIBS11', 'SIBS12', 'SIBS13', 'SIBS14', 'SIBS15', 
    'Sex', 'Age', 'Marital_status', 'Place_of_residence', 'Education_level', 'Current_occupation', 'Monthly_net_income', 
    'Household_size', 'Financial_situation', 'Satisfacion_with_life', 'Diet', 'Social_media', 'Religious_practices', 
    'Next_election', 
    'E-mail']
print(data_df.columns)

# Remove unnecessary columns
data_df = data_df.drop(columns=['Timestamp', 'E-mail'], axis=1)
data_df


print(data_df.info())


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


# Add columns with scores
data_df['NEP_score'] = data_df.loc[:, data_df.filter(regex = "^(NEP)").columns].sum(axis=1)
data_df['GEK_score'] = data_df.loc[:, data_df.filter(regex = "^(GEK)").columns].sum(axis=1)
data_df['SIBS_score'] = data_df.loc[:, data_df.filter(regex = "^(SIBS)").columns].sum(axis=1)

data_df.head()


print(data_df.info())


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
satisf_old_cat = data_df['Satisfacion_with_life'].unique().tolist()
print(satisf_old_cat)
satisf_new_cat = ['hard to say', 'rather satisfied', 'very satisfied', 'rather dissatisfied', 'very dissatisfied']
print(satisf_new_cat)
satisf_dict = dict(zip(satisf_old_cat, satisf_new_cat))
print(satisf_dict)

data_df['Satisfacion_with_life'] = data_df['Satisfacion_with_life'].replace(satisf_dict) 
print(data_df['Satisfacion_with_life'].head())


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
data_df['Satisfacion_with_life'].value_counts().plot.pie(colors=colors_05, autopct='%1.1f%%', shadow=False, startangle=150,  pctdistance = 1.0, labeldistance = 1.3)
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


#Spearman correlation for continuous variables
NG_cor = scipy.stats.spearmanr(data_df['NEP_score'],data_df['GEK_score'])
NS_cor = scipy.stats.spearmanr(data_df['NEP_score'],data_df['SIBS_score'])
SG_cor = scipy.stats.spearmanr(data_df['SIBS_score'],data_df['GEK_score'])
NG_cor, NS_cor, SG_cor


#output variable discretization
print(pd.qcut(data_df['SIBS_score'], q = 4).value_counts())
data_df['SIBS_label'] = pd.qcut(data_df['SIBS_score'], q = 4, labels = ['low', 'medium', 'high', 'very high']).astype(str)
data_df.head()


#input variables encoding

# float variables: NEP_score, GEK_score
# categorical variables: Sex, Age, Marital_status, Place_of_residence, Education_level, Current_occupation, Monthly_net_income, 
# Household_size, Financial_situation, Satisfacion_with_life, Diet, Social_media, Religious_practices, Next_election 


# numeric features - standarization
numeric_features = ['NEP_score', 'GEK_score']
#change integer to float
for f in numeric_features:
    data_df[f] = data_df[f].astype(np.float64)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# categorical features - transformation into binary
categorical_features = ['Sex', 'Age', 'Marital_status', 'Place_of_residence', 'Education_level', 'Current_occupation', 
                     'Monthly_net_income', 'Household_size', 'Financial_situation', 'Satisfacion_with_life', 'Diet', 
                     'Social_media', 'Religious_practices', 'Next_election']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# model pipeline (for given random seed)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', tree.DecisionTreeClassifier(random_state = 0,
                                                                max_depth = 10,
                                                                min_samples_leaf = 10))])

# dataset split

X = data_df[numeric_features+categorical_features]
print(X.head())
print(X.info())
y = data_df['SIBS_label']
print(y.head())


# model for 1k iterations - find the best score
scores = []
score_opt = 0
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X, y, test_size = 0.2)  

for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i)  
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)
    if score > score_opt:
        X_train_opt, X_test_opt, y_train_opt, y_test_opt = X_train, X_test, y_train, y_test
        score_opt = score


# results histogram
plt.hist(scores)
plt.plot()
plt.savefig('models_hist.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)
print(pd.Series(scores).describe())


# best model
print(score_opt)

clf = clf.fit(X_train_opt, y_train_opt)
#clf.score(X_test_opt, y_test_opt)


# tree export and visualisation

# variable labels
encoder = OneHotEncoder()
train_X_encoded = encoder.fit_transform(X_train_opt[categorical_features])
labels_cat = encoder.get_feature_names(X_train_opt[categorical_features].columns)
labels = numeric_features + list(labels_cat)

# plot tree
dot_data = StringIO()
tree.export_graphviz(clf.named_steps['classifier'], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                    class_names = y_train_opt.unique(),
                    feature_names = labels)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())

# interpretation: 
# https://towardsdatascience.com/understanding-decision-trees-for-classification-python-9663d683c952


# optimize the tree depth

max_depth_range = range(1, 11)
depth_accuracy = []

for depth in max_depth_range:
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', tree.DecisionTreeClassifier(random_state = 0,
                                                                max_depth = depth,
                                                                #min_samples_split = 10,
                                                                min_samples_leaf = 10))])
    clf.fit(X_train_opt, y_train_opt)
    score = clf.score(X_test_opt, y_test_opt)
    depth_accuracy.append(score)

depth_results = dict(zip(max_depth_range, depth_accuracy))
print(depth_results)

plt.plot(max_depth_range, depth_accuracy)

ymax = max(depth_accuracy)
xpos = depth_accuracy.index(ymax)
xmax = max_depth_range[xpos]
plt.plot(xmax, ymax, 'o')
plt.savefig('max_depth.png', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)

plt.show()


# optimal tree with depth = 8

clf_best = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', tree.DecisionTreeClassifier(random_state = 0,
                                                                max_depth = 8,
                                                                #min_samples_split = 10,
                                                                min_samples_leaf = 10))])
clf_best.fit(X_train_opt, y_train_opt)

# plot tree
dot_data = StringIO()
tree.export_graphviz(clf_best.named_steps['classifier'], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                    class_names = y_train_opt.unique(),
                    feature_names = labels)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree_best.png')
Image(graph.create_png())


# features importance

#clf.named_steps['classifier'].feature_importances_.tolist()

importance = pd.DataFrame({'feature':labels,
                           'importance':np.round(clf_best.named_steps['classifier'].feature_importances_,3)})
importance = importance.sort_values('importance',ascending=False)
print(importance.loc[importance['importance'] >0, :])

