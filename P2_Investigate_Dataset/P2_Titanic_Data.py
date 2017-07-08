

# import all necessary packages and functions.
import sys
import csv
import unicodecsv
from datetime import datetime
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# import data 
titanic_data = pd.read_csv('titanic_data.csv')
print "Dataset Sample: "
display (titanic_data.head())
display (titanic_data.tail())

# delete unneeded colomns
titanic_data.drop('PassengerId', axis=1, inplace=True)
titanic_data.drop('Name', axis=1, inplace=True)
titanic_data.drop('Ticket', axis=1, inplace=True)
titanic_data.drop('Cabin', axis=1, inplace=True)

# add new colomn 'Family' 
titanic_data['Family'] = (titanic_data['SibSp'] + titanic_data['Parch'])

# replace numeric values for plot readability: 
titanic_data['SurvivedLabel'] = titanic_data['Survived'].copy()
titanic_data['SurvivedLabel'].replace({0:'Died', 1:'Survived'}, inplace=True)
titanic_data['Embarked'].replace({'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)

print "Dataset Sample (after removing variables and adding new variable): "
display (titanic_data.head())
display (titanic_data.tail())

# get dataset overview and statistics 
print "Dataset Info: "
display (titanic_data.info())

print "Dataset Statistics: "
display (titanic_data.describe())



is_null = titanic_data.isnull().sum()
is_null = is_null[is_null > 0]
print "Count of missing data points: "
display (is_null)

print "Percentage of missing data points: "
display (is_null /len(titanic_data)*100)

age_is_null = titanic_data['Age'].notnull()
embarked_is_null = titanic_data['Embarked'].notnull()

print "Sample of missing Age records: "
display (titanic_data[age_is_null == False].head())
display (titanic_data[age_is_null == False].tail())

print "Sample of missing Embarked records: "
display (titanic_data[embarked_is_null == False])


# define generic methods: 

# get the length of dataset: 
titanic_data_count = len(titanic_data)

# displays given data and rename its columns: 
def Print_Info (name, data):
    data.columns = [name, "Count", "Percentage"]
    print (name + " Info:")
    display (data)

# returns distinct values of data column: 
def Get_Colomn_Values (data): 
    return data.unique().sort()

# calcuates the percentage of selected column in dataset: 
def Get_Percentage (column, data):
    data["Percentage"] = data[column]/titanic_data_count * 100 


# Survived data info and plot: 
from numpy import mean
Survived = titanic_data[["Pclass", "Survived"]].groupby(['Survived'],as_index=False).count()
Get_Percentage("Pclass", Survived)
Survived_values =  Get_Colomn_Values(titanic_data.SurvivedLabel)

Print_Info("Survived", Survived)

plot = sns.barplot(x='SurvivedLabel', y = Survived.Count,data=titanic_data, order=Survived_values)
plot.set_title("Survival Info")
plt.ylabel("Count")


plot = sns.barplot(x='SurvivedLabel', y = Survived.Percentage,data=titanic_data, order=Survived_values)
plot.set_title("Survival Info (Percentage)")
plt.ylabel("Percentage %")


# Gender info and plot: 
Sex = titanic_data[["Sex", "Survived"]].groupby(['Sex'],as_index=False).count()
Get_Percentage("Survived", Sex)

Sex_values = Get_Colomn_Values(titanic_data.Sex)
Print_Info("Sex", Sex)

plot = sns.countplot(x='Sex', data=titanic_data, order=Sex_values)
plot.set_title("Gender Info")


# Sex in relation to Survived plot: 
plot = sns.countplot(x='SurvivedLabel', hue='Sex', data=titanic_data)
plot.set_title("Sex and Survival Info ")


# Age info and plot: 
titanic_data_age = titanic_data.copy()
titanic_data_age.Age.replace("NaN",  np.nan, inplace = True)

print "Missing data points:"
display (titanic_data_age.isnull().sum())

titanic_data_age.dropna(inplace = True)

print "Missing data points (after cleaning):"
display (titanic_data_age.isnull().sum())

print "Age Statistics: "
display (titanic_data_age.Age.describe())

plot = sns.distplot(titanic_data_age.Age, bins=30, kde=False, rug=True)
plot.set_title("Age Distibution")
plt.ylabel ("Count")


print "Age Distibution based on Sex"
plot = sns.FacetGrid(titanic_data_age, col="Sex")  
plot.map(sns.distplot, "Age")  
plot.set_ylabels ("Density")


print "Age Distibution based on Survivial"
plot = sns.FacetGrid(titanic_data_age, col="SurvivedLabel")  
plot.map(sns.distplot, "Age")  
plot.set_ylabels ("Density")


Age_Survived = titanic_data_age[["Age", "Survived"]].groupby(['Survived'],as_index=False).mean()
print "Mean Age based on Survival:"
display (Age_Survived)
plot = sns.boxplot(x= "SurvivedLabel", y="Age", data=titanic_data)
plot.set_title("Age Distibution")
plt.xlabel ("Survival")



Age_Survived_Gender = titanic_data_age[["Age", "Survived", "Sex"]].groupby(['Survived', 'Sex'],as_index=False).mean()
print "Mean Age based on Sex and Survival:"
display (Age_Survived_Gender)

plot = sns.boxplot(x="Sex", y="Age", hue="SurvivedLabel", data=titanic_data)
plot.set_title("Age Distibution based on Survived and Gender")


# Pclass info and plot: 
Pclass = titanic_data[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).count()
Get_Percentage("Survived", Pclass)

Pclass_values = Get_Colomn_Values(titanic_data.Pclass)
Print_Info("Pclass", Pclass)

plot = sns.countplot(x='Pclass', data=titanic_data, order=Pclass_values)
plot.set_title("Class Info")


print "Count of Survived based on Pclass: "
Pclass_Survived = titanic_data[["Pclass", "Survived", "Sex"]].groupby(['Pclass','Survived'],as_index=False).count()
Pclass_Survived.columns = ["Pclass", "Survived", "Count"]
display(Pclass_Survived)

plot = sns.countplot(x='SurvivedLabel', hue='Pclass', data=titanic_data)
plot.set_title("Class Info based on Survival")



print "Count of Survived based on Pclass and Sex: "
Pclass_Survived = titanic_data[["Pclass", "Sex", "Survived"]].groupby(['Pclass', 'Sex'],as_index=False).count()
Pclass_Survived_Sex = titanic_data[["Pclass", "Sex", "Survived", 'Age']].groupby(['Pclass', 'Sex', 'Survived'],as_index=False).count()

display(Pclass_Survived)
plot = sns.barplot(x=titanic_data.Sex, y=titanic_data.Survived, hue=titanic_data.Pclass, 
            estimator=mean)
plot.set_title("Class Info based on Gender")
plt.ylabel ("Survival Rate")



print "Class Info and Survived rate based on Gender"
sns.factorplot(x="Sex", y="Survived", col="Pclass", data=titanic_data, kind="bar")


# Family info and plot: 
Family = titanic_data[["Family", "Survived"]].groupby(['Family'],as_index=False).count()
Get_Percentage("Survived", Family)

Family_values = Get_Colomn_Values(titanic_data.Family)
Print_Info("Family", Family)

plot = sns.countplot(x='Family', data=titanic_data, order=Family_values)
plot.set_title("Family Info")


# Family in relation to Survived plot: 
plot = sns.countplot(x='SurvivedLabel', hue='Family', data=titanic_data)
plot.set_title("Family Info based on Survival")


plot = sns.countplot(x='SurvivedLabel', data=titanic_data.where(titanic_data.Family == 0))
plot.set_title("Alone passengers Survival")


plot = sns.countplot(x='SurvivedLabel', hue = 'Family',data=titanic_data.where(titanic_data.Family > 0))
plot.set_title("Family passengers Survival")


print "Largest Family Survived:"
largest_family_survived = titanic_data.where(titanic_data.Survived == 1)
print largest_family_survived.Family.max()


print "Age Distibution based on Family"
plot = sns.FacetGrid(titanic_data_age, col="Family", col_wrap = 4)  
plot.map(sns.distplot, "Age", kde = False)  
plot.set_ylabels ("Count")


# Embarked info and plot: 
Embarked = titanic_data[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).count()
Get_Percentage("Survived", Embarked)

Embarked_values = Get_Colomn_Values(titanic_data.Embarked)
Print_Info("Embarked", Embarked)

plot = sns.countplot(x='Embarked', data=titanic_data, order=Embarked_values)
plot.set_title("Embarked Info")


plot = sns.barplot(x=titanic_data.Survived, y=titanic_data.Pclass, hue=titanic_data.Embarked, ci = None)
plot.set_title("Embarked Survival based on Class")
plt.ylabel ("Survival Rate")


plot = sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=titanic_data)
plot.set_title("Embarked Survival based on Gender")
plt.ylabel ("Survival Rate")



# Fare info and plot: 
print "Fare Statistics: "
display (titanic_data.Fare.describe())

plot = sns.distplot(titanic_data.Fare, bins=30, kde=False, rug=True)
plot.set_title("Fare Distibution")
plt.ylabel ("Count")


print "Fare Distibution based on Class"
plot = sns.FacetGrid(titanic_data, col="Pclass")  
plot.map(sns.distplot, "Fare") 


print "Fare Disribution based on Emabarked"
plot = sns.FacetGrid(titanic_data, col="Embarked")  
plot.map(sns.distplot, "Fare") 


print "Fare Disribution based on Survival"
plot = sns.FacetGrid(titanic_data, col="SurvivedLabel")  
plot.map(sns.distplot, "Fare") 


plot = sns.pointplot(x="Embarked", y="Fare", hue="Sex", data=titanic_data)
plot.set_title("Fare Disribution based on Embarked and Gender")
plt.ylabel ("Fare (Mean)")


plot = sns.pointplot(x="Embarked", y="Fare", hue="Pclass", data=titanic_data)
plot.set_title("Fare Disribution based on Embarked and Class")
plt.ylabel ("Fare (Mean)")


