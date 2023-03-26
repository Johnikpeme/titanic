#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[122]:


#load all csv files
df1 =pd.read_csv("C:\\Users\\User\\Downloads\\titanic\\train.csv")
df2 =pd.read_csv("C:\\Users\\User\\Downloads\\titanic\\test.csv")
df3 =pd.read_csv("C:\\Users\\User\\Downloads\\titanic\\gender_submission.csv")


# In[61]:


#print top row of each to confirm
print(df1.head())


# In[62]:


print(df2.head())


# In[63]:


print(df3.head())


# In[64]:


#merge datasets df2 and df3 to give us a full representation of the dataset
merged_df = pd.merge(df2, df3, on="PassengerId")


# In[65]:


print(merged_df.head())


# In[66]:


#concatenate both to get the full titanic dataset
titanic = pd.concat([df1, merged_df])


# In[27]:


print(titanic)


# In[67]:


#number of passengers
print("Total number of passengers=",len(titanic))


# In[68]:


#number of survivors and deaths
print("Total number of survivors=",len(titanic[titanic["Survived"]==1]))
print("Total number of deaths=",len(titanic[titanic["Survived"]==0]))


# In[142]:


no_survivors=len(titanic[titanic["Survived"]==1])
no_deaths=len(titanic[titanic["Survived"]==0])
plt.pie([no_survivors,no_deaths],labels=(['Survived','Deaths']),autopct='%1.1f%%')
plt.title=("Survivors and Deaths")
plt.show()


# In[69]:


#survival rate of passengers
print(("Survival rate="),((len(titanic[titanic["Survived"]==1])/len(titanic))*100),"%")


# In[70]:


#class of passengers distribution plot
sns.countplot(x='Pclass',data=titanic)
plt.title=("Passenger class distribution")
plt.show()


# In[90]:


#average age of passengers
print("Average passenger age=", round(titanic["Age"].mean(),2))


# In[91]:


#survival rate based on gender
print("Survival rate based on gender:")
print(titanic.groupby("Sex")["Survived"].mean())


# In[145]:


Surv_gender= (titanic.groupby("Sex")["Survived"].mean()*100)
plt.pie(Surv_gender, labels=(['Female','Male']),autopct='%1.1f%%')
plt.title="Survival rate based on gender"
plt.show()


# In[95]:


#survival based on passenger class and gender
sns.barplot(x="Pclass",y="Survived",hue="Sex",data=titanic)
plt.title="Survival rate based on class and gender"
plt.show()


# In[98]:


#average fare paid by different classes of passengers
print("Average fare paid by passengers from each class:")
print(titanic.groupby("Pclass")["Fare"].mean())


# In[102]:


#survival rate for each port departure
print("Survival rate for each port departure:")
print(titanic.groupby("Embarked")["Survived"].mean())


# In[104]:


#survival rate based on cabin location
survival_rate= (titanic.groupby("Cabin")["Survived"].mean())


# In[105]:


#exclude NaN Cabins
cabin_to_exclude = "NaN"
if cabin_to_exclude in survival_rate.index:
    survival_rates.drop(cabin_to_exclude, inplace=True)


# In[131]:


#survival rates per cabin

# Drop rows where the Cabin column is missing
titanic_cabin = titanic.dropna(subset=["Cabin"])

# Calculate the survival rate by cabin
survival_rate_by_cabin = titanic.groupby("Cabin").agg({"Survived": "mean", "PassengerId": "count"}).reset_index()
survival_rate_by_cabin = survival_rate_by_cabin.rename(columns={"Survived": "SurvivalRate", "PassengerId": "Count"})

# Sort the data by the survival rate in descending order
survival_rate_by_cabin = survival_rate_by_cabin.sort_values(by="SurvivalRate", ascending=False)

# Plot a bar chart of the data
plt.figure(figsize=(10, 6))
sns.barplot(x="Cabin", y="SurvivalRate", data=survival_rate_by_cabin)
plt.title=("Survival Rates per Cabin")
plt.xlabel("Cabin")
plt.ylabel("Survival Rate")
plt.show()


# In[116]:


#average number of siblings/spouses aboard the titanic & average number of parents and children
avg_SibSp= titanic["SibSp"].mean()
avg_Parch= titanic["Parch"].mean()
print("Average number of siblings/spouses on the titanic=",avg_SibSp)
print("Average number of parents/children on the titanic=",avg_Parch)


# In[123]:


#siblings/spouses & parents/children based on gender and ticket class
print("correlation:")
print(titanic.groupby(["Sex","Pclass"])[["SibSp","Parch"]].mean())


# In[129]:


#survival rate based on age and sex

# Drop rows where the Age column is missing
titanic_age = titanic.dropna(subset=["Age"])
print("Survival rate based on age and sex distribution:")
survival_age= titanic_age.groupby(["Sex", pd.cut(titanic_age.Age, [0, 18, 30, 50, 80])])["Survived"].mean()
print(survival_age)
ax = survival_age.plot(kind="bar", stacked=True)
ax.set_title("Survival Rate by Sex and Age")
ax.set_xlabel("Age Group")
ax.set_ylabel("Survival Rate")
plt.show()


# In[150]:


# Create a pivot table to calculate the survival rate based on the number of siblings/spouses and parents/children
pivot_table = pd.pivot_table(titanic, values='Survived', index=['SibSp', 'Parch'], aggfunc=[np.mean, np.sum])

# Rename the columns
pivot_table.columns = ['Survival Rate', 'Number of Survivors']

# Print the pivot table
print(pivot_table)


# In[151]:


# Calculate the survival rate based on SibSp and Parch
sibsp_parch = titanic.groupby(['SibSp', 'Parch'])['Survived'].mean().reset_index()

# Create a pivot table to reshape the data for heat map plotting
sibsp_parch_pivot = sibsp_parch.pivot(index='SibSp', columns='Parch', values='Survived')

# Plot the heat map
sns.heatmap(sibsp_parch_pivot, cmap='coolwarm', annot=True, fmt='.2f')


# In[153]:


import re

# Extract the first letter of each cabin value to get the deck level
deck = titanic['Cabin'].apply(lambda x: re.findall("([A-Za-z]+)", str(x))[0][0])

# Create a new column for the deck level
titanic['Deck'] = deck


# In[154]:


# Calculate survival rates by deck
deck_survival = titanic.groupby(['Deck'])['Survived'].mean()

print(deck_survival)


# In[160]:


# Plot the survival rates by deck as a bar chart
sns.barplot(x='Deck', y='Survived', data=titanic, ci=None)

