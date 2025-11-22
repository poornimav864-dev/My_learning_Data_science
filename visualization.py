import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# Plot the bar chart
survival_counts = df['Survived'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(['Did Not Survive', 'Survived'], survival_counts, color=['red', 'green'])
plt.title('Titanic Survival Count')
plt.xlabel('Outcome')
plt.ylabel('Number of Passengers')
plt.show()

# Drop missing ages
df = df.dropna(subset=['Age'])

# Plot age distribution by survival using Histplot
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='Set1', alpha=0.5)
plt.title("Titanic Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

#Box plot for fare distribution by pclass
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Pclass', legend=False, palette='Set3')
plt.title("Fare Distribution by Passenger Class")
plt.show()

# Keep only numeric columns
numeric_df = df[['Survived','Age','Fare','SibSp','Parch']].dropna()
# Pairplot
sns.pairplot(numeric_df, hue='Survived', diag_kind='kde', palette='Set1')
plt.show()

#Scatter Plot
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', alpha=0.6)
plt.title("Scatterplot with alpha=0.6")
plt.show()

# correlation matrix
corr = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()

# Pie chart for passenger class distribution
pclass_counts = df['Pclass'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(pclass_counts, labels=['Class 1','Class 2','Class 3'], autopct='%1.1f%%', startangle=90,
        colors=['lightblue','orange','pink'])
plt.title("Passenger Class Distribution")
plt.show()