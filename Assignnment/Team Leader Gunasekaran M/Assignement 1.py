import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the provided URL
url = 'https://drive.google.com/uc?id=1OvfEP-D6rFrEKhO8mbpKl_9Fjw0Kv6CX'
data = pd.read_csv(url)

# Perform univariate analysis by creating histograms for each numerical column
numerical_columns = data.select_dtypes(include='number').columns
data[numerical_columns].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Perform bivariate analysis by creating boxplots and scatterplots
sns.boxplot(x='smoker', y='charges', data=data)
plt.show()

sns.scatterplot(x='age', y='charges', hue='smoker', data=data)
plt.show()

# Perform multi-variate analysis by creating a heatmap of the correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Display descriptive statistics using the describe() method
print(data.describe())

# Check for missing values in the dataset using isnull().sum()
print(data.isnull().sum())

# Fill any missing values with the mean of the respective column using fillna()
data.fillna(data.mean(), inplace=True)
