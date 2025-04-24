import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ LOAD AND CLEAN DATA ------------------ #
df = pd.read_csv("D:/MLdataset/zomato/zomato_dataset_dirty_modified.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Clean 'Average Delivery Time'
df = df[df['Average Delivery Time'].str.contains(r'^\d+\s*min$', na=False)]
df['Average Delivery Time'] = df['Average Delivery Time'].str.extract(r'(\d+)').astype(float)
df = df[(df['Average Delivery Time'] >= 15) & (df['Average Delivery Time'] <= 90)]

# Clean 'Average Price'
wrong_formats = ["??", "N/A", "free", "none", "abc", "$$$", "---", "error"]
df = df[~df['Average Price'].isin(wrong_formats)]

def clean_price(val):
    if isinstance(val, str):
        digits = re.sub(r"[^\d]", "", val)
        return float(digits) if digits else np.nan
    return float(val)

df['Average Price'] = df['Average Price'].apply(clean_price)
df.dropna(subset=['Average Price'], inplace=True)

# Remove outliers in price
Q1 = df['Average Price'].quantile(0.25)
Q3 = df['Average Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Average Price'] < (Q1 - 1.5 * IQR)) | (df['Average Price'] > (Q3 + 1.5 * IQR)))]

# Clean 'Rating'
df = df[~df['Rating'].isin(wrong_formats)]
df = df[df['Rating'].str.replace('.', '', 1).str.isnumeric()]
df['Rating'] = df['Rating'].astype(float)
df = df[(df['Rating'] >= 2) & (df['Rating'] <= 5)]

# ------------------ FEATURE ENGINEERING ------------------ #
df['Price_per_Minute'] = df['Average Price'] / df['Average Delivery Time']
df['Value_Score'] = df['Rating'] * (1 / df['Price_per_Minute'])
df['Premium_Flag'] = np.where(df['Average Price'] > df['Average Price'].median(), 1, 0)
df['Delivery_Speed'] = np.where(df['Average Delivery Time'] < df['Average Delivery Time'].median(), 'Fast', 'Slow')
df['Price_Range'] = pd.cut(df['Average Price'], bins=3, labels=['Low', 'Medium', 'High'])

# Reduce high-cardinality categories
top_restaurants = df['Restaurant Name'].value_counts().nlargest(20).index
top_cuisines = df['Cuisine'].value_counts().nlargest(20).index
top_locations = df['Location'].value_counts().nlargest(20).index

df['Restaurant Name'] = np.where(df['Restaurant Name'].isin(top_restaurants), df['Restaurant Name'], 'Other')
df['Cuisine'] = np.where(df['Cuisine'].isin(top_cuisines), df['Cuisine'], 'Other')
df['Location'] = np.where(df['Location'].isin(top_locations), df['Location'], 'Other')

# ------------------ MODELING ------------------ #
categorical_cols = ['Restaurant Name', 'Cuisine', 'Safety Measure', 'Location', 'Delivery_Speed', 'Price_Range']
numeric_cols = ['Average Price', 'Average Delivery Time', 'Price_per_Minute', 'Value_Score', 'Premium_Flag']

X = df[categorical_cols + numeric_cols]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Use lightweight single model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n📊 Lightweight Optimized Model:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"✅ Accuracy: {r2 * 100:.2f}%")

# ------------------ VISUALIZATIONS (with sampling) ------------------ #
sample_df = df.sample(frac=0.5, random_state=1)

# 1. Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(sample_df['Rating'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 2. Price vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Price', y='Rating', data=sample_df, alpha=0.6, color='green')
plt.title('Average Price vs Rating')
plt.xlabel('Average Price')
plt.ylabel('Rating')
plt.show()

# 3. Delivery Time vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Delivery Time', y='Rating', data=sample_df, alpha=0.6, color='orange')
plt.title('Delivery Time vs Rating')
plt.xlabel('Delivery Time (minutes)')
plt.ylabel('Rating')
plt.show()

# 4. Cuisine Rating Boxplot
top_cuisines = sample_df['Cuisine'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cuisine', y='Rating', data=sample_df[sample_df['Cuisine'].isin(top_cuisines)], palette='viridis')
plt.title('Rating Distribution by Cuisine (Top 10)')
plt.xticks(rotation=45)
plt.show()

# 5. Safety Measure vs Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='Safety Measure', y='Rating', data=sample_df, palette='pastel')
plt.title('Rating Distribution by Safety Measures')
plt.xticks(rotation=45)
plt.show()

# 6. Location vs Average Price
plt.figure(figsize=(12, 6))
sns.barplot(x='Location', y='Average Price', data=sample_df, ci=None, palette='coolwarm',
           order=sample_df.groupby('Location')['Average Price'].mean().sort_values(ascending=False).index)
plt.title('Average Price by Location')
plt.xticks(rotation=45)
plt.show()
