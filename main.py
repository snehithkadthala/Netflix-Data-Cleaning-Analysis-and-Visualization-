import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px  # Import Plotly for interactive plots
import warnings
import xgboost as xgb


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load dataset and clean
data = pd.read_csv('netflix1.csv')
data.columns = data.columns.str.strip()  # Strip column names
print(data.head())

# Check for missing values
print(data.isnull().sum())
# Remove duplicates and rows with missing values in important columns
data.drop_duplicates(inplace=True)
data.dropna(subset=['director', 'title', 'country', 'type'], inplace=True)

# Convert 'date_added' to datetime format
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')

# Extract year and month from 'date_added'
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month

# Count the number of genres per movie
data['genre_count'] = data['listed_in'].apply(lambda x: len(x.split(', ')))

# Extract duration in minutes
data['duration_minutes'] = data['duration'].str.extract('(\d+)').astype(float)

# Show the new features
print(data[['title', 'listed_in', 'genre_count', 'duration', 'duration_minutes']].head())

# Print the count of unique genres
unique_genres = data['listed_in'].str.split(', ').explode().nunique()
print(f"Count of unique genres: {unique_genres}")

# Print the count of movies for each genre
genre_counts = data['listed_in'].str.split(', ').explode().value_counts()
print("Count of movies for each genre:")
print(genre_counts)

# Print the count of movies with specific durations
duration_counts = data['duration_minutes'].value_counts()
print("Count of movies by duration in minutes:")
print(duration_counts)

# Show data types to confirm changes
print(data.dtypes)

# EDA: Plot content added over the years using Plotly
fig_years = px.histogram(data, x='year_added', title='Content Added Over Time',
                          labels={'year_added': 'Year'},
                          color_discrete_sequence=['#1f77b4'])
fig_years.update_layout(xaxis_title='Year', yaxis_title='Count')
fig_years.show()

# Filter out missing or "Not Given" directors
filtered_data = data[data['director'] != 'Not Given']

# Plot top 10 directors after filtering using Plotly
top_directors = filtered_data['director'].value_counts().head(10)
fig_directors = px.bar(x=top_directors.values, y=top_directors.index,
                        title='Top 10 Directors with the Most Titles',
                        labels={'x': 'Number of Titles', 'y': 'Director'},
                        color=top_directors.values, color_continuous_scale=px.colors.sequential.Blues)
fig_directors.update_layout(yaxis_title='Director', xaxis_title='Number of Titles')
fig_directors.show()

# Generate word cloud for movie titles
movie_titles = data[data['type'] == 'Movie']['title']
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Content-based recommendation system using genres
data['listed_in'] = data['listed_in'].fillna('')
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
genre_matrix = vectorizer.fit_transform(data['listed_in'])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# Director/Creator Influence Trend Prediction
# Feature engineering (one-hot encode 'listed_in' and 'country')
data = pd.get_dummies(data, columns=['country', 'listed_in'], drop_first=True)

# Print out the column names after one-hot encoding
print("Columns after one-hot encoding:")
print(data.columns.tolist())

# Identify the columns related to genres
genre_columns = [col for col in data.columns if col.startswith('listed_in_')]
print("Available genre columns:", genre_columns)

data['duration_minutes'] = pd.to_numeric(data['duration_minutes'], errors='coerce')

# Selecting features for modeling
X = data[['release_year', 'duration_minutes', 'country_United States', 'genre_count'] + genre_columns[:5]]  # Modify to include relevant genres

data['popular'] = np.where(data['rating'] == 'TV-MA', 1, 0)  # Example condition
y = data['popular']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model's performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Model Accuracy:", accuracy_xgb)
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix for XGBoost
plt.figure(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Popular', 'Popular'],
            yticklabels=['Not Popular', 'Popular'])
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importances for XGBoost
feature_importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(feature_importances_xgb)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances for XGBoost')
plt.bar(range(X.shape[1]), feature_importances_xgb[indices_xgb], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices_xgb], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# ROC Curve for XGBoost
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
plt.legend(loc='lower right')
plt.show()