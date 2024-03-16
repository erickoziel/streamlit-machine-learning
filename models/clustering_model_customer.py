import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Assuming the dataset is in a CSV file named 'mall_customers.csv'
url = 'https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv'
df = pd.read_csv(url)

# Preprocessing
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Encoding Gender
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]  # Selecting relevant features

# Initialize and train the model
model = KMeans(n_clusters=5, random_state=42)  # Choosing 5 as an example number of clusters
model.fit(X)

# save the model to disk
with open('models/clustering_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Test with one new instance
new_instance = [[1, 35, 90, 10]]  # A new customer
pred = model.predict(new_instance)
print(f"Prediction for new instance: {pred[0]}")  # The cluster number
