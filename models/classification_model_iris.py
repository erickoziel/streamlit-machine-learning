from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# save the model to disk
with open('models/classification_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Test with one new instance
new_instance = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(new_instance)
print(f"Prediction for new instance: {iris.target_names[pred][0]}")
