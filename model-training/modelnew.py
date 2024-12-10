import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Import joblib to save the model

# Load the dataset from a CSV file
df = pd.read_csv('../data/data_without_mouseevents_123.csv')

# Fill NaN values with 0
df = df.fillna(0)

# Use iloc to select features and label
X = df.iloc[:, :-1]  # Select all rows and columns except the last one for features
y = df.iloc[:, -1]   # Select all rows and only the last column for label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary to store models and their names
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ], voting='soft')
}

# Iterate through models, train, save, and evaluate them
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    output_folder = "models_without_mouseevents"
    filename = f'{name.lower().replace(" ", "_")}_model_best10.pkl'
    os.makedirs(output_folder, exist_ok=True)

# Full path to save the file
    file_path = os.path.join(output_folder, filename)
    
    joblib.dump(model, file_path)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "-"*50 + "\n")

