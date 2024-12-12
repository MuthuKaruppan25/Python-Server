import warnings
from pymongo import MongoClient
warnings.filterwarnings("ignore")
from bson.objectid import ObjectId
import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import math
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

# Load the models from the files
logreg = joblib.load('./models/logistic_regression_model_best10.pkl')
rf_model = joblib.load('./models/random_forest_model_best10.pkl')
svc_model = joblib.load('./models/svm_model_best10.pkl')
bagging_model = joblib.load('./models/bagging_model_best10.pkl')
voting_trees = joblib.load('./models/voting_classifier_model_best10.pkl')

logreg_withoutmouseevents = joblib.load('./models_without_mouseevents/logistic_regression_model_best10.pkl')
rf_model_withoutmouseevents = joblib.load('./models_without_mouseevents/random_forest_model_best10.pkl')
svc_model_withoutmouseevents = joblib.load('./models_without_mouseevents/svm_model_best10.pkl')
bagging_model_withoutmouseevents = joblib.load('./models_without_mouseevents/bagging_model_best10.pkl')
voting_trees_withoutmouseevents = joblib.load('./models_without_mouseevents/voting_classifier_model_best10.pkl')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        if request.method == 'OPTIONS':
            print("Options request received")
            return '', 200

        # Get the JSON data from the request
        data = request.get_json()

        # Extract features from the JSON data
        features = data.get('features')
        print("Received Features:", features)

        # Replace NaN values with 0
        for i in range(len(features)):
            if features[i] is None:
                features[i] = 0
            if math.isnan(features[i]):
                features[i] = 0
        if features[0] != 0:

            new_data = np.array([features])

            # List to store predictions and confidence scores
            zero_count = 0
            total_conf_bot = 0  # To store the cumulative confidence for bot (0)
            total_conf_human = 0  # To store the cumulative confidence for human (1)
            model_count = 0  # To count the number of models used

            # Function to make predictions and print results
            def make_prediction(model, model_name):
                nonlocal zero_count, total_conf_bot, total_conf_human, model_count
                pred = model.predict(new_data)[0]
                conf = model.predict_proba(new_data)[0]  # Probability for bot (0) and human (1)
                print(f"{model_name} Prediction: {int(pred)}")
                print(f"{model_name} Confidence: bot (0): {conf[0]}, human (1): {conf[1]}")

                # Accumulate the confidence scores for averaging later
                total_conf_bot += conf[0]
                total_conf_human += conf[1]
                model_count += 1

                if pred == 0:
                    zero_count += 1

            make_prediction(logreg, "Logistic Regression")
            make_prediction(rf_model, "Random Forest")
            make_prediction(svc_model, "SVM")
            make_prediction(bagging_model, "Bagging")
            make_prediction(voting_trees, "voting Trees")

            if zero_count >= 3:
                classification = 'bot'
            else:
                classification = 'human'

            avg_conf_bot = total_conf_bot / 5
            avg_conf_human = total_conf_human / 5

            print(f"\nFinal Classification: {classification}")
            print(f"Average Confidence for bot (0): {avg_conf_bot:.2f}")
            print(f"Average Confidence for human (1): {avg_conf_human:.2f}")

            response = {
                'classification': classification,
                'average_confidence_human': avg_conf_human,
                'average_confidence_bot': avg_conf_bot
            }

            return jsonify(response), 200  # Optional: No need to return anything to the client
        else:

            new_data = np.array([features])
            zero_count = 0
            total_conf_bot = 0  # To store the cumulative confidence for bot (0)
            total_conf_human = 0  # To store the cumulative confidence for human (1)
            model_count = 0  # To count the number of models used

            # Function to make predictions and print results
            def make_prediction(model, model_name):
                nonlocal zero_count, total_conf_bot, total_conf_human, model_count
                pred = model.predict(new_data)[0]
                conf = model.predict_proba(new_data)[0]  # Probability for bot (0) and human (1)
                print(f"{model_name} Prediction: {int(pred)}")
                print(f"{model_name} Confidence: bot (0): {conf[0]}, human (1): {conf[1]}")

                # Accumulate the confidence scores for averaging later
                total_conf_bot += conf[0]
                total_conf_human += conf[1]
                model_count += 1

                if pred == 0:
                    zero_count += 1

            # Make predictions with all models
            make_prediction(logreg_withoutmouseevents, "Logistic Regression")
            make_prediction(rf_model_withoutmouseevents, "Random Forest")
            make_prediction(svc_model_withoutmouseevents, "SVM")
            make_prediction(bagging_model_withoutmouseevents, "Bagging")
            make_prediction(voting_trees_withoutmouseevents, "voting Trees")

            # Determine if it's a bot or a human based on the count of 0s
            if zero_count >= 3:  # Majority voting rule: if 6 or more models predict "bot"
                classification = 'bot'
            else:
                classification = 'human'

            # Calculate the average confidence for both bot (0) and human (1)
            avg_conf_bot = total_conf_bot / 5
            avg_conf_human = total_conf_human / 5

            print(f"\nFinal Classification: {classification}")
            print(f"Average Confidence for bot (0): {avg_conf_bot:.2f}")
            print(f"Average Confidence for human (1): {avg_conf_human:.2f}")

            response = {
                'classification': classification,
                'average_confidence_human': avg_conf_human,
                'average_confidence_bot': avg_conf_bot
            }
            print("jiujiii")
            return jsonify(response), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return '', 500
# MongoDB Atlas setup
MONGO_URI = "mongodb+srv://Gowtham:gowtham@bill.fx5nzqb.mongodb.net/?retryWrites=true&w=majority&appName=bill"
mongo_client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)



@app.route('/train', methods=['POST'])
def train_models():
    try:
        # MongoDB Data Fetching and Merging

        if request.method == 'OPTIONS':
            print("Options request received")
            return '', 200

            # Get the JSON data from the request
        req_data = request.get_json()

        # Extract features from the JSON data
        features = req_data.get('label')

        db = mongo_client["test"]

        if(features=="bot"):
            collection = db["reinforcementbotdatas"]
        else:
            collection = db["reinforcementhumandatas"]


        data = list(collection.find())

        if not data:
            print("No documents found in the collection.")
            exit(1)

        print("First document:", data[0])

        # Convert to DataFrame
        mongo_data = pd.DataFrame(data)

        # Drop MongoDB's `_id` field if present
        if '_id' in mongo_data.columns:
            mongo_data = mongo_data.drop('_id', axis=1)
            mongo_data = mongo_data.drop('__v', axis=1)


        # Group data based on `mouseSpeed`
        mongo_data['mouseSpeed'] = pd.to_numeric(mongo_data['mouseSpeed'], errors='coerce')
        group_mouseevent_0 = mongo_data[mongo_data['mouseSpeed'] == 0]
        group_other = mongo_data[mongo_data['mouseSpeed'] != 0]




        # Paths for datasets
        csv_paths = {
            "with_mouseevents": "./data/data_with_mouseevents_new.csv",
            "without_mouseevents": "./data/data_without_mouseevents_123.csv"
        }

        results = {}

        for key, csv_path in csv_paths.items():
            # Load the respective CSV file
            try:
                existing_data = pd.read_csv(csv_path)
            except FileNotFoundError:
                print(f"File not found: {csv_path}. Proceeding with MongoDB data only.")
                existing_data = pd.DataFrame()

            # Merge MongoDB data
            if key == "with_mouseevents":
                if not group_other.empty:
                    merged_data = pd.concat([existing_data, group_other], ignore_index=True)
                    print("saved the data")
            else:
                if not group_mouseevent_0.empty:
                    merged_data = pd.concat([existing_data, group_mouseevent_0], ignore_index=True)
                    print("saved the data")

            # Save the merged data back to the CSV file
            merged_data.to_csv(csv_path, index=False)

            # Fill NaN values with 0
            merged_data = merged_data.fillna(0)

            # Separate features and labels
            X = merged_data.iloc[:, :-1]  # All columns except the last one
            y = merged_data.iloc[:, -1]  # The last column

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Define models
            models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ], voting='soft')
}

            # Folder for saving models
            output_folder = f"modelss_{key}"
            os.makedirs(output_folder, exist_ok=True)

            results[key] = {}

            # Train, save, and evaluate each model
            for name, model in models.items():
                model.fit(X_train, y_train)

                # Save the model
                filename = f'{name.lower().replace(" ", "_")}_model.pkl'
                file_path = os.path.join(output_folder, filename)
                joblib.dump(model, file_path)

                # Evaluate the model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Save results for this model
                results[key][name] = {
                    "accuracy": accuracy,
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
                }

        collection.delete_many({})
        if(features=="bot"):
            document_id = ObjectId("6759dc21c0d66e07dba4315d")
            collection = db["botdatacounts"]
            # Update the count field to 0
            collection.update_one({"_id": document_id}, {"$set": {"count": 0}})

        else:
            document_id = ObjectId("6759c9a7c0d66e07dba4314f")
            collection = db["humandatacounts"]
            # Update the count field to 0
            collection.update_one({"_id": document_id}, {"$set": {"count": 0}})



        return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)