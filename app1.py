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
knn_model = joblib.load('./models/k-nearest_neighbors_model_best10.pkl')
nb_model = joblib.load('./models/naive_bayes_model_best10.pkl')
dt_model = joblib.load('./models/decision_tree_model_best10.pkl')
ada_model = joblib.load('./models/adaboost_model_best10.pkl')
bagging_model = joblib.load('./models/bagging_model_best10.pkl')
extra_trees_model = joblib.load('./models/extra_trees_model_best10.pkl')
voting_trees = joblib.load('./models/voting_classifier_model_best10.pkl')

logreg_withoutmouseevents = joblib.load('./models_without_mouseevents/logistic_regression_model_best10.pkl')
rf_model_withoutmouseevents = joblib.load('./models_without_mouseevents/random_forest_model_best10.pkl')
svc_model_withoutmouseevents = joblib.load('./models_without_mouseevents/svm_model_best10.pkl')
knn_model_withoutmouseevents = joblib.load('./models_without_mouseevents/k-nearest_neighbors_model_best10.pkl')
nb_model_withoutmouseevents = joblib.load('./models_without_mouseevents/naive_bayes_model_best10.pkl')
dt_model_withoutmouseevents = joblib.load('./models_without_mouseevents/decision_tree_model_best10.pkl')
ada_model_withoutmouseevents = joblib.load('./models_without_mouseevents/adaboost_model_best10.pkl')
bagging_model_withoutmouseevents = joblib.load('./models_without_mouseevents/bagging_model_best10.pkl')
extra_trees_model_withoutmouseevents = joblib.load('./models_without_mouseevents/extra_trees_model_best10.pkl')
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
        if features[0]!=0:

      
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

            # Make predictions with all models
            make_prediction(logreg, "Logistic Regression")
            make_prediction(rf_model, "Random Forest")
            make_prediction(svc_model, "SVM")
            make_prediction(knn_model, "K-Nearest Neighbors")
            make_prediction(nb_model, "Naive Bayes")
            make_prediction(dt_model, "Decision Tree")
            make_prediction(ada_model, "AdaBoost")
            make_prediction(bagging_model, "Bagging")
            make_prediction(extra_trees_model, "Extra Trees")
            make_prediction(voting_trees, "voting Trees")

            # Determine if it's a bot or a human based on the count of 0s
            if zero_count >= 6:  # Majority voting rule: if 6 or more models predict "bot"
                classification = 'bot'
            else:
                classification = 'human'

            # Calculate the average confidence for both bot (0) and human (1)
            avg_conf_bot = total_conf_bot /10
            avg_conf_human = total_conf_human /10

            print(f"\nFinal Classification: {classification}")
            print(f"Average Confidence for bot (0): {avg_conf_bot:.2f}")
            print(f"Average Confidence for human (1): {avg_conf_human:.2f}")

            response = {
                'classification': classification,
                'average_confidence_human': avg_conf_human,
                'average_confidence_bot': avg_conf_bot
            }
            
            return jsonify(response), 200 # Optional: No need to return anything to the client
        else :
            
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
            make_prediction(knn_model_withoutmouseevents, "K-Nearest Neighbors")
            make_prediction(nb_model_withoutmouseevents, "Naive Bayes")
            make_prediction(dt_model_withoutmouseevents, "Decision Tree")
            make_prediction(ada_model_withoutmouseevents, "AdaBoost")
            make_prediction(bagging_model_withoutmouseevents, "Bagging")
            make_prediction(extra_trees_model_withoutmouseevents, "Extra Trees")
            make_prediction(voting_trees_withoutmouseevents, "voting Trees")

            # Determine if it's a bot or a human based on the count of 0s
            if zero_count >= 6:  # Majority voting rule: if 6 or more models predict "bot"
                classification = 'bot'
            else:
                classification = 'human'

            # Calculate the average confidence for both bot (0) and human (1)
            avg_conf_bot = total_conf_bot /10
            avg_conf_human = total_conf_human /10

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

if __name__ == '__main__':
    app.run(debug=True)
