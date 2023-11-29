import feature_eng
import model






file_path = 'data/Telco-Customer-Churn.csv'

# Load and clean data
churn_data = feature_eng.load_and_clean_data(file_path)

# Feature engineering
churn_data_fe = feature_eng.feature_engineering(churn_data)

# Model evaluation
model_performance = model.evaluate_models(churn_data_fe.drop('Churn', axis=1), churn_data_fe['Churn'])
print(model_performance)
"""{'Logistic Regression': {'Accuracy': 0.7924662402274343, 'F1 Score': 0.562874251497006, 'Precision': 0.6394557823129252, 'Recall': 0.5026737967914439},
 'KNN': {'Accuracy': 0.767590618336887, 'F1 Score': 0.49614791987673346, 'Precision': 0.5854545454545454, 'Recall': 0.4304812834224599}, 
 'SVM': {'Accuracy': 0.7341862117981521, 'F1 Score': 0.0, 'Precision': 0.0, 'Recall': 0.0}, 
 'Decision Tree': {'Accuracy': 0.7228144989339019, 'F1 Score': 0.49612403100775193, 'Precision': 0.48, 'Recall': 0.5133689839572193}, 
 'Random Forest': {'Accuracy': 0.7889125799573561, 'F1 Score': 0.547945205479452, 'Precision': 0.6360424028268551, 'Recall': 0.48128342245989303}, 
 'Gradient Boosting': {'Accuracy': 0.7910447761194029, 'F1 Score': 0.5611940298507463, 'Precision': 0.6351351351351351, 'Recall': 0.5026737967914439}, 
 'XGBoost': {'Accuracy': 0.7775408670931059, 'F1 Score': 0.533532041728763, 'Precision': 0.6026936026936027, 'Recall': 0.4786096256684492}, 
 'Catboost': {'Accuracy': 0.7846481876332623, 'F1 Score': 0.5484351713859911, 'Precision': 0.6195286195286195, 'Recall': 0.4919786096256685}}
(PROJELER) ➜  telcho_"""