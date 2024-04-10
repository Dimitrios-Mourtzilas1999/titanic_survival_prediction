from data_processing import DataPreprocessor
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from visualization import ResultVisualizer

# URL of the Titanic dataset
titanic_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

# Initialize DataPreprocessor
data_processor = DataPreprocessor(titanic_url)

# Load and preprocess data
titanic_df = data_processor.load_data()
processed_df = data_processor.preprocess_data(titanic_df)

# Initialize FeatureSelector
feature_selector = FeatureSelector(processed_df, target_column='Survived', threshold=0.3)

# Perform feature selection
selected_features_df = feature_selector.select_features()

# Initialize ModelTrainer
model_trainer = ModelTrainer(selected_features_df, target_column='Survived')

# Train the model
model, accuracy = model_trainer.train_model()

# Get test data and make predictions
X_test = selected_features_df.drop(columns=['Survived'])
y_test = selected_features_df['Survived']
y_pred = model.predict(X_test)

# Visualize model results using ResultVisualizer
result_visualizer = ResultVisualizer(y_test, y_pred)
result_visualizer.visualize_results()
