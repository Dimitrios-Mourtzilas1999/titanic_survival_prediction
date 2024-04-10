import pandas as pd

class DataPreprocessor:
    def __init__(self, url):
        self.url = url

    def load_data(self):
        # Load Titanic dataset from URL into a DataFrame
        titanic_df = pd.read_csv(self.url)
        return titanic_df

    def preprocess_data(self, df):
        
        # Identify unnecessary columns to remove (e.g., 'Name', 'Ticket', etc.)
        columns_to_drop = ['Name']  # Define columns to be removed

        # Remove the identified columns from the DataFrame
        df = df.drop(columns=columns_to_drop, axis=1)
        # Encode categorical features ('Sex', 'Embarked')
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
        df['Age'].fillna(df['Age'].median(), inplace=True)

        return df
