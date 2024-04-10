from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.model = None
        self.accuracy = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_model(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        return self.model, self.accuracy
