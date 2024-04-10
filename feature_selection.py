class FeatureSelector:
    def __init__(self, df, target_column, threshold=0.5):
        self.df = df
        self.target_column = target_column
        self.threshold = threshold

    def select_features(self):
        # Perform feature selection based on correlation analysis with the target column
        correlations = self.df.corr()[self.target_column].abs()
        relevant_features = correlations[correlations >= self.threshold].index.tolist()
        return self.df[relevant_features]
