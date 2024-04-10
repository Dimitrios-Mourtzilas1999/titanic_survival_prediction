import seaborn as sns
import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def visualize_results(self):
        # Create a scatter plot using Seaborn
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.y_test, y=self.y_pred, color='blue', alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--', linewidth=2)
        plt.title('Predicted vs. Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()
