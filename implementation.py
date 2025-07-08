import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """ Train the logistic regression model """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                loss = self._loss(y, y_predicted)
                print(f"Iteration {i}: Loss = {loss:.4f}")
        
    def predict(self, X):
        """ Predict labels (0 or 1) """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

    def _loss(self, y, y_hat):
        """ Binary cross-entropy loss """
        epsilon = 1e-15  # for numerical stability
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


## The above code is for implementation of the Logistic Regression Model.

#---------------------------> Solving a Dataset using the Above model <------------------------------

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Binary Classification Dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Training/Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Importing the above created
model = LogisticRegressionScratch(learning_rate=0.01,iterations = 1000)
model.fit(X_train,y_train)

# Predictions on the Model
predictions = model.predict(X_test)

# Evaluate
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")