import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Initialize the neural network, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = len(set(y))
model = SimpleNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate probability predictions
model.eval()
with torch.no_grad():
    train_probs = torch.softmax(model(X_train_tensor), dim=1).numpy()
    test_probs = torch.softmax(model(X_test_tensor), dim=1).numpy()


from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

def generate_labels(tree, feature_names):
    labels = []
    for i in range(tree.node_count):
        if tree.children_left[i] == tree.children_right[i]:  # Leaf node
            values = tree.value[i][0]
            argmax = np.argmax(values)
            labels.append(f"class = {argmax}\nsamples = {tree.n_node_samples[i]}\nvalues = {values}")
        else:  # Decision node
            feature = feature_names[tree.feature[i]]
            threshold = tree.threshold[i]
            labels.append(f"{feature} <= {threshold:.2f}\nsamples = {tree.n_node_samples[i]}")
    return labels

# Train a multivariate regression tree on the neural network probabilities
tree_regressor = DecisionTreeRegressor(random_state=42, max_depth=5)  # Limiting depth for better visualization
tree_regressor.fit(X_train, train_probs)

# Predict on the test set
test_probs_tree = tree_regressor.predict(X_test)

# Evaluate the regression tree
mse = mean_squared_error(test_probs, test_probs_tree)
print("Mean Squared Error between NN and Tree probabilities:", mse)

# Convert probabilities to class predictions for evaluation
test_preds_tree = np.argmax(test_probs_tree, axis=1)
test_preds_nn = np.argmax(test_probs, axis=1)
print("Decision Tree accuracy mimicking NN:", accuracy_score(y_test, test_preds_tree))
print("Neural Network accuracy:", accuracy_score(y_test, test_preds_nn))

# Generate custom labels for the tree nodes
labels = generate_labels(tree_regressor.tree_, [f'Feature {i}' for i in range(X.shape[1])])

# Export the decision tree to a DOT format with custom labels
dot_data = export_graphviz(tree_regressor, out_file=None,
                           feature_names=[f'Feature {i}' for i in range(X.shape[1])],
                           filled=True, rounded=True, special_characters=True,
                           node_ids=True, impurity=False, proportion=True)

# Modify the DOT data to include custom labels
dot_data_lines = dot_data.split("\n")
for i in range(len(dot_data_lines)):
    if "label=" in dot_data_lines[i]:
        node_id = int(dot_data_lines[i].split(" ")[0])
        label_index = dot_data_lines[i].find("label=")
        label_end_index = dot_data_lines[i].find("]", label_index)
        # Ensure proper formatting with quotes around the label
        dot_data_lines[i] = dot_data_lines[i][:label_index] + f'label="{labels[node_id]}"' + dot_data_lines[i][label_end_index:]

dot_data = "\n".join(dot_data_lines)

# Use graphviz to render the tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree_custom")  # Save the visualization to a file
graph.view()  # Display the visualization