import numpy as np
import graphviz

from data_loaders import get_mnist_dataLoader


class CustomDecisionTree:
    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=None):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth and len(y) >= self.min_samples_split and len(np.unique(y)) > 1:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, depth + 1)
                    if self._can_prune(node):
                        node.left = None
                        node.right = None
        return node

    def _can_prune(self, node):
        if node.left is None or node.right is None:
            return False
        if node.left.predicted_class == node.right.predicted_class:
            return True
        return False

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes))

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    # def export_tree(self, feature_names, class_names, filename='tree'):
    #     dot_data = self._generate_dot(self.tree, feature_names, class_names)
    #     graph = graphviz.Source(dot_data)
    #     graph.render(filename, format='png', cleanup=True)

    def export_graphviz(self, feature_names, class_names):
        dot_data = ["digraph Tree {"]
        dot_data.append('node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
        dot_data.append('edge [fontname="helvetica"] ;')

        def recurse(node, node_id):
            if node.left is None and node.right is None:
                dot_data.append(
                    f'{node_id} [label="gini = {node.gini:.2f}\\nsamples = {node.num_samples}\\n'
                    f'value = {node.num_samples_per_class}\\nclass = {class_names[node.predicted_class]}", fillcolor="#ffffff"] ;')
            else:
                dot_data.append(
                    f'{node_id} [label="{feature_names[node.feature_index]} <= {node.threshold:.5f}\\n'
                    f'gini = {node.gini:.2f}\\nsamples = {node.num_samples}\\n'
                    f'value = {node.num_samples_per_class}\\nclass = {class_names[node.predicted_class]}", fillcolor="#ffffff"] ;')
                left_id = node_id * 2 + 1
                right_id = node_id * 2 + 2
                recurse(node.left, left_id)
                recurse(node.right, right_id)
                dot_data.append(f'{node_id} -> {left_id} ;')
                dot_data.append(f'{node_id} -> {right_id} ;')

        recurse(self.tree, 0)
        dot_data.append("}")
        return "\n".join(dot_data)


# Example usage:
def example_usage():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load the dataset
    # iris = load_iris()
    # X, y = iris.data, iris.target
    #
    # # Split the dataset into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    train_dl, val_dl, test_dl = get_mnist_dataLoader(batch_size=32)
    X_train = train_dl.dataset[:][1].numpy()
    y_train = train_dl.dataset[:][2].numpy()

    # Train the custom decision tree classifier on the training set
    clf = CustomDecisionTree(max_depth=1000)
    clf.fit(X_train, y_train)

    # Predict on the test set
    # y_pred = clf.predict(X_test)
    #
    # # Print the predictions
    # print("Predictions:", y_pred)
    # print("Actual labels:", y_test)

    concept_names = ["thickness_small", "thickness_medium", "thickness_large",
                      "thickness_xlarge", "width_small", "width_medium", "width_large",
                      "width_xlarge", "length_small", "length_medium", "length_large",
                      "length_xlarge"]
    class_names = ["6", "8", "9"]

    # Export the original tree to Graphviz format
    original_dot_data = clf.export_graphviz(
        feature_names=concept_names,
        class_names=class_names
    )
    original_graph = graphviz.Source(original_dot_data)
    original_graph.render("original_tree", format='png', cleanup=True)


    # Export the tree
    # clf.export_tree(feature_names=concept_names, class_names=class_names, filename='custom_tree')

example_usage()
