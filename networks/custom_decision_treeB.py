import numpy as np
import graphviz

def tree_to_dict(tree):
    tree_ = tree.tree_
    feature_name = [f"Feature {i}" for i in range(tree_.n_features)]

    def recurse(node):
        if tree_.feature[node] != -2:
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            feature_index = tree_.feature[node]
            threshold = tree_.threshold[node]
            return {
                (feature_index, threshold): {
                    'left': recurse(left_child),
                    'right': recurse(right_child)
                }
            }
        else:
            return None

    return recurse(0)

# Define the TreeNode class
class TreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class, value):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.value = value  # New attribute to store class counts
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

def gini(y):
    m = len(y)
    return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

# Define the CustomDecisionTree class
class CustomDecisionTree:
    def __init__(self, fixed_splits, max_depth=None, min_samples_leaf=1):
        self.fixed_splits = fixed_splits
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.node_count = 0  # Node counter
        self.feature_importances_ = None  # Feature importances

    def fit(self, X, y):
        self.node_count = 0  # Reset node counter before fitting
        self.feature_importances_ = np.zeros(X.shape[1])  # Initialize feature importances
        self.tree = self._grow_tree(X, y, 0, self.fixed_splits)
        self._compute_feature_importances()  # Compute feature importances

    def _grow_tree(self, X, y, depth, fixed_splits):
        classes, num_samples_per_class = np.unique(y, return_counts=True)
        predicted_class = classes[np.argmax(num_samples_per_class)]
        value = np.array([num_samples_per_class[classes == c][0] if c in classes else 0 for c in np.unique(y)])
        node = TreeNode(
            gini=gini(y),
            num_samples=len(y),
            num_samples_per_class=dict(zip(classes, num_samples_per_class)),
            predicted_class=predicted_class,
            value=value
        )
        self.node_count += 1  # Increment node counter

        if fixed_splits is not None and isinstance(fixed_splits, dict):
            (feature_index, threshold), children = list(fixed_splits.items())[0]
            node.feature_index = feature_index
            node.threshold = threshold
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)

            if isinstance(children, dict):
                left_children = children.get('left', None)
                right_children = children.get('right', None)
            else:
                left_children = None
                right_children = None
        else:
            X_left, X_right, y_left, y_right, feature_index, threshold = self._best_split(X, y)
            if feature_index is not None and threshold is not None:
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    return node
                node.feature_index = feature_index
                node.threshold = threshold
            else:
                return node
            left_children = None
            right_children = None

        if depth < self.max_depth and X_left.shape[0] > 0 and X_right.shape[0] > 0:
            node.left = self._grow_tree(X_left, y_left, depth + 1, left_children)
            node.right = self._grow_tree(X_right, y_right, depth + 1, right_children)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None, None, None, None, None

        classes = np.unique(y)
        num_parent = np.array([np.sum(y == c) for c in classes])
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, sorted_y = zip(*sorted(zip(X[:, idx], y)))
            num_left = np.zeros_like(num_parent)
            num_right = num_parent.copy()
            for i in range(1, m):
                c = sorted_y[i - 1]
                num_left[classes == c] += 1
                num_right[classes == c] -= 1
                gini_left = 1.0 - sum((num_left / i) ** 2)
                gini_right = 1.0 - sum((num_right / (m - i)) ** 2)
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        if best_idx is None or best_thr is None:
            return None, None, None, None, None, None

        X_left, X_right, y_left, y_right = split_dataset(X, y, best_idx, best_thr)
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return None, None, None, None, None, None
        return X_left, X_right, y_left, y_right, best_idx, best_thr

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _compute_feature_importances(self):
        def traverse_and_collect(node):
            if node.left is not None and node.right is not None:
                left_weight = node.left.num_samples / node.num_samples
                right_weight = node.right.num_samples / node.num_samples
                self.feature_importances_[node.feature_index] += node.gini - (left_weight * node.left.gini + right_weight * node.right.gini)
                traverse_and_collect(node.left)
                traverse_and_collect(node.right)

        traverse_and_collect(self.tree)
        self.feature_importances_ /= self.feature_importances_.sum()


def export_tree(tree, feature_names, class_colors, class_names):
    import matplotlib.colors as mcolors

    # Define the colors for different conditions
    light_grey = "#DDDDDD"  # Light green color
    light_yellow = "#F7F7F7"  # Light red color

    dot_data = ["digraph Tree {"]
    dot_data.append(
        'node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
    dot_data.append('edge [fontname="helvetica"] ;')

    def add_node(node, node_id):
        if node.left or node.right:
            threshold = node.threshold
            # Handle the special case for fixed splits
            if threshold == 0.5:
                threshold_str = "== 0"
                fillcolor = light_grey
            else:
                threshold_str = f"<= {threshold:.2f}"
                fillcolor = light_yellow

            dot_data.append(
                f'{node_id} [label="{feature_names[node.feature_index]} {threshold_str}\\n'
                f'gini = {node.gini:.2f}\\nsamples = {node.num_samples}\\n'
                f'value = {list(node.num_samples_per_class.values())}\\nclass = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')

            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            add_node(node.left, left_id)
            add_node(node.right, right_id)
            dot_data.append(f'{node_id} -> {left_id} ;')
            dot_data.append(f'{node_id} -> {right_id} ;')
        else:
            # Leaf node: use the color of the predicted class
            fillcolor = class_colors[node.predicted_class % len(class_colors)]
            dot_data.append(
                f'{node_id} [label="gini = {node.gini:.2f}\\nsamples = {node.num_samples}\\n'
                f'value = {list(node.num_samples_per_class.values())}\\nclass = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')

    add_node(tree.tree, 0)
    dot_data.append("}")
    return "\n".join(dot_data)

# Example usage with predefined data
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
#
# # Create a sample dataset
# X, y = make_classification(n_samples=10000, n_features=4, n_classes=2,
#                            random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     random_state=42)
#
# # Define the forced splits
# root_split = (0, 0.5)  # (feature_index, threshold)
# left_split = (1, 0.5)
# right_split = (2, 0.5)

# Define fixed splits as a nested dictionary
# fixed_splits = {
#     (0, 0.5): {
#         'left': {
#             (1, 0.0): {
#                 'left': {(2, 1.0): None},
#                 'right': None
#             }
#         },
#         'right': {(2, 1.5): None}
#     }
# }
# fixed_splits = {
#     (0, 0.5): {
#         'left': {(1, 0.5): None},
#         'right': {(2, 0.5): None}
#     }
# }
#
# custom_tree = CustomDecisionTree(fixed_splits, max_depth=100, min_samples_leaf=100)
# custom_tree.fit(X_train, y_train)
#
# dot_data = export_tree(custom_tree, [f'Feature {i}' for i in range(X.shape[1])])
# graph = graphviz.Source(dot_data)
# graph.render("custom_tree", format="pdf")
# print("Node Count:", custom_tree.node_count)  # Access node_count attribute
# print("Feature Importances:", custom_tree.feature_importances_)  # Access feature importances
