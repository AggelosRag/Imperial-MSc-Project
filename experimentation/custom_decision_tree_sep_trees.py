import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

# Create a sample dataset
X, y = make_classification(n_samples=10000, n_features=4, n_classes=2,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Define the forced splits
root_split = (0, 0.5)  # (feature_index, threshold)
left_split = (1, 0.5)
right_split = (2, 0.5)


# Custom function to apply the first three forced splits
def custom_split(X, y, root_split, left_split, right_split):
    root_feature_idx, root_threshold = root_split
    left_feature_idx, left_threshold = left_split
    right_feature_idx, right_threshold = right_split

    indices_root_left = X[:, root_feature_idx] <= root_threshold
    X_root_left, y_root_left = X[indices_root_left], y[indices_root_left]
    X_root_right, y_root_right = X[~indices_root_left], y[~indices_root_left]

    indices_left_left = X_root_left[:, left_feature_idx] <= left_threshold
    X_left_left, y_left_left = X_root_left[indices_left_left], y_root_left[
        indices_left_left]
    X_left_right, y_left_right = X_root_left[~indices_left_left], y_root_left[
        ~indices_left_left]

    indices_right_left = X_root_right[:, right_feature_idx] <= right_threshold
    X_right_left, y_right_left = X_root_right[indices_right_left], y_root_right[
        indices_right_left]
    X_right_right, y_right_right = X_root_right[~indices_right_left], \
    y_root_right[~indices_right_left]

    return (X_left_left, y_left_left, X_left_right, y_left_right,
            X_right_left, y_right_left, X_right_right, y_right_right)


# Apply the custom split
(X_left_left, y_left_left, X_left_right, y_left_right,
 X_right_left, y_right_left, X_right_right, y_right_right) = custom_split(
    X_train, y_train, root_split, left_split, right_split)

# Train a decision tree on each of the remaining subsets
tree_left_left = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_left_left.fit(X_left_left, y_left_left)

tree_left_right = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_left_right.fit(X_left_right, y_left_right)

tree_right_left = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_right_left.fit(X_right_left, y_right_left)

tree_right_right = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_right_right.fit(X_right_right, y_right_right)


# Define a function to manually create the first three splits and integrate them into the fitted trees
class CustomDecisionTree:
    def __init__(self, root_split, left_split, right_split, trees):
        self.root_split = root_split
        self.left_split = left_split
        self.right_split = right_split
        self.trees = trees
        self.n_features = trees["left_left"].n_features_in_

    def predict(self, X):
        def traverse(tree, x):
            if tree.tree_.children_left[0] == tree.tree_.children_right[0] == -1:
                return tree.tree_.value[0].argmax()
            node = 0
            while tree.tree_.children_left[node] != -1:
                if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                    node = tree.tree_.children_left[node]
                else:
                    node = tree.tree_.children_right[node]
            return tree.tree_.value[node].argmax()

        predictions = []
        for x in X:
            if x[self.root_split[0]] <= self.root_split[1]:
                if x[self.left_split[0]] <= self.left_split[1]:
                    pred = traverse(self.trees['left_left'], x)
                else:
                    pred = traverse(self.trees['left_right'], x)
            else:
                if x[self.right_split[0]] <= self.right_split[1]:
                    pred = traverse(self.trees['right_left'], x)
                else:
                    pred = traverse(self.trees['right_right'], x)
            predictions.append(pred)
        return np.array(predictions)

    def plot_tree(self):
        subtrees = ['left_left', 'left_right', 'right_left', 'right_right']
        dot_files = {}

        for subtree in subtrees:
            dot_data = export_graphviz(self.trees[subtree], out_file=None, filled=True, rounded=True,
                                       special_characters=True,
                                       feature_names=[f'Feature {i}' for i in range(self.n_features)])
            dot_files[subtree] = dot_data

        for name, dot in dot_files.items():
            graph = graphviz.Source(dot)
            graph.render(filename=f"{name}_tree", format="pdf")

    # def plot_tree(self):
    #     fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    #
    #     plot_tree(self.trees['left_left'], filled=True,
    #               feature_names=[f'Feature {i}' for i in
    #                              range(self.n_features)], ax=axs[0, 0],
    #               rounded=True)
    #     axs[0, 0].set_title("Left-Left Subtree")
    #
    #     plot_tree(self.trees['left_right'], filled=True,
    #               feature_names=[f'Feature {i}' for i in
    #                              range(self.n_features)], ax=axs[0, 1],
    #               rounded=True)
    #     axs[0, 1].set_title("Left-Right Subtree")
    #
    #     plot_tree(self.trees['right_left'], filled=True,
    #               feature_names=[f'Feature {i}' for i in
    #                              range(self.n_features)], ax=axs[1, 0],
    #               rounded=True)
    #     axs[1, 0].set_title("Right-Left Subtree")
    #
    #     plot_tree(self.trees['right_right'], filled=True,
    #               feature_names=[f'Feature {i}' for i in
    #                              range(self.n_features)], ax=axs[1, 1],
    #               rounded=True)
    #     axs[1, 1].set_title("Right-Right Subtree")
    #
    #     plt.show()


# Create a custom tree
trees = {
    'left_left': tree_left_left,
    'left_right': tree_left_right,
    'right_left': tree_right_left,
    'right_right': tree_right_right
}
custom_tree = CustomDecisionTree(root_split, left_split, right_split, trees)

# Predict using the custom predict function
predictions = custom_tree.predict(X_test)
print(predictions)

# Plot the decision tree
custom_tree.plot_tree()


# Function to create a combined DOT representation of the tree
def create_combined_dot(root_split, left_split, right_split, trees):
    dot_data = """
    digraph Tree {
        node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
        edge [fontname=helvetica] ;
    """

    node_id = 0

    def add_node(node_id, feature, threshold, samples, value):
        label = f'feature {feature} <= {threshold}\\nsamples = {samples}\\nvalue = {value}'
        return f'{node_id} [label="{label}", fillcolor="#ffffff"] ;\n'

    def add_leaf(node_id, samples, value):
        label = f'samples = {samples}\\nvalue = {value}'
        return f'{node_id} [label="{label}", fillcolor="#ffffff"] ;\n'

    # Root node
    root_feature_idx, root_threshold = root_split
    dot_data += add_node(node_id, root_feature_idx, root_threshold, len(y_train), np.bincount(y_train))
    root_node_id = node_id
    node_id += 1

    # Left split
    left_feature_idx, left_threshold = left_split
    dot_data += add_node(node_id, left_feature_idx, left_threshold, len(y_train[y_train <= left_threshold]), np.bincount(y_train[y_train <= left_threshold]))
    root_left_node_id = node_id
    node_id += 1

    dot_data += f'{root_node_id} -> {root_left_node_id} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'

    # Right split
    right_feature_idx, right_threshold = right_split
    dot_data += add_node(node_id, right_feature_idx, right_threshold, len(y_train[y_train > left_threshold]), np.bincount(y_train[y_train > left_threshold]))
    root_right_node_id = node_id
    node_id += 1

    dot_data += f'{root_node_id} -> {root_right_node_id} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'

    # Add subtrees
    def add_subtree(dot_data, tree, parent_node_id, edge_label, node_id):
        tree_dot_data = export_graphviz(tree, out_file=None, filled=True)
        for line in tree_dot_data.splitlines():
            if '->' in line:
                parts = line.split(' -> ')
                from_node = int(parts[0].strip())
                to_node = int(parts[1].split('[')[0].strip().replace(';', ''))
                dot_data += f'{parent_node_id} -> {node_id + to_node} [label="{edge_label}"] ;\n'
            elif 'label=' in line:
                line_id = int(line.split(' ')[0])
                dot_data += line.replace(f'{line_id}',
                                         f'{node_id + line_id}') + '\n'
        return dot_data, node_id + len(tree.tree_.feature)

    # Add left-left subtree
    dot_data, node_id = add_subtree(dot_data, tree_left_left, root_left_node_id, 'True', node_id)
    # Add left-right subtree
    dot_data, node_id = add_subtree(dot_data, tree_left_right, root_left_node_id, 'False', node_id)
    # Add right-left subtree
    dot_data, node_id = add_subtree(dot_data, tree_right_left, root_right_node_id, 'True', node_id)
    # Add right-right subtree
    dot_data, node_id = add_subtree(dot_data, tree_right_right, root_right_node_id, 'False', node_id)

    dot_data += '}'
    return dot_data

# Create the combined DOT data
dot_data = create_combined_dot(root_split, left_split, right_split, {
    'left_left': tree_left_left,
    'left_right': tree_left_right,
    'right_left': tree_right_left,
    'right_right': tree_right_right
})

# Visualize the combined tree using graphviz
graph = graphviz.Source(dot_data)
graph.render("combined_tree")  # Save the visualization to a file
graph.view()  # Display the visualization