import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree, \
    plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import graphviz

from to_be_added.build_combined_tree import build_combined_tree


def get_leaf_samples_and_features(tree, X):
    """Group samples by their leaf nodes and extract feature indices used in decision paths."""
    leaf_samples_indices = {}
    leaf_features = {}
    leaf_indices = tree.apply(X)

    for leaf in np.unique(leaf_indices):
        sample_indices = np.where(leaf_indices == leaf)[0]
        leaf_samples_indices[leaf] = sample_indices
        leaf_features[leaf] = get_features_used_in_path(tree, X[sample_indices])

    return leaf_samples_indices, leaf_features


def get_features_used_in_path(tree, X):
    """Get the feature indices used in the decision path to each leaf node."""
    decision_paths = tree.decision_path(X)
    feature_indices = []

    for sample_id in range(X.shape[0]):
        path = decision_paths.indices[
               decision_paths.indptr[sample_id]: decision_paths.indptr[
                   sample_id + 1]
               ]
        features_in_path = np.unique(
            tree.tree_.feature[path][
                tree.tree_.feature[path] != _tree.TREE_UNDEFINED]
        )
        feature_indices.append(features_in_path)

    # Get unique feature indices across all paths for this leaf
    unique_features = np.unique(np.concatenate(feature_indices))

    return unique_features


def fit_trees_on_leaves(tree, X, y):
    """Fit a new decision tree for the data at each leaf of the original tree."""
    leaf_samples_indices, leaf_features = get_leaf_samples_and_features(tree, X)
    leaf_trees = {}
    for leaf, sample_indices in leaf_samples_indices.items():
        X_leaf, y_leaf = X[sample_indices], y[sample_indices]
        if len(np.unique(
                y_leaf)) > 1:  # Ensure there is more than one class to fit a tree
            new_tree = DecisionTreeClassifier(min_samples_leaf=1,
                                              random_state=0)
            new_tree.fit(X_leaf, y_leaf)
            leaf_trees[leaf] = new_tree
    return leaf_trees, leaf_features


def export_tree_to_graphviz(tree, feature_names, class_names, filename):
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename, format='png', cleanup=True)


# Example usage
# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# Train the decision tree classifier on the training set with min_samples_leaf=20
clf = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)
clf.fit(X_train, y_train)

# Fit new trees for the data at the leaf of each decision path with min_samples_leaf=1
leaf_trees, leaf_features = fit_trees_on_leaves(clf, X_train, y_train)

# Create a directory for the exported trees
output_dir = './dt_with_dts'
os.makedirs(output_dir, exist_ok=True)

# Export the original tree
print("Exporting original tree...")
export_tree_to_graphviz(clf, iris.feature_names, iris.target_names,
                        os.path.join(output_dir, 'original_tree'))

# Export each fitted tree for the leaf nodes
for leaf, tree in leaf_trees.items():
    print(f"Exporting leaf tree {leaf}...")
    export_tree_to_graphviz(tree, iris.feature_names, iris.target_names,
                            os.path.join(output_dir, f'leaf_tree_{leaf}'))

# Print feature indices used in decision paths to each leaf
for leaf, features in leaf_features.items():
    print(f"Features used in decision path to leaf {leaf}: {features}")

# Building the combined tree
print("Building the combined tree...")
combined_tree = build_combined_tree(clf, leaf_trees, X, y)

# plot the combined tree
plt.figure(figsize=(20, 10))
plt.axis('off')
plot_tree(combined_tree, class_names=iris.target_names, feature_names=iris.feature_names,
         label='all', rounded=True)
plt.show()

print(f"All trees have been exported to {output_dir}")
