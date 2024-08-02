import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz


def build_combined_tree_scikit(original_tree, leaf_trees, X, y):
    """Construct a new tree where the leaves of the original tree are replaced with new trees."""
    original_tree_ = original_tree.tree_

    # Calculate the total number of nodes needed (including leaves of the original tree)
    leaf_node_ids = np.where((original_tree_.children_left == _tree.TREE_LEAF) & (original_tree_.children_right == _tree.TREE_LEAF))[0]
    non_leaf_nodes = original_tree_.node_count - len(leaf_node_ids)
    leaf_node_ids_having_tree = list(leaf_trees.keys())
    leaf_node_ids_not_having_tree = [id for id in leaf_node_ids if id not in leaf_node_ids_having_tree]
    total_nodes = non_leaf_nodes + len(leaf_node_ids_not_having_tree) + sum(leaf.tree_.node_count for leaf in leaf_trees.values())

    print(f"Total nodes in the combined tree: {total_nodes}")

    # Initialize arrays for the new tree
    n_classes = len(np.unique(y))
    children_left = np.full(total_nodes, _tree.TREE_LEAF, dtype=np.int64)
    children_right = np.full(total_nodes, _tree.TREE_LEAF, dtype=np.int64)
    feature = np.full(total_nodes, -2, dtype=np.int64)
    threshold = np.full(total_nodes, -2.0, dtype=np.float64)
    impurity = np.zeros(total_nodes, dtype=np.float64)
    n_node_samples = np.zeros(total_nodes, dtype=np.int64)
    weighted_n_node_samples = np.zeros(total_nodes, dtype=np.float64)
    value = np.zeros((total_nodes, 1, n_classes), dtype=np.float64)

    # Copy the structure of the original tree for non-leaf nodes
    for i in range(original_tree_.node_count):
        if i not in leaf_node_ids_having_tree:
            children_left[i] = original_tree_.children_left[i]
            children_right[i] = original_tree_.children_right[i]
            feature[i] = original_tree_.feature[i]
            threshold[i] = original_tree_.threshold[i]
            impurity[i] = original_tree_.impurity[i]
            n_node_samples[i] = original_tree_.n_node_samples[i]
            weighted_n_node_samples[i] = original_tree_.weighted_n_node_samples[i]
            value[i, 0, :original_tree_.value.shape[2]] = original_tree_.value[i, 0, :]

    node_offset = original_tree_.node_count
    # Replace the leaf nodes with the new trees
    for leaf_node_id in leaf_node_ids_having_tree:
        new_tree = leaf_trees[leaf_node_id]
        new_tree_ = new_tree.tree_

        # Add new nodes from the subtree
        left_of_root = new_tree_.children_left[0]
        right_of_root = new_tree_.children_right[0]
        feature_of_root = new_tree_.feature[0]
        threshold_of_root = new_tree_.threshold[0]
        impurity_of_root = new_tree_.impurity[0]
        n_node_samples_of_root = new_tree_.n_node_samples[0]
        weighted_n_node_samples_of_root = new_tree_.weighted_n_node_samples[0]
        value_of_root = new_tree_.value[0, 0, :]

        new_tree_children_left_without_root = new_tree_.children_left[1:]
        new_tree_children_right_without_root = new_tree_.children_right[1:]
        new_tree_features_without_root = new_tree_.feature[1:]
        new_tree_thresholds_without_root = new_tree_.threshold[1:]
        new_tree_impurities_without_root = new_tree_.impurity[1:]
        new_tree_n_node_samples_without_root = new_tree_.n_node_samples[1:]
        new_tree_weighted_n_node_samples_without_root = new_tree_.weighted_n_node_samples[1:]
        new_tree_values_without_root = new_tree_.value[1:, 0, :]

        start_id = node_offset
        end_id = start_id + new_tree_.node_count - 1

        children_left[leaf_node_id] = np.where(left_of_root == _tree.TREE_LEAF, _tree.TREE_LEAF, left_of_root + (start_id-1))
        children_right[leaf_node_id] = np.where(right_of_root == _tree.TREE_LEAF, _tree.TREE_LEAF, right_of_root + (start_id-1))
        children_left[start_id:end_id] = np.where(new_tree_children_left_without_root == _tree.TREE_LEAF, _tree.TREE_LEAF, new_tree_children_left_without_root + (start_id-1))
        children_right[start_id:end_id] = np.where(new_tree_children_right_without_root == _tree.TREE_LEAF, _tree.TREE_LEAF, new_tree_children_right_without_root + (start_id-1))
        feature[leaf_node_id] = feature_of_root
        feature[start_id:end_id] = new_tree_features_without_root
        threshold[leaf_node_id] = threshold_of_root
        threshold[start_id:end_id] = new_tree_thresholds_without_root
        impurity[leaf_node_id] = impurity_of_root
        impurity[start_id:end_id] = new_tree_impurities_without_root
        n_node_samples[leaf_node_id] = n_node_samples_of_root
        n_node_samples[start_id:end_id] = new_tree_n_node_samples_without_root
        weighted_n_node_samples[leaf_node_id] = weighted_n_node_samples_of_root
        weighted_n_node_samples[start_id:end_id] = new_tree_weighted_n_node_samples_without_root
        value[leaf_node_id, 0, :new_tree_.value.shape[2]] = value_of_root
        value[start_id:end_id, 0, :new_tree_.value.shape[2]] = new_tree_values_without_root

        node_offset = end_id

    # Create a new Tree object with the combined structure
    combined_tree_ = _tree.Tree(X.shape[1], np.array([n_classes], dtype=np.int64), 1)
    combined_tree_.capacity = total_nodes
    combined_tree_.node_count = total_nodes

    # Set the attributes using the Tree's API
    combined_tree_.children_left[:total_nodes] = children_left
    combined_tree_.children_right[:total_nodes] = children_right
    combined_tree_.feature[:total_nodes] = feature
    combined_tree_.threshold[:total_nodes] = threshold
    combined_tree_.impurity[:total_nodes] = impurity
    combined_tree_.n_node_samples[:total_nodes] = n_node_samples
    combined_tree_.weighted_n_node_samples[:total_nodes] = weighted_n_node_samples
    combined_tree_.value[:total_nodes, :, :] = value

    combined_tree = DecisionTreeClassifier(random_state=0)
    combined_tree.tree_ = combined_tree_

    # Manually set n_features_in_ for the combined tree
    combined_tree.n_features_in_ = X.shape[1]

    # Print combined tree attributes for debugging
    print("Combined tree attributes:")
    print(f"Children left: {combined_tree_.children_left}")
    print(f"Children right: {combined_tree_.children_right}")
    print(f"Feature: {combined_tree_.feature}")
    print(f"Threshold: {combined_tree_.threshold}")
    print(f"Impurity: {combined_tree_.impurity}")
    print(f"n_node_samples: {combined_tree_.n_node_samples}")
    print(f"weighted_n_node_samples: {combined_tree_.weighted_n_node_samples}")
    print(f"Value: {combined_tree_.value}")

    return combined_tree



# Example usage
def example_usage():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import graphviz

    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train the decision tree classifier on the training set
    clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=20)
    clf.fit(X_train, y_train)

    # Fit new trees for the data at the leaf of each decision path with min_samples_leaf=1
    leaf_trees = {}
    for leaf_id in np.where((clf.tree_.children_left == _tree.TREE_LEAF) & (clf.tree_.children_right == _tree.TREE_LEAF))[0]:
        indices = np.where(clf.apply(X_train) == leaf_id)[0]
        X_leaf, y_leaf = X_train[indices], y_train[indices]
        if len(np.unique(y_leaf)) > 1:  # Ensure there is more than one class to fit a tree
            leaf_tree = DecisionTreeClassifier(min_samples_leaf=1, random_state=0)
            leaf_tree.fit(X_leaf, y_leaf)
            leaf_trees[leaf_id] = leaf_tree

    # Build combined tree
    combined_tree = build_combined_tree(clf, leaf_trees, X_train, y_train)

    # Export the combined tree to Graphviz format
    dot_data = export_graphviz(
        combined_tree,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=False,
        rounded=True,
        special_characters=True
    )

    # Render the combined tree
    graph = graphviz.Source(dot_data)
    graph.render("combined_tree", format='png', cleanup=True)

example_usage()