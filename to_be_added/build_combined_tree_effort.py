import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree


def build_combined_tree(original_tree, leaf_trees, X, y):
    """Construct a new tree where the leaves of the original tree are replaced with new trees."""
    original_tree_ = original_tree.tree_

    # Calculate the total number of nodes needed (excluding leaves of the original tree)
    non_leaf_nodes = np.where(
        (original_tree_.children_left != _tree.TREE_LEAF) & (
                    original_tree_.children_right != _tree.TREE_LEAF))[0]
    total_nodes = original_tree_.node_count
    for leaf in leaf_trees.values():
        total_nodes += leaf.tree_.node_count

    print(f"Total nodes in the combined tree: {total_nodes - len(leaf_trees)}")

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

    # Copy the structure of the original tree
    for i in range(original_tree_.node_count):
        children_left[i] = original_tree_.children_left[i]
        children_right[i] = original_tree_.children_right[i]
        feature[i] = original_tree_.feature[i]
        threshold[i] = original_tree_.threshold[i]
        impurity[i] = original_tree_.impurity[i]
        n_node_samples[i] = original_tree_.n_node_samples[i]
        weighted_n_node_samples[i] = original_tree_.weighted_n_node_samples[i]
        value[i, 0, :original_tree_.value.shape[2]] = original_tree_.value[i, 0,
                                                      :]

    # Replace the leaf nodes with the new trees
    node_offset = original_tree_.node_count
    leaf_node_ids = np.where(
        (original_tree_.children_left == _tree.TREE_LEAF) & (
                    original_tree_.children_right == _tree.TREE_LEAF))[0]

    for leaf_node_id in leaf_node_ids:
        if leaf_node_id in leaf_trees:
            new_tree = leaf_trees[leaf_node_id]
            new_tree_ = new_tree.tree_

            # Add new nodes from the subtree
            start_id = node_offset
            end_id = start_id + new_tree_.node_count

            children_left[start_id:end_id] = np.where(
                new_tree_.children_left == _tree.TREE_LEAF, _tree.TREE_LEAF,
                new_tree_.children_left + start_id)
            children_right[start_id:end_id] = np.where(
                new_tree_.children_right == _tree.TREE_LEAF, _tree.TREE_LEAF,
                new_tree_.children_right + start_id)
            feature[start_id:end_id] = new_tree_.feature
            threshold[start_id:end_id] = new_tree_.threshold
            impurity[start_id:end_id] = new_tree_.impurity
            n_node_samples[start_id:end_id] = new_tree_.n_node_samples
            weighted_n_node_samples[
            start_id:end_id] = new_tree_.weighted_n_node_samples
            value[start_id:end_id, 0, :new_tree_.value.shape[2]] = new_tree_.value[:, 0, :]

            # Update the original tree to point to the new subtree
            children_left[leaf_node_id] = start_id
            children_right[leaf_node_id] = start_id + 1

            # Update the node offset
            node_offset = end_id

    # Create a new Tree object with the combined structure
    combined_tree_ = _tree.Tree(X.shape[1],
                                np.array([n_classes], dtype=np.int64), 1)
    combined_tree_.capacity = total_nodes
    combined_tree_.node_count = total_nodes

    # Set the attributes using the Tree's API
    combined_tree_.children_left[:total_nodes] = children_left
    combined_tree_.children_right[:total_nodes] = children_right
    combined_tree_.feature[:total_nodes] = feature
    combined_tree_.threshold[:total_nodes] = threshold
    combined_tree_.impurity[:total_nodes] = impurity
    combined_tree_.n_node_samples[:total_nodes] = n_node_samples
    combined_tree_.weighted_n_node_samples[
    :total_nodes] = weighted_n_node_samples
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
