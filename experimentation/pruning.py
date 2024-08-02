import numpy as np
from sklearn.tree import _tree, DecisionTreeClassifier, export_graphviz


def prune(tree):
    def is_leaf(node):
        return (tree.children_left[node] == _tree.TREE_LEAF and
                tree.children_right[node] == _tree.TREE_LEAF)

    def can_prune(node):
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        if is_leaf(left_child) and is_leaf(right_child):
            left_class = np.argmax(tree.value[left_child])
            right_class = np.argmax(tree.value[right_child])
            return left_class == right_class

        return False

    def prune_nodes(node):
        if node == _tree.TREE_LEAF:
            return

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        prune_nodes(left_child)
        prune_nodes(right_child)

        if can_prune(node):
            tree.children_left[node] = _tree.TREE_LEAF
            tree.children_right[node] = _tree.TREE_LEAF
            tree.feature[node] = -2  # Set feature to undefined
            tree.threshold[node] = -2.0  # Set threshold to undefined
            # Update the node value with the sum of the child values
            #tree.value[node] = tree.value[left_child] + tree.value[right_child]

    prune_nodes(0)

    # Identify the nodes to keep
    nodes_to_keep = []
    def collect_nodes(node):
        if node == _tree.TREE_LEAF:
            return
        nodes_to_keep.append(node)
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]
        if left_child != _tree.TREE_LEAF:
            collect_nodes(left_child)
        if right_child != _tree.TREE_LEAF:
            collect_nodes(right_child)

    collect_nodes(0)

    # Map old node ids to new node ids
    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(nodes_to_keep)}

    # Create new arrays for the pruned tree
    total_nodes = len(nodes_to_keep)
    children_left = np.full(total_nodes, _tree.TREE_LEAF, dtype=np.int64)
    children_right = np.full(total_nodes, _tree.TREE_LEAF, dtype=np.int64)
    feature = np.full(total_nodes, -2, dtype=np.int64)
    threshold = np.full(total_nodes, -2.0, dtype=np.float64)
    impurity = np.zeros(total_nodes, dtype=np.float64)
    n_node_samples = np.zeros(total_nodes, dtype=np.int64)
    weighted_n_node_samples = np.zeros(total_nodes, dtype=np.float64)
    value = np.zeros((total_nodes, 1, tree.value.shape[2]), dtype=np.float64)

    # Copy the attributes of the remaining nodes
    for old_id, new_id in old_to_new_id.items():
        children_left[new_id] = old_to_new_id.get(tree.children_left[old_id], _tree.TREE_LEAF)
        children_right[new_id] = old_to_new_id.get(tree.children_right[old_id], _tree.TREE_LEAF)
        feature[new_id] = tree.feature[old_id]
        threshold[new_id] = tree.threshold[old_id]
        impurity[new_id] = tree.impurity[old_id]
        n_node_samples[new_id] = tree.n_node_samples[old_id]
        weighted_n_node_samples[new_id] = tree.weighted_n_node_samples[old_id]
        value[new_id, 0, :] = tree.value[old_id, 0, :]

    # Create a new Tree object with the pruned structure
    n_classes = tree.value.shape[2]
    pruned_tree_ = _tree.Tree(tree.n_features, np.array([n_classes], dtype=np.int64), 1)
    pruned_tree_.capacity = total_nodes
    pruned_tree_.node_count = total_nodes

    pruned_tree_.children_left[:total_nodes] = children_left
    pruned_tree_.children_right[:total_nodes] = children_right
    pruned_tree_.feature[:total_nodes] = feature
    pruned_tree_.threshold[:total_nodes] = threshold
    pruned_tree_.impurity[:total_nodes] = impurity
    pruned_tree_.n_node_samples[:total_nodes] = n_node_samples
    pruned_tree_.weighted_n_node_samples[:total_nodes] = weighted_n_node_samples
    pruned_tree_.value[:total_nodes, :, :] = value

    return pruned_tree_

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

    # Prune the tree
    pruned_tree = prune(clf.tree_)

    # Create a new DecisionTreeClassifier with the pruned tree
    pruned_clf = DecisionTreeClassifier(random_state=0)
    pruned_clf.tree_ = pruned_tree
    pruned_clf.n_features_in_ = clf.n_features_in_

    # Export the pruned tree to Graphviz format
    dot_data = export_graphviz(
        pruned_clf,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Render the pruned tree
    graph = graphviz.Source(dot_data)
    graph.render("pruned_tree", format='png', cleanup=True)

example_usage()
