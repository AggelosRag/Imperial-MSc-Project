import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
import hashlib
from sklearn.inspection import PartialDependenceDisplay

def visualize_decision_tree(clf, feature_names, class_names, title):
    dot_data = export_graphviz(
        clf, out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render('./trees/' + title)  # Save the graph as a .pdf file
    return graph


def compute_structural_similarity(tree1, tree2):
    depth1 = tree1.get_depth()
    depth2 = tree2.get_depth()
    num_nodes1 = tree1.tree_.node_count
    num_nodes2 = tree2.tree_.node_count

    # Node splits comparison
    splits1 = tree1.tree_.threshold
    splits2 = tree2.tree_.threshold
    splits_similarity = np.mean(splits1 == splits2)

    # Compare splitting criteria of internal nodes (features and thresholds)
    tree1_nodes = tree1.tree_
    tree2_nodes = tree2.tree_

    split_features1 = tree1_nodes.feature[tree1_nodes.feature >= 0]
    split_features2 = tree2_nodes.feature[tree2_nodes.feature >= 0]

    split_thresholds1 = tree1_nodes.threshold[tree1_nodes.feature >= 0]
    split_thresholds2 = tree2_nodes.threshold[tree2_nodes.feature >= 0]

    feature_similarity = np.mean(split_features1 == split_features2)
    threshold_similarity = np.mean(
        np.isclose(split_thresholds1, split_thresholds2, atol=1e-4))

    structural_similarity = {
        'depth_difference': abs(depth1 - depth2),
        'num_nodes_difference': abs(num_nodes1 - num_nodes2),
        'splits_similarity': splits_similarity,
        'feature_similarity': feature_similarity,
        'threshold_similarity': threshold_similarity
    }

    return structural_similarity

def compare_feature_importances(tree1, tree2, feature_names):
    importances1 = tree1.feature_importances_
    importances2 = tree2.feature_importances_

    plt.figure(figsize=(10, 6))
    indices = np.arange(len(feature_names))
    width = 0.35
    plt.bar(indices - width/2, importances1, width, label='Tree 1')
    plt.bar(indices + width/2, importances2, width, label='Tree 2')
    plt.xticks(indices, feature_names, rotation=90)
    plt.ylabel('Feature Importance')
    plt.title('Comparison of Feature Importances')
    plt.legend(loc='best')
    plt.show()


def plot_partial_dependence_trees(tree1, tree2, X, feature_names, target=0):
    fig, ax = plt.subplots(2, len(feature_names), figsize=(15, 6), sharey=True)
    for i, feature in enumerate(feature_names):
        PartialDependenceDisplay.from_estimator(tree1, X, [i], target=target, ax=ax[0, i], feature_names=feature_names, grid_resolution=50)
        PartialDependenceDisplay.from_estimator(tree2, X, [i], target=target, ax=ax[1, i], feature_names=feature_names, grid_resolution=50)
    ax[0, 0].set_ylabel('Tree 1')
    ax[1, 0].set_ylabel('Tree 2')
    plt.suptitle('Partial Dependence Plots')
    plt.show()

def compute_semantic_similarity(tree1, tree2, X):
    leaf_ids1 = tree1.apply(X)
    leaf_ids2 = tree2.apply(X)

    # Leaf node distribution
    unique_leaf_ids1, counts1 = np.unique(leaf_ids1, return_counts=True)
    unique_leaf_ids2, counts2 = np.unique(leaf_ids2, return_counts=True)

    leaf_dist1 = dict(zip(unique_leaf_ids1, counts1))
    leaf_dist2 = dict(zip(unique_leaf_ids2, counts2))

    common_leaves = set(leaf_dist1.keys()).intersection(set(leaf_dist2.keys()))
    leaf_similarity = sum(min(leaf_dist1[leaf], leaf_dist2[leaf]) for leaf in
                          common_leaves) / min(sum(counts1), sum(counts2))

    # Path similarity
    paths1 = [tree1.decision_path([x]).toarray() for x in X]
    paths2 = [tree2.decision_path([x]).toarray() for x in X]
    path_similarity = np.mean(
        [np.array_equal(p1, p2) for p1, p2 in zip(paths1, paths2)])

    semantic_similarity = {
        'leaf_similarity': leaf_similarity,
        'path_similarity': path_similarity
    }

    return semantic_similarity

def serialize_tree(tree):
    # Serialize the tree structure and node values
    return hashlib.md5(tree.tree_.__getstate__()['nodes'].tobytes()).hexdigest()

def main():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Number of different random states to visualize
    num_trees = 1000
    performance_threshold = 0.02  # Performance range (e.g., within 2% of the best performance)
    min_performance = 0.9  # Minimum performance to consider (e.g., accuracy of 90%)

    # First pass to collect all scores
    scores = []
    classifiers = []

    for i in range(num_trees):
        random_state = i
        clf = DecisionTreeClassifier(random_state=random_state, min_samples_leaf=5)
        clf.fit(X, y)  # Train on the full dataset
        y_pred = clf.predict(X)
        mean_score = accuracy_score(y, y_pred)
        classifiers.append((clf, random_state, mean_score))
        scores.append(mean_score)

    # Determine the best score
    best_score = max(scores)

    # Filter classifiers with similar performance
    similar_trees = [
        (clf, random_state, mean_score)
        for clf, random_state, mean_score in classifiers
        if best_score - performance_threshold <= mean_score <= best_score + performance_threshold and mean_score >= min_performance
    ]

    # Remove identical trees
    unique_trees = []
    seen_hashes = set()

    for clf, random_state, mean_score in similar_trees:
        tree_hash = serialize_tree(clf)
        if tree_hash not in seen_hashes:
            seen_hashes.add(tree_hash)
            unique_trees.append((clf, random_state, mean_score))

    # Visualize the unique trees with similar performance
    for clf, random_state, mean_score in unique_trees:
        title = f"tree_random_state_{random_state}_score_{mean_score:.4f}"
        graph = visualize_decision_tree(clf, feature_names, class_names, title)
        #graph.view()

    # Example comparison of two trees
    if len(unique_trees) >= 2:
        tree1, _, _ = unique_trees[0]
        tree2, _, _ = unique_trees[1]

        structural_similarity = compute_structural_similarity(tree1, tree2)
        semantic_similarity = compute_semantic_similarity(tree1, tree2, X)

        print("Structural Similarity:", structural_similarity)
        print("Semantic Similarity:", semantic_similarity)

        compare_feature_importances(tree1, tree2, feature_names)
        plot_partial_dependence_trees(tree1, tree2, X, feature_names, target=0)  # Change target as needed


if __name__ == "__main__":
    main()
