import re
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import hashlib
import graphviz
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz, DecisionTreeClassifier,  _tree


def extract_features_from_splits(tree):
    feature_indices = set()

    def recurse(node):
        if tree.feature[node] != -2:  # -2 indicates a leaf node
            feature_indices.add(tree.feature[node])
            recurse(tree.children_left[node])
            recurse(tree.children_right[node])

    recurse(0)
    return sorted(list(feature_indices))


def get_light_colors(num_colors):
    # Use a light palette from seaborn
    palette = sns.color_palette("pastel", num_colors)
    # Convert to hex colors
    light_colors = [mcolors.rgb2hex(color) for color in palette]
    return light_colors


def replace_splits(dot_data, old_split="&le; 0.5", new_split="== 0"):
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        new_line = line.replace(old_split, new_split)
        new_lines.append(new_line)
    return '\n'.join(new_lines)


def modify_dot_with_colors(dot_data, color_map, clf, node_color="#DDDDDD"):
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        match = re.match(r'(\d+) \[label=.*\]', line)
        if match:
            node_id = int(match.group(1))
            if (clf.children_left[node_id] != _tree.TREE_LEAF or
                clf.children_right[node_id] != _tree.TREE_LEAF):
                # Node is not a leaf
                color = node_color
            else:
                # Node is a leaf
                node_class = clf.value[node_id].argmax()
                color = color_map[node_class]
            # Add fillcolor and style to the node definition
            line = re.sub(r'(?<=>)\]',
                          f', style="filled,rounded", fillcolor="{color}"]',
                          line)
        new_lines.append(line)
    return '\n'.join(new_lines)

def get_leaf_samples_and_features(tree, X):
    """Group samples by their leaf nodes and extract feature indices used in decision paths."""
    leaf_samples_indices = {}
    leaf_features_per_path = {}
    leaf_indices = tree.apply(X)

    for leaf in np.unique(leaf_indices):
        sample_indices = np.where(leaf_indices == leaf)[0]
        leaf_samples_indices[leaf] = sample_indices
        decision_path = tree.decision_path(X[sample_indices])
        leaf_features_per_path[leaf] = decision_path.feature_indices[decision_path.feature_indptr[0]: decision_path.feature_indptr[1]]

    return leaf_samples_indices, leaf_features_per_path

def get_features_used_in_path(tree, X):
    """Get the feature indices used in the decision path to each leaf node."""
    decision_paths = tree.decision_path(X)
    feature_indices = []


    for sample_id in range(X.shape[0]):
        path = decision_paths.indices[
               decision_paths.indptr[sample_id]: decision_paths.indptr[
                   sample_id + 1]
               ]
        features_in_path = np.unique([tree.tree.feature_index if tree.tree.feature_index != -2 else -1 for node_id in path]
        )
        feature_indices.append(features_in_path)

    # Get unique feature indices across all paths for this leaf
    unique_features = np.unique(np.concatenate(feature_indices))

    return unique_features

# def get_features_used_in_path(tree, X):
#     """Get the feature indices used in the decision path to each leaf node."""
#     decision_paths = tree.decision_path(X)
#     feature_indices = []
#
#     for sample_id in range(X.shape[0]):
#         path = decision_paths.indices[
#                decision_paths.indptr[sample_id]: decision_paths.indptr[
#                    sample_id + 1]
#                ]
#         features_in_path = np.unique(
#             tree.tree_.feature[path][
#                 tree.tree_.feature[path] != _tree.TREE_UNDEFINED]
#         )
#         feature_indices.append(features_in_path)
#
#     # Get unique feature indices across all paths for this leaf
#     unique_features = np.unique(np.concatenate(feature_indices))
#
#     return unique_features

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


def extract_paths_and_counts(tree, X, y, feature_names):
    paths = []
    path_counts = []
    path_classifications = []
    path_indices = []

    def traverse(node, path, indices):
        if tree.children_left[node] == tree.children_right[node]:  # leaf
            paths.append(path)
            path_counts.append(len(indices))
            path_classifications.append(np.bincount(y[indices]).argmax())
            path_indices.append(indices)
            return
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        left_indices = indices[X[indices, tree.feature[node]] <= threshold]
        right_indices = indices[X[indices, tree.feature[node]] > threshold]
        left_path = path + [(feature, "<=", threshold)]
        right_path = path + [(feature, ">", threshold)]
        traverse(tree.children_left[node], left_path, left_indices)
        traverse(tree.children_right[node], right_path, right_indices)

    traverse(0, [], np.arange(X.shape[0]))
    return paths, path_counts, path_classifications, path_indices

def calculate_accuracy_per_path(path_indices, y, path_classifications):
    accuracies = []
    for indices, classification in zip(path_indices, path_classifications):
        if len(indices) > 0:
            accuracy = np.mean(np.array(y[indices] == classification))
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)
    return accuracies

def print_paths_with_data(paths, path_counts, path_classifications, path_indices, accuracies):
    for path, count, classification, indices, accuracy in zip(paths, path_counts, path_classifications, path_indices, accuracies):
        path_str = " AND ".join([f"{feature} {op} {threshold:.2f}" for feature, op, threshold in path])
        print(f"Path: {path_str}")
        print(f"Data indices: {indices}")
        print(f"Count: {count}")
        print(f"Classification: {classification}")
        print(f"Accuracy: {accuracy:.4f}\n")

def merge_conditions(path):
    condition_dict = {}
    for feature, op, threshold in path:
        if feature not in condition_dict:
            condition_dict[feature] = [None, None]  # [lower_bound, upper_bound]
        if op == "<=":
            condition_dict[feature][1] = threshold
        elif op == ">":
            condition_dict[feature][0] = threshold

    merged_path = []
    for feature, (lower, upper) in condition_dict.items():
        if lower is not None and upper is not None:
            merged_path.append((feature, "in", (lower, upper)))
        elif lower is not None:
            merged_path.append((feature, ">", lower))
        elif upper is not None:
            merged_path.append((feature, "<=", upper))
    return merged_path


def rename_binary_conditions(path):
    renamed_path = []
    for feature, op, threshold in path:
        if threshold == 0.5:
            if op == "<=":
                renamed_path.append((feature, "=", 0))
            elif op == ">":
                renamed_path.append((feature, "=", 1))
        else:
            renamed_path.append((feature, op, threshold))
    return renamed_path


def export_tree_graphviz(tree, feature_names, class_names, file_name):
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
    graph.render(file_name)
    return graph


def calculate_path_similarity(binary_path, prob_path):
    shared_features = set(f for f, op, t in binary_path) & set(
        f for f, op, t in prob_path)
    total_similarity = 0
    total_shared_nodes = 0

    for feature in shared_features:
        binary_thresholds = [t for f, op, t in binary_path if f == feature]
        prob_thresholds = [t for f, op, t in prob_path if f == feature]

        for bt in binary_thresholds:
            for pt in prob_thresholds:
                if isinstance(bt, tuple) and isinstance(pt, tuple):
                    if bt[0] <= pt[0] or bt[1] >= pt[1]:
                        similarity = 1 - abs(bt[0] - pt[1])
                    else:
                        similarity = 1 - abs(bt[1] - pt[0])
                elif isinstance(pt, tuple):
                    if bt <= pt[0]:
                        similarity = 1 - abs(bt - pt[0])
                    else:
                        similarity = 1 - abs(bt - pt[1])
                elif isinstance(bt, tuple):
                    if pt <= bt[0]:
                        similarity = 1 - abs(pt - bt[0])
                    else:
                        similarity = 1 - abs(pt - bt[1])
                else:
                    similarity = 1 - abs(bt - pt)

                total_similarity += similarity
                total_shared_nodes += 1

    return total_similarity / total_shared_nodes if total_shared_nodes > 0 else 0


def print_loaded_matching_pairs(loaded_matching_pairs, class_names):
    if not loaded_matching_pairs:
        print("No matching pairs found.")
        return

    print(f"\nMatching paths and similarities for binary paths = 1:")
    for entry in loaded_matching_pairs["binary_paths_1"]:
        binary_path = entry["binary_path"]
        binary_count = entry["binary_count"]
        binary_classification = entry["binary_classification"]
        print(
            f"Binary Path: IF {' AND '.join([f'{feature} {op} {threshold}' for feature, op, threshold in binary_path])} -> | Count: {binary_count} | Classification: {binary_classification}")

        for prob_path, prob_count, prob_class, similarity in entry[
            "prob_paths"]:
            print(
                f"  Prob Path: IF {' AND '.join([f'{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]' if op == 'in' else f'{feature} {op} {threshold:.2f}' for feature, op, threshold in prob_path])} -> | Count: {prob_count} | Classification: {class_names[prob_class]}")
            print(f"  Similarity: {similarity:.4f}")

    print(f"\nMatching paths and similarities for binary paths = 0:")
    for entry in loaded_matching_pairs["binary_paths_0"]:
        binary_path = entry["binary_path"]
        binary_count = entry["binary_count"]
        binary_classification = entry["binary_classification"]
        print(
            f"Binary Path: IF {' AND '.join([f'{feature} {op} {threshold}' for feature, op, threshold in binary_path])} -> | Count: {binary_count} | Classification: {binary_classification}")

        for prob_path, prob_count, prob_class, similarity in entry[
            "prob_paths"]:
            print(
                f"  Prob Path: IF {' AND '.join([f'{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]' if op == 'in' else f'{feature} {op} {threshold:.2f}' for feature, op, threshold in prob_path])} -> | Count: {prob_count} | Classification: {class_names[prob_class]}")
            print(f"  Similarity: {similarity:.4f}")


def find_closest_paths_to_0_and_1(classifications_prob, counts_prob,
                                  pruned_branches_prob, target_feature):
    # Find the paths in the probabilistic tree where the threshold of feature "petal width (cm)" is closest to 1 and 0
    target_feature = target_feature
    closest_paths_prob_1 = []
    closest_threshold_diff_1 = float('inf')
    closest_paths_prob_0 = []
    closest_threshold_diff_0 = float('inf')
    for path, count, classification in zip(pruned_branches_prob, counts_prob,
                                           classifications_prob):
        for feature, op, threshold in path:
            if feature == target_feature:
                if isinstance(threshold, tuple):
                    threshold_diff_1 = min(abs(threshold[0] - 1),
                                           abs(threshold[1] - 1))
                    threshold_diff_0 = min(abs(threshold[0]), abs(threshold[1]))
                    if abs(threshold[1] - 1) <= closest_threshold_diff_1:
                        if abs(threshold[1] - 1) < closest_threshold_diff_1:
                            closest_paths_prob_1 = []
                            closest_threshold_diff_1 = abs(threshold[1] - 1)
                        closest_paths_prob_1.append(
                            (path, count, classification))
                    if abs(threshold[0] - 1) <= closest_threshold_diff_1:
                        if abs(threshold[0] - 1) < closest_threshold_diff_1:
                            closest_paths_prob_1 = []
                            closest_threshold_diff_1 = abs(threshold[0] - 1)
                        closest_paths_prob_1.append(
                            (path, count, classification))
                    if abs(threshold[0]) <= closest_threshold_diff_0:
                        if abs(threshold[0]) < closest_threshold_diff_0:
                            closest_paths_prob_0 = []
                            closest_threshold_diff_0 = abs(threshold[0])
                        closest_paths_prob_0.append(
                            (path, count, classification))
                    if abs(threshold[1]) <= closest_threshold_diff_0:
                        if abs(threshold[1]) < closest_threshold_diff_0:
                            closest_paths_prob_0 = []
                            closest_threshold_diff_0 = abs(threshold[1])
                        closest_paths_prob_0.append(
                            (path, count, classification))
                else:
                    threshold_diff_1 = abs(threshold - 1)
                    threshold_diff_0 = abs(threshold)
                    if threshold_diff_1 <= closest_threshold_diff_1:
                        if threshold_diff_1 < closest_threshold_diff_1:
                            closest_paths_prob_1 = []
                            closest_threshold_diff_1 = threshold_diff_1
                        closest_paths_prob_1.append(
                            (path, count, classification))
                    if threshold_diff_0 <= closest_threshold_diff_0:
                        if threshold_diff_0 < closest_threshold_diff_0:
                            closest_paths_prob_0 = []
                            closest_threshold_diff_0 = threshold_diff_0
                        closest_paths_prob_0.append(
                            (path, count, classification))
    return closest_paths_prob_0, closest_paths_prob_1, target_feature


def find_best_trees(X, y, num_trees = 1500, threshold = 0.02, min_performance = 0.9,
                    min_samples_leaf=1):

    # Number of different random states to visualize
    num_trees = num_trees
    performance_threshold = threshold  # Performance range (e.g., within 2% of the best performance)
    min_performance = min_performance  # Minimum performance to consider (e.g., accuracy of 90%)
    # First pass to collect all scores

    scores = []
    classifiers = []
    for i in range(num_trees):
        random_state = i
        clf = DecisionTreeClassifier(random_state=random_state,
                                     min_samples_leaf=min_samples_leaf)
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
        if
        best_score - performance_threshold <= mean_score <= best_score + performance_threshold and mean_score >= min_performance
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
    # for clf, random_state, mean_score in unique_trees:
    #     title = f"tree_random_state_{random_state}_score_{mean_score:.4f}"
    #     graph = export_tree_graphviz(clf, feature_names, class_names, title)
        # graph.view()

    return unique_trees


def calculate_feature_similarity(tree1, tree2):
    # Extract feature importances
    features1 = tree1.feature_importances_
    features2 = tree2.feature_importances_

    # Compute dot product similarity
    similarity = np.dot(features1, features2)

    return similarity


def find_highest_similarity_pair(trees_list1, trees_list2, feature_names):
    max_similarity = -1
    best_pair = (None, None)

    for tree1 in trees_list1:
        for tree2 in trees_list2:
            similarity = calculate_feature_similarity(tree1, tree2)
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (tree1, tree2)

    # find the feature with the highest importance in the best pair
    best_pair_features = []
    for tree in best_pair:
        best_pair_features.append(tree.feature_importances_.argmax())

    # raise an error if the best feature is not the same in both trees
    if best_pair_features[0] != best_pair_features[1]:
        raise ValueError('The best feature is not the same in both trees')

    # print the feature importances in the two trees
    print(f"Feature importances for the best pair of trees:")
    print(f"Tree 1: {best_pair[0].feature_importances_}")
    print(f"Tree 2: {best_pair[1].feature_importances_}")

    print(f"Best feature: {feature_names[best_pair_features[0]]}")
    best_feature_name = feature_names[best_pair_features[0]]

    return best_pair, max_similarity, best_feature_name


def print_tree_splits(tree, feature_names, class_names):
    def traverse(node, depth):
        indent = "  " * depth
        if tree.children_left[node] == _tree.TREE_LEAF:
            class_name = class_names[np.argmax(tree.value[node])]
            print(f"{indent}Leaf node: Predicts class {class_name}")
        else:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            print(f"{indent}Node: If {feature} <= {threshold:.20f}")
            traverse(tree.children_left[node], depth + 1)
            print(f"{indent}Else {feature} > {threshold:.20f}")
            traverse(tree.children_right[node], depth + 1)

    traverse(0, 0)


def weighted_node_count(tree, X_train):
    """Weighted node count by example"""
    leaf_indices = tree.apply(X_train)
    leaf_counts = np.bincount(leaf_indices)
    leaf_i = np.arange(tree.tree_.node_count)
    node_count = np.dot(leaf_i, leaf_counts) / float(X_train.shape[0])
    return node_count