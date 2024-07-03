import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from scipy.optimize import linear_sum_assignment

def find_split_points(X, y):
    unique_values = np.unique(X)
    split_points = []
    for value in unique_values:
        if value != unique_values[-1]:
            split_points.append(value)
    return split_points

def adjust_thresholds(tree, X):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # Not a leaf node
            feature_index = feature[i]
            unique_values = np.unique(X[:, feature_index])
            for value in unique_values:
                if threshold[i] <= value:
                    threshold[i] = value
                    break

def extract_branches(tree, feature_names, binary_input=False):
    branches = []
    def traverse(node, path):
        if tree.children_left[node] == tree.children_right[node]:  # leaf
            branches.append(path)
            return
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        if binary_input:
            threshold = 1 if threshold >= 0.5 else 0  # Adjusting thresholds for binary inputs
        left_path = path + [(feature_names[feature], "<=", threshold)]
        right_path = path + [(feature_names[feature], ">", threshold)]
        traverse(tree.children_left[node], left_path)
        traverse(tree.children_right[node], right_path)

    traverse(0, [])
    return branches

def print_branches(branches):
    for branch in branches:
        print(" -> ".join([f"{feature} {op} {threshold:.2f}" for feature, op, threshold in branch]))

def normalize_threshold(threshold, feature_index, X):
    min_val = X[:, feature_index].min()
    max_val = X[:, feature_index].max()
    return (threshold - min_val) / (max_val - min_val)

def branch_similarity(branch1, branch2, X_binary, X_prob):
    match_count = 0
    for (feature1, op1, threshold1), (feature2, op2, threshold2) in zip(branch1, branch2):
        if feature1 == feature2 and op1 == op2:
            norm_threshold1 = normalize_threshold(threshold1, feature_names.index(feature1), X_binary)
            norm_threshold2 = normalize_threshold(threshold2, feature_names.index(feature1), X_prob)
            if np.isclose(norm_threshold1, norm_threshold2):
                match_count += 1
    return match_count / max(len(branch1), len(branch2))

def compute_similarity_matrix(branches1, branches2, X_binary, X_prob):
    similarity_matrix = np.zeros((len(branches1), len(branches2)))
    for i, branch1 in enumerate(branches1):
        for j, branch2 in enumerate(branches2):
            similarity_matrix[i, j] = branch_similarity(branch1, branch2, X_binary, X_prob)
    return similarity_matrix

def optimal_branch_matching(similarity_matrix):
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # maximize similarity
    return row_ind, col_ind, similarity_matrix[row_ind, col_ind].sum()

# Load dataset
data = load_iris()
X_binary = (data.data > np.median(data.data, axis=0)).astype(int)
X_prob = data.data / data.data.max(axis=0)
y = data.target
feature_names = data.feature_names

# Train a Decision Tree model
clf_binary = DecisionTreeClassifier(random_state=42)
clf_binary.fit(X_binary, y)
adjust_thresholds(clf_binary.tree_, X_binary)
branches_binary = extract_branches(clf_binary.tree_, feature_names, binary_input=True)
print("Binary Tree Branches:")
print_branches(branches_binary)

clf_prob = DecisionTreeClassifier(random_state=42)
clf_prob.fit(X_prob, y)
adjust_thresholds(clf_prob.tree_, X_prob)
branches_prob = extract_branches(clf_prob.tree_, feature_names, binary_input=False)
print("\nProbability Tree Branches:")
print_branches(branches_prob)

# Example usage
similarity_matrix = compute_similarity_matrix(branches_binary, branches_prob, X_binary, X_prob)
row_ind, col_ind, total_similarity = optimal_branch_matching(similarity_matrix)

print("\nOptimal matching:")
for i, j in zip(row_ind, col_ind):
    print(f"Branch {i} (Binary Tree) <-> Branch {j} (Probability Tree) with similarity {similarity_matrix[i, j]:.2f}")
print(f"Total similarity: {total_similarity:.2f}")
