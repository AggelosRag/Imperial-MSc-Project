def extract_features_from_splits(tree):
    feature_indices = set()

    def recurse(node):
        if tree.feature[node] != -2:  # -2 indicates a leaf node
            feature_indices.add(tree.feature[node])
            recurse(tree.children_left[node])
            recurse(tree.children_right[node])

    recurse(0)
    return sorted(list(feature_indices))


# Example usage:
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

# Extract features used in the splits
used_features = extract_features_from_splits(clf.tree_)
used_feature_names = [feature_names[i] for i in used_features]

print("feature names:", feature_names)
print("Used feature indices:", used_features)
print("Used feature names:", used_feature_names)
