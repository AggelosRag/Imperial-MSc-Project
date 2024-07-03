from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def calculate_feature_similarity(tree1, tree2):
    # Extract feature importances
    features1 = tree1.feature_importances_
    features2 = tree2.feature_importances_

    # Compute dot product similarity
    similarity = np.dot(features1, features2)

    return similarity


def find_highest_similarity_pair(trees_list1, trees_list2):
    max_similarity = -1
    best_pair = (None, None)

    for tree1 in trees_list1:
        for tree2 in trees_list2:
            similarity = calculate_feature_similarity(tree1, tree2)
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (tree1, tree2)

    return best_pair, max_similarity


def main():

    data = load_iris()
    X_binary = (data.data > np.median(data.data, axis=0)).astype(int)
    X_prob = data.data / data.data.max(axis=0)
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Assuming you have two lists of trees
    trees_list1 = [DecisionTreeClassifier(random_state=i).fit(X_binary, y) for i
                   in range(10)]
    trees_list2 = [DecisionTreeClassifier(random_state=i + 10).fit(X_prob, y)
                   for i in range(10)]

    # Find the pair of trees with the highest feature similarity
    best_pair, max_similarity = find_highest_similarity_pair(trees_list1,
                                                             trees_list2)

    print(f"Highest similarity: {max_similarity:.4f}")
    print(f"Best pair of trees found with highest feature similarity")


if __name__ == "__main__":
    main()
