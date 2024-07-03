import importlib

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

from experimentation.tree_comparison_utils import compute_structural_similarity, \
    compare_feature_importances, compute_semantic_similarity, \
    extract_paths_and_counts, merge_conditions, rename_binary_conditions, \
    export_tree_graphviz, calculate_path_similarity, \
    print_loaded_matching_pairs, find_closest_paths_to_0_and_1, find_best_trees, \
    find_highest_similarity_pair, prune, print_tree_splits, \
    calculate_accuracy_per_path


def compare_two_trees(tree_binary, tree_prob, X, X_binary, X_prob, y_binary, y_prob, y, feature_names,
                      class_names, best_feature_name, output_path):

    branches_binary, counts_binary, classifications_binary, path_indices = extract_paths_and_counts(
        tree_binary.tree_, X_binary, y_binary, feature_names)
    # Calculate accuracy per path
    accuracies = calculate_accuracy_per_path(path_indices, y, classifications_binary)
    pruned_branches_binary = [merge_conditions(path) for path in
                              branches_binary]
    renamed_branches_binary = [rename_binary_conditions(path) for path in
                               pruned_branches_binary]
    print("Binary Tree Paths after Merging and Renaming:")
    for path, count, classification, indices, accuracy in zip(renamed_branches_binary,
                                                               counts_binary,
                                                               classifications_binary,
                                                               path_indices, accuracies):
        print(" -> ".join([
            f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold:.2f}"
            for feature, op, threshold in path]))

        #print(f"Data indices: {indices}")
        print(f"Count: {count}")
        print(f"Classification: {class_names[classification]}")
        print(f"Accuracy: {accuracy:.4f}\n")

    # Export the binary decision tree
    export_tree_graphviz(tree_binary, feature_names, class_names, output_path + "/trees/binary_tree")

    # Store variables into a pickle file
    with open(output_path + '/trees/tree_binary_data.pkl', 'wb') as f:
        pickle.dump({
            'X': X,
            'c': X_binary,
            'y': y,
            'tree': tree_binary,
            'paths': branches_binary,
            'path_counts': counts_binary,
            'path_classifications': classifications_binary,
            'path_indices': path_indices,
            'accuracies': accuracies
        }, f)

    branches_prob, counts_prob, classifications_prob, path_indices_prob = extract_paths_and_counts(
        tree_prob.tree_, X_prob, y_prob, feature_names)
    accuracies_prob = calculate_accuracy_per_path(path_indices_prob, y, classifications_prob)
    pruned_branches_prob = [merge_conditions(path) for path in branches_prob]
    print("\nProbability Tree Paths after Merging:")
    for path, count, classification, indices, accuracy in zip(pruned_branches_prob, counts_prob,
                                                               classifications_prob,
                                                               path_indices_prob, accuracies_prob):
        print(" -> ".join([
            f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold:.2f}"
            for feature, op, threshold in path]))

        #print(f"Data indices: {indices}")
        print(f"Count: {count}")
        print(f"Classification: {class_names[classification]}")
        print(f"Accuracy: {accuracy:.4f}\n")

    # Store variables into a pickle file
    with open(output_path + '/trees/tree_prob_data.pkl', 'wb') as f:
        pickle.dump({
            'X': X,
            'c': X_prob,
            'y': y,
            'tree': tree_prob,
            'paths': branches_prob,
            'path_counts': counts_prob,
            'path_classifications': classifications_prob,
            'path_indices': path_indices_prob,
            'accuracies': accuracies_prob
        }, f)

    # Export the probability decision tree
    export_tree_graphviz(tree_prob, feature_names, class_names, output_path + "/trees/probability_tree")

    # Find the paths in the probabilistic tree where the threshold of feature "petal width (cm)" is closest to 1 and 0
    closest_paths_prob_0, closest_paths_prob_1, target_feature = find_closest_paths_to_0_and_1(
        classifications_prob, counts_prob, pruned_branches_prob, best_feature_name)

    print(
        f"\nClosest paths in probabilistic tree for {target_feature} closest to 1:")
    for path, count, classification in closest_paths_prob_1:
        print(" -> ".join([
                              f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold:.2f}"
                              for feature, op, threshold in path]) +
              f" | Count: {count} | Classification: {class_names[classification]}")

    print(
        f"\nClosest paths in probabilistic tree for {target_feature} closest to 0:")
    for path, count, classification in closest_paths_prob_0:
        print(" -> ".join([
                              f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold:.2f}"
                              for feature, op, threshold in path]) +
              f" | Count: {count} | Classification: {class_names[classification]}")

    # Find the paths in the binary tree where the feature "petal width (cm)" is equal to 1
    matching_paths_binary_1 = []
    matching_counts_binary_1 = []
    matching_classifications_binary_1 = []

    for path, count, classification in zip(renamed_branches_binary,
                                           counts_binary,
                                           classifications_binary):
        for feature, op, threshold in path:
            if feature == target_feature and op == "=" and threshold == 1:
                matching_paths_binary_1.append(path)
                matching_counts_binary_1.append(count)
                matching_classifications_binary_1.append(classification)

    print(f"\nMatching paths in binary tree for {target_feature} = 1:")
    for path, count, classification in zip(matching_paths_binary_1,
                                           matching_counts_binary_1,
                                           matching_classifications_binary_1):
        print(" -> ".join([
                              f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold}"
                              for feature, op, threshold in path]) +
              f" | Count: {count} | Classification: {class_names[classification]}")

    # Find the paths in the binary tree where the feature "petal width (cm)" is equal to 0
    matching_paths_binary_0 = []
    matching_counts_binary_0 = []
    matching_classifications_binary_0 = []

    for path, count, classification in zip(renamed_branches_binary,
                                           counts_binary,
                                           classifications_binary):
        for feature, op, threshold in path:
            if feature == target_feature and op == "=" and threshold == 0:
                matching_paths_binary_0.append(path)
                matching_counts_binary_0.append(count)
                matching_classifications_binary_0.append(classification)

    print(f"\nMatching paths in binary tree for {target_feature} = 0:")
    for path, count, classification in zip(matching_paths_binary_0,
                                           matching_counts_binary_0,
                                           matching_classifications_binary_0):
        print(" -> ".join([
                              f"{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]" if op == "in" else f"{feature} {op} {threshold}"
                              for feature, op, threshold in path]) +
              f" | Count: {count} | Classification: {class_names[classification]}")

    # Dictionary to store all pairs information
    matching_pairs = {
        "binary_paths_1": [],
        "binary_paths_0": []
    }

    # Print matching paths, calculate similarity, and store information
    print(f"\nMatching paths and similarities for {target_feature} = 1:")
    for binary_path, binary_count, binary_class in zip(matching_paths_binary_1,
                                                       matching_counts_binary_1,
                                                       matching_classifications_binary_1):
        print(
            f"Binary Path: {' -> '.join([f'{feature} {op} {threshold}' for feature, op, threshold in binary_path])} | Count: {binary_count} | Classification: {class_names[binary_class]}")
        similarities = []
        for prob_path, prob_count, prob_class in closest_paths_prob_1:
            similarity = calculate_path_similarity(binary_path, prob_path)
            similarities.append((prob_path, prob_count, prob_class, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[3], reverse=True)

        # Store in dictionary
        matching_pairs["binary_paths_1"].append({
            "binary_path": binary_path,
            "binary_count": binary_count,
            "binary_classification": class_names[binary_class],
            "prob_paths": similarities
        })

        for prob_path, prob_count, prob_class, similarity in similarities:
            print(
                f"  Prob Path: {' -> '.join([f'{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]' if op == 'in' else f'{feature} {op} {threshold:.2f}' for feature, op, threshold in prob_path])} | Count: {prob_count} | Classification: {class_names[prob_class]}")
            print(f"  Similarity: {similarity:.4f}")

    print(f"\nMatching paths and similarities for {target_feature} = 0:")
    for binary_path, binary_count, binary_class in zip(matching_paths_binary_0,
                                                       matching_counts_binary_0,
                                                       matching_classifications_binary_0):
        print(
            f"Binary Path: {' -> '.join([f'{feature} {op} {threshold}' for feature, op, threshold in binary_path])} | Count: {binary_count} | Classification: {class_names[binary_class]}")
        similarities = []
        for prob_path, prob_count, prob_class in closest_paths_prob_0:
            similarity = calculate_path_similarity(binary_path, prob_path)
            similarities.append((prob_path, prob_count, prob_class, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[3], reverse=True)

        # Store in dictionary
        matching_pairs["binary_paths_0"].append({
            "binary_path": binary_path,
            "binary_count": binary_count,
            "binary_classification": class_names[binary_class],
            "prob_paths": similarities
        })

        for prob_path, prob_count, prob_class, similarity in similarities:
            print(
                f"  Prob Path: {' -> '.join([f'{feature} {op} [{threshold[0]:.2f}, {threshold[1]:.2f}]' if op == 'in' else f'{feature} {op} {threshold:.2f}' for feature, op, threshold in prob_path])} | Count: {prob_count} | Classification: {class_names[prob_class]}")
            print(f"  Similarity: {similarity:.4f}")

    # Save the dictionary to a file
    with open(output_path + '/trees/matching_pairs.pkl', 'wb') as f:
        pickle.dump(matching_pairs, f)

    # Load the dictionary from the file
    # with open(output_path + '/trees/matching_pairs.pkl', 'rb') as f:
    #     loaded_matching_pairs = pickle.load(f)

    # Print everything from the loaded dictionary
    #print_loaded_matching_pairs(loaded_matching_pairs, class_names)

    print("done")

def leakage_visualizer(X, X_binary, y_binary, y, X_prob, y_prob,
                       feature_names, class_names, output_path,
                        num_trees = 1500, threshold = 0.02,
                        min_performance = 0.5, min_samples_leaf=5):

    # load the ground truth concepts as X_binary
    # load the predictions of the NN as y_binary
    # load the predicted concepts as X_prob
    # load the predictions of the NN as y_prob

    # collect best trees for the ground truth tree
    unique_trees_binary = find_best_trees(X_binary, y_binary,
                                          num_trees=num_trees,
                                          threshold=threshold,
                                          min_performance=min_performance,
                                          min_samples_leaf=min_samples_leaf)

    # collect best trees for the probabilistic tree
    unique_trees_prob = find_best_trees(X_prob, y_prob,
                                        num_trees=num_trees,
                                        threshold=threshold,
                                        min_performance=min_performance,
                                        min_samples_leaf=min_samples_leaf)

    # stack the first elements of each tuple into a new list
    trees_list1 = [tree for tree, _, _ in unique_trees_binary]
    trees_list2 = [tree for tree, _, _ in unique_trees_prob]

    # Find the pair of trees with the highest feature similarity
    best_pair, max_similarity, best_feature_name = find_highest_similarity_pair(
        trees_list1, trees_list2,feature_names
    )

    print(f"Total feature similarity of the best pair: {max_similarity:.4f}")

    # Comparison of the two trees
    clf_binary = best_pair[0]
    clf_prob = best_pair[1]

    # prune the two trees
    prune(clf_binary.tree_)
    prune(clf_prob.tree_)

    # Print the splits of the tree
    print_tree_splits(clf_binary.tree_, feature_names, class_names)
    print_tree_splits(clf_prob.tree_, feature_names, class_names)

    # plot the two trees
    export_tree_graphviz(clf_binary, feature_names, class_names, output_path + '/binary_tree')
    export_tree_graphviz(clf_prob, feature_names, class_names, output_path + '/prob_tree')

    #structural_similarity = compute_structural_similarity(clf_binary, clf_prob)
    #semantic_similarity = compute_semantic_similarity(clf_binary, clf_prob, X)

    # print("Structural Similarity:", structural_similarity)
    # print("Semantic Similarity:", semantic_similarity)

    compare_two_trees(clf_binary, clf_prob, X, X_binary, X_prob, y_binary, y_prob, y,
                      feature_names, class_names, best_feature_name, output_path)

def perform_leakage_visualization(data_loader, arch, config):

    # load checkpoint from the leaky model
    checkpoint = torch.load(config['explainer']['path_to_leaky_model'])
    arch.model.load_state_dict(checkpoint['state_dict'])
    arch.model.eval()

    # make a forward pass to get the predictions using the predicted concepts
    with torch.no_grad():
        X = data_loader.dataset[:][0]
        y = data_loader.dataset[:][2]
        c = data_loader.dataset[:][1]
        c_pred = arch.model.mn_model.concept_predictor(X)
        y_pred = arch.model.mn_model.label_predictor(c_pred)

        if y_pred.shape[1] == 1:
            y_hat_pred_prob = torch.where(y_pred > 0.5, 1, 0).cpu()
        elif y_pred.shape[1] >= 3:
            y_hat_pred_prob = torch.argmax(y_pred, 1).cpu()
        else:
            raise ValueError('Invalid number of output classes')

        # calculate accuracy
        accuracy = torch.sum(y == y_hat_pred_prob).item() / len(y)
        print(f'Accuracy: {accuracy}')

    # load checkpoint from the ground truth model
    checkpoint = torch.load(config['explainer']['path_to_gt_model'])
    arch_module = importlib.import_module("architectures")
    module_name = config['arch']['gt_arch']
    arch_gt = getattr(arch_module, module_name)(config)
    arch_gt.model.load_state_dict(checkpoint['state_dict'])
    arch_gt.model.eval()

    # make a forward pass to get the predictions using the predicted concepts
    with torch.no_grad():
        y = data_loader.dataset[:][2]
        c = data_loader.dataset[:][1]
        y_pred = arch_gt.model(c)

        if y_pred.shape[1] == 1:
            y_hat_pred_binary = torch.where(y_pred > 0.5, 1, 0).cpu()
        elif y_pred.shape[1] >= 3:
            y_hat_pred_binary = torch.argmax(y_pred, 1).cpu()
        else:
            raise ValueError('Invalid number of output classes')

        # calculate accuracy
        accuracy = torch.sum(y == y_hat_pred_binary).item() / len(y)
        print(f'Accuracy: {accuracy}')

    leakage_visualizer(X = X,
                       X_binary = c,
                       y_binary = y_hat_pred_binary,
                       y = y,
                       X_prob = c_pred,
                       y_prob = y_hat_pred_prob,
                       feature_names = config['dataset']['feature_names'],
                       class_names = config['dataset']['class_names'],
                       output_path = str(config.log_dir),
                       num_trees = config['explainer']['num_trees'],
                       threshold = config['explainer']['threshold'],
                       min_performance = config['explainer']['min_performance'],
                       min_samples_leaf= config['explainer']['min_samples_leaf'])


if __name__ == "__main__":
    # Load dataset
    data = load_iris()
    X_binary = (data.data > np.median(data.data, axis=0)).astype(int)
    X_prob = data.data / data.data.max(axis=0)
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    leakage_visualizer(X = X_binary,
                       X_binary = X_binary,
                       y_binary = y,
                       y = y,
                       X_prob = X_prob,
                       y_prob = y,
                       feature_names = feature_names,
                       class_names = class_names,
                       output_path = '.',
                       num_trees = 200,
                       threshold = 0.02,
                       min_performance = 0.5,
                       min_samples_leaf=5)