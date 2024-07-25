import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  random_state=0)

# Train the decision tree classifier on the training set
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Get the leaf nodes for the validation set predictions
leaf_nodes = clf.apply(X_val)

# Count the number of samples in each leaf node for the validation set
node_counts = np.bincount(leaf_nodes, minlength=clf.tree_.node_count)


def export_tree_with_counts(clf, feature_names, class_names, node_counts,
                            filename):
    # Export the decision tree to DOT format
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        node_ids=True
    )

    # Modify the DOT data to include sample counts at each node
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        if 'label="' in line:
            parts = line.split('label="')
            before_label = parts[0]
            label_content = parts[1].split('"')[0]
            after_label = '"'.join(parts[1].split('"')[1:])

            node_id_str = before_label.split()[0]
            if node_id_str.isdigit():
                node_id = int(node_id_str)
                if node_id < len(node_counts):
                    label_content += f'\\nsamples = {node_counts[node_id]}'
            new_line = before_label + 'label="' + label_content + '"' + after_label
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Combine the lines back into a single string
    modified_dot_data = '\n'.join(new_lines)

    # Render the modified DOT data using Graphviz
    graph = graphviz.Source(modified_dot_data)
    graph.render(filename, format='png', cleanup=True)


def export_tree_with_basic_info(clf, feature_names, class_names, filename):
    # Export the decision tree to DOT format
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        node_ids=True
    )

    # Modify the DOT data to include only basic information
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        if 'label="' in line:
            parts = line.split('label="')
            before_label = parts[0]
            label_content = parts[1].split('"')[0]
            # Extract the basic info
            basic_info = label_content.split('\\n')[
                         0:3]  # Attribute, threshold, values, and class
            label_content = '\\n'.join(basic_info)
            after_label = '"'.join(parts[1].split('"')[1:])
            new_line = before_label + 'label="' + label_content + '"' + after_label
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Combine the lines back into a single string
    modified_dot_data = '\n'.join(new_lines)

    # Render the modified DOT data using Graphviz
    graph = graphviz.Source(modified_dot_data)
    graph.render(filename, format='png', cleanup=True)


# Export the trained tree with sample counts for the validation set
export_tree_with_counts(clf, iris.feature_names, iris.target_names, node_counts,
                        'decision_tree_with_counts')

# Export the tree showing only basic information for the validation set predictions
export_tree_with_basic_info(clf, iris.feature_names, iris.target_names,
                            'validation_tree_with_basic_info')
