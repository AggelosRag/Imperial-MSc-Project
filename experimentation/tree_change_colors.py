from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import re

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target
class_names = data.target_names

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Define a color map for the classes with specific color values
color_map = {
    0: "#FF9999",  # Light red for Class 0
    1: "#99FF99",  # Light green for Class 1
    2: "#9999FF"   # Light blue for Class 2
}

# Export the tree to dot format with filled=False and rounded=True
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=data.feature_names,
    class_names=class_names,
    filled=False,  # Disable automatic coloring
    rounded=True,  # Enable rounded corners
    special_characters=True
)

# Function to modify the exported dot file with colors
def modify_dot_with_colors(dot_data, color_map, clf):
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        match = re.match(r'(\d+) \[label=.*\]', line)
        if match:
            node_id = int(match.group(1))
            node_class = clf.tree_.value[node_id].argmax()
            color = color_map[node_class]
            # Add fillcolor and style to the node definition
            line = re.sub(r'(?<=>)\]', f', style="filled,rounded", fillcolor="{color}"]', line)
        new_lines.append(line)
    return '\n'.join(new_lines)

# Modify the dot data to include the specific colors
dot_data_with_colors = modify_dot_with_colors(dot_data, color_map, clf)

# Render the graph
graph = graphviz.Source(dot_data_with_colors)
graph.render("tree_with_specific_colors", format="png", view=True)
