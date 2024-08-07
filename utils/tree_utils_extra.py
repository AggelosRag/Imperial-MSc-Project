import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
import copy


DEFAULT_FEATURE = -2
DEFAULT_IMPURITY = 0
DEFAULT_THRESHOLD = -2
DEFAULT_LEAF = -1

def gen_blob_map(children_left, children_right):
    unique_node_i = np.sort(
        list(
            set(np.concatenate((children_left, children_right))) -
            set([DEFAULT_LEAF])
        )
    )
    blob_map = dict(zip(unique_node_i, [k+1 for k in range(unique_node_i.size)]))
    blob_map[DEFAULT_LEAF] = DEFAULT_LEAF
    return blob_map


def prune_order(tree):
    r"""Sort nodes by depth and ignore all leaves since
    pruning a leaf node is equivalent to adding it back.
    """
    node_depth = np.zeros(shape=tree.tree_.node_count)
    is_leaves = np.zeros(shape=tree.tree_.node_count, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]):
            stack.append((tree.tree_.children_left[node_id], parent_depth + 1))
            stack.append((tree.tree_.children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    # get non-leaves + sort by depth
    ii = np.where(~is_leaves)[0]
    prune_order = ii[np.argsort(node_depth[ii])[::-1]].tolist()
    return prune_order

def reduced_error_prune(tree, X_validation, y_validation):
    r"""Reduced Error Pruning (simplest one)
    https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29
    Arguments
    ---------
    tree: trained scikit-learn Decision Tree instance
    X_validation: validation set inputs
    y_validation: validation set outputs
    Starting a leaves, each node is replaced with most
    popular class. If prediction accuracy is not affected,
    keep the change.
    """
    base_tree = copy.deepcopy(tree)
    base_score = base_tree.score(X_validation, y_validation)
    node_order = prune_order(base_tree)
    num_nodes_to_prune = len(node_order)
    blob_map = gen_blob_map(base_tree.tree_.children_left,
                            base_tree.tree_.children_right)

    for i in range(num_nodes_to_prune):
        node_i = node_order[i]
        new_tree, blob_map = replace_sub_tree_with_leaf(base_tree, node_i)
        new_score = new_tree.score(X_validation, y_validation)

        if new_score >= base_score:
            base_tree = copy.deepcopy(new_tree)
            base_score = new_score
            # node order changes as blobs get renamed
            node_order = [blob_map[x] if x in blob_map else x for x in node_order]

    return base_tree

def replace_sub_tree_with_leaf(tree, i):
    """Create a new tree that is identical to tree except the sub-tree
    rooted at (i) is replaced with a node that returns the majority class.
    """
    tree2 = copy.deepcopy(tree)  # don't change original tree
    # we are going to replace the current node with a leaf node
    # so we can ignore all its children
    black_list = (
        _dfs_sklearn_tree(tree2,
                          tree2.tree_.children_left[i],
                          order='pre') +
        _dfs_sklearn_tree(tree2,
                          tree2.tree_.children_right[i],
                          order='pre')
    )
    # remaining nodes must be preserved
    white_list = list(set(range(tree2.tree_.capacity)) - set(black_list))
    cur_node_i = white_list.index(i)
    # store left 1/2 of tree and replace current node with -1 (no left children)
    children_left = tree2.tree_.children_left[white_list]
    children_left[cur_node_i] = DEFAULT_LEAF
    # store right 1/2 of tree and replace current node with -1 (no right children)
    children_right = tree2.tree_.children_right[white_list]
    children_right[cur_node_i] = DEFAULT_LEAF
    # store features and replace current node with -2
    feature = tree2.tree_.feature[white_list]
    feature[cur_node_i] = DEFAULT_FEATURE
    # store impurities and replace current node with 0
    impurity = tree2.tree_.impurity[white_list]
    impurity[cur_node_i] = DEFAULT_IMPURITY
    # store values in array
    sub_tree_values = tree2.tree_.value[i]
    # do the same for node_samples and weighted_node_samples
    n_node_samples = tree2.tree_.n_node_samples[white_list]
    n_node_samples[cur_node_i] = max(np.sum(tree2.tree_.value[i], axis=1))
    weighted_n_node_samples = tree2.tree_.weighted_n_node_samples[white_list]
    weighted_n_node_samples[cur_node_i] = n_node_samples[cur_node_i]
    # store thresholds and replace current node with -1
    threshold = tree2.tree_.threshold[white_list]
    threshold[cur_node_i] = DEFAULT_THRESHOLD
    # node_count = remaining number of nodes
    node_count = len(white_list)
    capacity = node_count
    n_classes = max(tree2.tree_.n_classes)
    n_output = len(tree2.tree_.n_classes)
    # replace value with max of new leaf's values (aka make a leaf)
    value = tree2.tree_.value[white_list, :, :]
    _value = np.zeros((n_output, n_classes))
    for ii in range(n_output):
        _value[
            ii, np.argmax(sub_tree_values, axis=1)[ii]
        ] = np.sum(tree2.tree_.value[i][ii])
    value[cur_node_i, :, :] = _value
    # we need to remap the nodes
    index_remap = gen_blob_map(children_left, children_right)
    # set tree objects!
    tree2.tree_.node_count = node_count
    tree2.tree_.capacity = capacity
    delta = value[cur_node_i, :, :] - tree2.tree_.value[i, :, :]

    for ii in range(node_count):
        tree2.tree_.children_left[ii] = index_remap[children_left[ii]]
        tree2.tree_.children_right[ii] = index_remap[children_right[ii]]
        tree2.tree_.feature[ii] = feature[ii]
        tree2.tree_.impurity[ii] = impurity[ii]
        tree2.tree_.n_node_samples[ii] = n_node_samples[ii]
        tree2.tree_.weighted_n_node_samples[ii] = weighted_n_node_samples[ii]
        tree2.tree_.threshold[ii] = threshold[ii]
        tree2.tree_.value[ii, :, :] = value[ii, :, :]
    update_node_value(tree2, delta, i)
    return tree2, index_remap

def _dfs_sklearn_tree(tree, i, order='post'):
    """Depth first search across scikit-learn arrays
    """
    left_order = []
    right_order = []
    root = []
    if tree.tree_.children_left[i] != -1:
        left_order = _dfs_sklearn_tree(tree, tree.tree_.children_left[i], order=order)
    if tree.tree_.children_right[i] != -1:
        right_order = _dfs_sklearn_tree(tree, tree.tree_.children_right[i], order=order)
    if i != -1:
        root.append(i)

    if order == 'pre':
        return root + left_order + right_order
    elif order == 'in':
        return left_order + root + right_order
    elif order == 'post':
        return left_order + right_order + root
    else:
        raise Exception('unknown order "{}" provided')

def update_node_value(tree, delta, node_i):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    parent_left = np.where(children_left == node_i)[0]
    parent_right = np.where(children_right == node_i)[0]

    if len(parent_left) > 0:
        tree.tree_.value[parent_left] += delta
        update_node_value(tree, delta, parent_left)

    if len(parent_right) > 0:
        tree.tree_.value[parent_right] += delta
        update_node_value(tree, delta, parent_right)


def get_path_length(data, predictions, strategy='tree',
                    n_trees=100, min_samples_leaf=1, size_sum=True):
    r"""Compute the average options length over the training dataset.

    @param data: torch.Tensor
                 size: n_data x n_input_features
    @param predictions: torch.Tensor
                        size: n_data x n_output_features
                        (these are floats, not rounded)
    @param strategy: string [default: tree]
                     tree|forest
    @param min_samples_leaf: integer [default: 1]
                             number of samples to define a leaf node
    @param n_trees: integer [default: 100]
                    number of trees in a random forest
                    (this only does something if strategy='forest')
    @param size_sum: boolean [default: True]

    @return: float
             average path lengths from a decision tree
    """
    if len(data) == 0:  # given empty inputs
        return 0

    data = data.cpu().data.numpy()
    predictions = predictions.cpu().data.numpy()
    predictions = np.rint(predictions).astype(int)

    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    path_length = []
    for i in range(predictions.shape[1]):
        if strategy == 'tree':
            clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

        clf.fit(data, predictions[:, i])
        path_length.append(average_path_length(clf, data))

    path_length = np.array(path_length)
    if size_sum:
        path_length = np.sum(path_length)

    return path_length

def average_path_length(tree, X):
    r"""Compute average path length: cost of simulating the average
    example; this is used in the objective function.

    @param tree: DecisionTreeClassifier instance
    @param X: NumPy array (D x N)
              D := number of dimensions
              N := number of examples
    @return path_length: float
                         average path length
    """
    leaf_indices = tree.apply(X)
    leaf_counts = np.bincount(leaf_indices)
    leaf_depths = get_node_depths(tree.tree_)
    path_length = np.dot(leaf_depths, leaf_counts) / float(X.shape[0])

    return path_length


def get_node_depths(tree):
    r"""Get the node depths of the decision tree

    @param tree: DecisionTreeClassifier instance
    @return depths: np.array

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_node_depths(d.tree_)
        array([0, 1, 1, 2, 2])
    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths)
    return np.array(depths)