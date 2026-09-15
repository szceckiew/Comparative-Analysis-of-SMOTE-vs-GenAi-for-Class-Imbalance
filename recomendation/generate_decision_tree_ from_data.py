import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np

feature_names = ['Imbalance_Ratio', 'Linearity_AUC']
class_names = ['Advanced_GAN', 'SMOTE', 'Simple_GAN']
n_nodes = 5

children_left = np.array([1, -1, 3, -1, -1], dtype=np.int64)
children_right = np.array([2, -1, 4, -1, -1], dtype=np.int64)
feature = np.array([0, -2, 1, -2, -2], dtype=np.int64)
threshold = np.array([0.83, -2.0, 0.977, -2.0, -2.0], dtype=np.float64)
n_node_samples = np.array([15, 2, 13, 10, 3], dtype=np.int64)
impurity = np.array([0.667, 0.0, 0.612, 0.577, 0.0], dtype=np.float64)

value = np.array([
    [[5.0, 5.0, 5.0]],
    [[3.333, 0.0, 0.0]],
    [[1.667, 5.0, 5.0]],
    [[1.667, 2.0, 5.0]],
    [[0.0, 3.0, 0.0]]
], dtype=np.float64)

weighted_n_node_samples = np.sum(value, axis=2).flatten()


class FixedTree:
    def __init__(self, n_nodes):
        self.node_count = n_nodes
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.n_node_samples = n_node_samples
        self.impurity = impurity
        self.value = value
        self.weighted_n_node_samples = weighted_n_node_samples

        self.n_outputs = 1
        self.n_classes = np.array([3], dtype=np.int64)
        self.max_n_classes = 3


clf = DecisionTreeClassifier(random_state=42)
dummy_X = [[0, 0], [0, 0], [0, 0]]
dummy_y = [0, 1, 2]
clf.fit(dummy_X, dummy_y)

clf.tree_ = FixedTree(n_nodes)

plt.figure(figsize=(12, 8), dpi=100)
plot_tree(clf,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Decision Tree: SMOTE vs Simple GAN vs Advanced GAN", fontsize=12)

plt.savefig("decision_tree_fixed.png")
plt.show()