import numpy as np
import pandas as pd


class Node():
    """
        Class for defining each node of the Decision Tree.
    """
    def __init__(self, attr = None, pred = None, class_label = None) -> None:
        self.attr = attr
        self.children = None
        self.isLeaf = False
        self.pred = pred 
        self.class_label = class_label


class DecisionTreeClassifierID3():
    """
        Class for implementing Decision Tree Classifier using ID3 (Iterative Dichotomiser 3) Algorithm.
    """

    def __init__(self):
        self.root = None

    def isBalanced(self, df):
        """
            Used to check if all tuples belong to a single class
            : param y: array, label or true values
            : return: boolean, True if all tuples belong to a single class, False otherwise.
        """
        return len(list(df.value_counts())) == 1

    def getEntropy(self, total, df):
        """
            Used to calculate entropy for a particular class value of a column
            : param total: int, total number of row/tuples/training examples
            : param df: array, column
            : return: int
        """
        labels = sorted(df.value_counts().to_dict().items())
        entropy = 0
        for label in labels:
            f = (label[1] / total)
            entropy -= f * np.log(f)
        return entropy

    def gain(self, column, y):
        """
            Used to calculate gain for a column
            : param column: array, column
            : param y: array, label or true values
            : return: int, gain for the column
        """
        total = len(column)
        labels = sorted(y.value_counts().to_dict().items())
        fp = (labels[0][1] / total)
        fn = (labels[1][1] / total)
        total_entropy = - (fp * np.log(fp)) - (fn * np.log(fn))
        g = total_entropy
        concat_df = pd.concat([column, y], axis=1)
        df_dict = {g: d['label']
                   for g, d in concat_df.groupby(by=[concat_df.columns[0]])}
        for key, value in df_dict.items():
            g -= (len(value) / total) * self.getEntropy(key, total, value)
        return g

    def getMaxGain(self, X, y):
        """
            Used to find the attribute which provides maximum gain. 
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: tuple, tuple of attribute name/column name and entropy value
        """
        cols = X.columns
        gain_dict = {}
        for col in cols:
            a = X[col]
            gain_dict[col] = self.gain(a, y)

        return sorted(gain_dict.items(), key=lambda x: x[1], reverse=True)[0]

    def buildTree(self, X, y, attr_classes, class_val=None):
        """
            Used to build the decision tree.
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : param attr_classes: dict, dictionary of list of distinct classes for each column
            : param class_val: string, distinct class, classification is based on
            : return: Node, a node for the tree
        """
        root = Node()
        if self.isBalanced(y):
            root.isLeaf = True
            root.pred = y.iloc[0]
        elif X is None:
            root.isLeaf = True
            root.pred = y.mode()
        else:
            maxGain = self.getMaxGain(X, y)
            maxGainCol = maxGain[0]
            pred = y.mode()[0]
            attr_list = attr_classes[maxGainCol].copy()
            concat_df = pd.concat([X, y], axis=1)
            df_dict = {g: d for g, d in concat_df.groupby(by=[maxGainCol])}
            root.attr = maxGainCol
            root.children = []
            for key, value in df_dict.items():
                attr_list.remove(key)
                new_X = value.drop(maxGainCol, axis=1).iloc[:, :-1]
                new_y = value.iloc[:, -1]
                root.children.append(self.buildTree(
                    new_X, new_y, attr_classes, key))
            if len(attr_list) > 0:
                root.pred = pred
        root.class_label = class_val
        return root

    def printTree(self, root, num_spaces=0):
        """
            Used to print the decision tree.
            : param root: Node, node of the decision tree
            : param num_spaces: int, number of spaces to be printed
            : return: None
        """
        print("\t" * num_spaces, end="")
        print(root.class_label, "->", end=" ")
        if root.children is None:
            print(root.pred)
        else:
            print(root.attr)
            for child in root.children:
                self.printTree(child, num_spaces + 1)

    def train(self, X, y):
        """
            Used to train the Decision Tree Classifier
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: None
        """
        attr_classes = {}
        cols = X.columns
        for col in cols:
            attr_classes[col] = list(X[col].value_counts().keys())

        self.root = self.buildTree(X, y, attr_classes)

    def predict_one_example(self, X, root):
        """
            Used to predict the value of y for a single example.
            : param X: tuple, one single data entry
            : param root: Node, node of the decision tree
            : return: array, predicted values
        """
        if root.isLeaf:
            return root.pred
        col = root.attr
        val = X[col]
        next_root = [x for x in root.children if x.class_label == val]
        if len(next_root) == 0:
            return root.pred
        return self.predict_one_example(X, next_root[0])

    def predict(self, X):
        """
            Used to predict the value of y
            : param X: 2D array, matrix of features, with each row being a data entry
            : return: array, predicted values
        """
        pred_y = []
        for i in range(len(X)):
            pred_y.append(self.predict_one_example(
                X.iloc[i, :], self.root))

        return pred_y
