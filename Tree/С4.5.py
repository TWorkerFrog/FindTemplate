import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y, y_left, y_right):
    n = len(y)
    if n == 0:
        return 0
    weighted = (len(y_left) / n) * entropy(y_left) + \
               (len(y_right) / n) * entropy(y_right)
    return entropy(y) - weighted


def split_info(y_left, y_right):
    n = len(y_left) + len(y_right)
    if n == 0:
        return 1
    p_left = len(y_left) / n
    p_right = len(y_right) / n
    info = 0
    for p in (p_left, p_right):
        if p > 0:
            info -= p * np.log2(p)
    return info if info > 0 else 1


def gain_ratio(y, y_left, y_right):
    return information_gain(y, y_left, y_right) / split_info(y_left, y_right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class C45Tree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=self._majority(y))

        best = self._best_split(X, y)
        if best is None:
            return Node(value=self._majority(y))

        feature, threshold = best
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or \
           right_mask.sum() < self.min_samples_leaf:
            return Node(value=self._majority(y))

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        best_ratio = -1
        best_split = None

        for feature in range(X.shape[1]):
            values = X[:, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 20:
                thresholds = np.quantile(values, np.linspace(0.05, 0.95, 20))

            for t in thresholds:
                left_mask = values <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                ratio = gain_ratio(y, y_left, y_right)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_split = (feature, t)

        return best_split

    def _majority(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=columns, skipinitialspace=True, na_values='?')
test_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    names=columns, skipinitialspace=True, na_values='?', skiprows=1)
test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

full_df = pd.concat([train_df, test_df], ignore_index=True)
df = full_df.dropna().copy()
df = df.drop(['fnlwgt', 'education'], axis=1)

for col in df.select_dtypes(include=['str', 'object']).columns:
    if col != 'income':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df['income'] = (df['income'] == '>50K').astype(int)

X = df.drop('income', axis=1).values
y = df['income'].values


train_ratios = [0.6, 0.7, 0.8, 0.9]
results = []

for ratio in train_ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=ratio, random_state=42, stratify=y)

    tree = C45Tree(max_depth=8, min_samples_split=20, min_samples_leaf=10)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    row = {
        'train_ratio': f"{int(ratio*100)}:{int((1-ratio)*100)}",
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
    }
    results.append(row)
    print(f"C4.5 | {row['train_ratio']} | Acc: {row['accuracy']:.4f} | "
          f"Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | "
          f"F1: {row['f1']:.4f}")

results_df = pd.DataFrame(results)
print("\nСводная таблица C4.5:")
print(results_df.to_string(index=False))

train_ratios = [0.6, 0.7, 0.8, 0.9]
results = []

for ratio in train_ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=ratio, random_state=42, stratify=y)

    models = {
        'C4.5 (Gain Ratio)': C45Tree(max_depth=8, min_samples_split=20, min_samples_leaf=10),
        'ID3 (Entropy)':     DecisionTreeClassifier(
                                criterion='entropy',
                                max_depth=8,
                                min_samples_split=20,
                                min_samples_leaf=10,
                                random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'model': name,
            'train_ratio': f"{int(ratio*100)}:{int((1-ratio)*100)}",
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
        })

results_df = pd.DataFrame(results)

print("\nСводная таблица:")
print(results_df.to_string(index=False))


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

colors = {
    'C4.5 (Gain Ratio)': '#4A90D9',
    'ID3 (Entropy)':     '#ED7D31',
}
markers = {
    'C4.5 (Gain Ratio)': 'o',
    'ID3 (Entropy)':     's',
}

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]

    for name in colors:
        subset = results_df[results_df['model'] == name]
        ax.plot(subset['train_ratio'], subset[metric],
                marker=markers[name], color=colors[name],
                linewidth=2, markersize=7, label=name)

        for x, yv in zip(subset['train_ratio'], subset[metric]):
            ax.annotate(f'{yv:.3f}', (x, yv),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Соотношение обучающей и тестовой выборок')
    ax.set_ylabel(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Сравнение критериев: C4.5 (Gain Ratio) vs ID3 (Entropy)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gainratio_vs_gain.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: gainratio_vs_gain.png")