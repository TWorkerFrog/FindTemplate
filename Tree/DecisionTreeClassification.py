# Классификация с помощью дерева решений
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=columns, skipinitialspace=True, na_values='?')
test_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", names=columns, skipinitialspace=True, na_values='?', skiprows=1)
test_df['income'] = test_df['income'].str.replace('.', '', regex=False) # Убираем точки в конце (">50K.")

full_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Всего записей: {len(full_df)}")
print(f"Распределение income:\n{full_df['income'].value_counts()}\n")

# Предобработка данных
df = full_df.dropna().copy()

# Удаляем неинформативные признаки: fnlwgt (вес записи) и education (дублирует education-num)
df = df.drop(['fnlwgt', 'education'], axis=1)

for col in df.select_dtypes(include='object').columns:
    if col != 'income':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Приводим str -> bool
df['income'] = (df['income'] == '>50K').astype(int)
print(f"После предобработки: {len(df)} записей, признаков: {df.shape[1] - 1}\n")

# Эксперимент
criteria = ['entropy', 'gini']
train_ratios = [0.6, 0.7, 0.8, 0.9]
results = []

for criterion in criteria:
    for train_ratio in train_ratios:
        X = df.drop('income', axis=1)
        y = df['income']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42, stratify=y)
        
        # Обучение дерева решений
        clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Результат и метрики
        results.append({
            'criterion': criterion,
            'train_ratio': f"{round(train_ratio*100)}:{round((1-train_ratio)*100)}",
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'tree_depth': clf.get_depth()
        })
        
        print(f"[{criterion.upper()}] {int(train_ratio*100)}% | "
              f"Acc: {results[-1]['accuracy']:.4f} | Prec: {results[-1]['precision']:.4f} | "
              f"Rec: {results[-1]['recall']:.4f} | F1: {results[-1]['f1']:.4f} | "
              f"Глубина: {results[-1]['tree_depth']}")


# Сводная таблица
results_df = pd.DataFrame(results)
print("\nСводная таблица результатов:")
print(results_df.to_string(index=False))

# Визуализация дерева
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

clf_viz = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=4)
clf_viz.fit(X_train, y_train)

plt.figure(figsize=(25, 15))
plot_tree(
    clf_viz,
    feature_names=X.columns.tolist(),
    class_names=['<=50K', '>50K'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Дерево решений для предсказания доходов населения в сравнении с 50к от 13 факторов", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_census.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nВизуализация дерева сохранена: decision_tree_census.png")

# Графики сравнения метрик
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Rmax_depthecall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    for criterion in criteria:
        subset = results_df[results_df['criterion'] == criterion]
        ax.plot(subset['train_ratio'], subset[metric], marker='o', label=criterion)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Соотношение обучающей и тестовой выборок')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сравнения метрик сохранён: metrics_comparison.png")

# Критерии оценки:
# Entropy(node) = −Σ pᵢ log₂(pᵢ) - из ID3
# Gain Ratio                     - из C4.5
# Gini(node) = 1 − Σ pᵢ²         - из CART

# Accuracy  = (TP + TN) / (TP + TN + FP + FN) - доля истинных ответов
# Precision = TP / (TP + FP) - сколько действительно истины из истинных в обучающей выборке
# Recall    = TP / (TP + FN) - сколько действительно истины из истинных в тестовой выборке
# F1        = 2 · Precision · Recall / (Precision + Recall) - ср. гармоническое Precision и Recall
# где TP, TN, FP, FN — истинно положительные, истинно отрицательные, ложноположительные и ложноотрицательные результаты для класса >50K