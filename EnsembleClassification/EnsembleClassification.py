# Ансамблевая классификация: Bagging/Random Forest/Boosting
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных (Census Income)
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week',
           'native-country', 'income']

train_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=columns, skipinitialspace=True, na_values='?')
test_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", names=columns, skipinitialspace=True, na_values='?', skiprows=1)
test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

full_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Всего записей: {len(full_df)}")

# Предобработка
df = full_df.dropna().copy()
df = df.drop(['fnlwgt', 'education'], axis=1)

for col in df.select_dtypes(include='object').columns:
    if col != 'income':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df['income'] = (df['income'] == '>50K').astype(int)

X = df.drop('income', axis=1)
y = df['income']
print(f"После предобработки: {len(df)} записей, признаков: {X.shape[1]}\n")

# Фиксированное разбиение, как в задаче с деревом решений
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

n_estimators_list = list(range(50, 101, 10))  # 50, 60, ..., 100

# Результаты по техникам
results = {
    'Bagging': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Random Forest': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Boosting': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
}

for n in n_estimators_list:
    print(f"--- Участников: {n} ---")

    # Bagging: деревья на bootstrap-выборках
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n, random_state=42, n_jobs=-1
    )
    # Random Forest: деревья + случайные признаки
    rf = RandomForestClassifier(
        n_estimators=n, random_state=42, n_jobs=-1
    )
    # Boosting (градиентный): последовательное обучение
    gb = GradientBoostingClassifier(
        n_estimators=n, random_state=42, learning_rate=0.1
    )

    for name, model in [('Bagging', bag),
                        ('Random Forest', rf),
                        ('Boosting', gb)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
        results[name]['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        results[name]['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        results[name]['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        print(f"  {name:15} "
              f"Acc: {results[name]['accuracy'][-1]:.4f}  "
              f"Prec: {results[name]['precision'][-1]:.4f}  "
              f"Rec: {results[name]['recall'][-1]:.4f}  "
              f"F1: {results[name]['f1'][-1]:.4f}")

# Одиночное дерево — как эталон из прошлой задачи
tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_metrics = {
    'accuracy':  accuracy_score(y_test, tree_pred),
    'precision': precision_score(y_test, tree_pred, zero_division=0),
    'recall':    recall_score(y_test, tree_pred, zero_division=0),
    'f1':        f1_score(y_test, tree_pred, zero_division=0),
}
print(f"\nОдиночное дерево: "
      f"Acc: {tree_metrics['accuracy']:.4f}  "
      f"Prec: {tree_metrics['precision']:.4f}  "
      f"Rec: {tree_metrics['recall']:.4f}  "
      f"F1: {tree_metrics['f1']:.4f}")

# ============ Визуализация ============
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-мера']
colors = {'Bagging': '#1f77b4', 'Random Forest': '#2ca02c', 'Boosting': '#d62728'}

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]

    for name, data in results.items():
        ax.plot(n_estimators_list, data[metric],
                marker='o', label=name, color=colors[name], linewidth=2)

    # Горизонтальная линия — одиночное дерево
    ax.axhline(tree_metrics[metric], color='gray',
               linestyle='--', linewidth=1.5,
               label='Одиночное дерево')

    # Подписи значений над точками
    for name, data in results.items():
        for x, y_val in zip(n_estimators_list, data[metric]):
            ax.annotate(f"{y_val:.3f}",
                        (x, y_val), textcoords="offset points",
                        xytext=(0, 6), ha='center', fontsize=7,
                        color=colors[name])

    # Подпись одиночного дерева
    ax.annotate(f"{tree_metrics[metric]:.3f}",
                (n_estimators_list[0], tree_metrics[metric]),
                textcoords="offset points", xytext=(0, -14),
                ha='center', fontsize=8, color='gray')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Количество участников ансамбля')
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

plt.suptitle('Ансамблевая классификация на Census Income',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафик сохранён: ensemble_comparison.png")

# Сводная таблица
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 45)
print(f"{'n':>4} | {'Bagging F1':>11} | {'RF F1':>8} | {'Boosting F1':>12}")
print("-" * 45)
for i, n in enumerate(n_estimators_list):
    print(f"{n:>4} | {results['Bagging']['f1'][i]:>11.4f} | "
          f"{results['Random Forest']['f1'][i]:>8.4f} | "
          f"{results['Boosting']['f1'][i]:>12.4f}")
print(f"\nОдиночное дерево F1: {tree_metrics['f1']:.4f}")