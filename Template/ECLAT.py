import time
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
with open("baskets.csv", 'r', encoding='cp1251') as f:
    lines = [line.strip() for line in f if line.strip()]

products = []
for line in lines:
    items = [item.strip() for item in line.split(',')]
    if items:
        products.append(items)
N = len(products)
print(f"Датасет загружен: {N} транзакций\n")


# ============================================================
# ФУНКЦИИ APRIORI
# ============================================================
def find_frequent_itemsets(products, min_support):
    frequent_itemsets = {}
    item_counts = defaultdict(int)
    for transaction in products:
        for item in transaction:
            item_counts[item] += 1

    L1 = {frozenset([item]): count for item, count in item_counts.items()
          if count / len(products) >= min_support}
    if not L1:
        return {}

    frequent_itemsets[1] = L1
    k = 2
    while True:
        prev_itemsets = list(frequent_itemsets[k - 1].keys())
        if len(prev_itemsets) < k:
            break

        candidates = set()
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = set(prev_itemsets[i]) | set(prev_itemsets[j])
                if len(union) == k:
                    is_valid = True
                    for subset in combinations(union, k - 1):
                        if frozenset(subset) not in frequent_itemsets[k - 1]:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.add(frozenset(union))

        if not candidates:
            break

        candidate_counts = defaultdict(int)
        for transaction in products:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1

        Lk = {c: cnt for c, cnt in candidate_counts.items()
              if cnt / len(products) >= min_support}
        if not Lk:
            break

        frequent_itemsets[k] = Lk
        k += 1

    return frequent_itemsets


def generate_rules(frequent_itemsets, N, min_confidence):
    rules = []
    for size, itemsets in frequent_itemsets.items():
        if size < 2:
            continue
        for itemset, count in itemsets.items():
            itemset_list = list(itemset)
            for r in range(1, len(itemset_list)):
                for antecedent_tuple in combinations(itemset_list, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    antecedent_count = frequent_itemsets[len(antecedent)][antecedent]
                    confidence = count / antecedent_count

                    if confidence >= min_confidence:
                        support = count / N
                        consequent_count = frequent_itemsets[len(consequent)][consequent]
                        consequent_support = consequent_count / N
                        lift = confidence / consequent_support if consequent_support > 0 else 0

                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift,
                        })
    return rules


# ============================================================
# ЭКСПЕРИМЕНТ
# ============================================================
min_support = 0.10                                # фиксируем 10%
confidences = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

rules_count = []
times = []

print("=" * 60)
print(f"Эксперименты при пороге поддержки {min_support * 100:.0f}%")
print("=" * 60)

for conf in confidences:
    start = time.time()

    frequent_itemsets = find_frequent_itemsets(products, min_support)
    rules = generate_rules(frequent_itemsets, N, conf)

    elapsed = time.time() - start

    rules_count.append(len(rules))
    times.append(elapsed)

    print(f"Порог достоверности {conf*100:.0f}%: "
          f"правил = {len(rules):4d}, время = {elapsed:.3f} с")


# ============================================================
# ГРАФИК 1: БЫСТРОДЕЙСТВИЕ ОТ ПОРОГА ДОСТОВЕРНОСТИ
# ============================================================
labels = [f"{int(c*100)}%" for c in confidences]

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(labels, times, marker='o', color='#4A90D9',
         linewidth=2, markersize=8, label='Время работы, с')

ax1.set_xlabel('Порог достоверности', fontsize=12)
ax1.set_ylabel('Время работы, с', fontsize=12)
ax1.set_title('Быстродействие при изменении порога достоверности',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

for x, yv in zip(labels, times):
    ax1.annotate(f'{yv:.3f}', (x, yv), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('confidence_time.png', dpi=150)
plt.show()
print("\nГрафик сохранён: confidence_time.png")


# ============================================================
# ГРАФИК 2: КОЛИЧЕСТВО ПРАВИЛ ОТ ПОРОГА ДОСТОВЕРНОСТИ
# ============================================================
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar(labels, rules_count, color='#ED7D31', width=0.5)

ax2.set_xlabel('Порог достоверности', fontsize=12)
ax2.set_ylabel('Количество правил', fontsize=12)
ax2.set_title('Количество правил при изменении порога достоверности',
              fontsize=13, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

for i, v in enumerate(rules_count):
    ax2.annotate(f'{v}', (i, v), textcoords='offset points',
                 xytext=(0, 5), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('confidence_rules.png', dpi=150)
plt.show()
print("График сохранён: confidence_rules.png")