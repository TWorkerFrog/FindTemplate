# Априорный алгоритм поиска шаблонов + ассоциативные правила
from collections import defaultdict
from itertools import combinations

min_support = float(input("Введите минимальную поддержку (в %): ")) / 100
min_confidence = float(input("Введите минимальную достоверность (в %): ")) / 100

# Загрузка данных
with open("baskets.csv", 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

products = []
for line in lines:
    items = [item.strip() for item in line.split(',')]
    if items:
        products.append(items)
N = len(products)
print(f"Датасет загружен: {N} транзакций")

print("Априорный алгоритм поиска ассоциативных правил\n")
frequent_itemsets = {}

item_counts = defaultdict(int)
for transaction in products:
    for item in transaction:
        item_counts[item] += 1

L1 = {frozenset([item]): count for item, count in item_counts.items()
      if count / N >= min_support}

if not L1:
    print("Нет товаров, удовлетворяющих критической частоте!")
    exit()

frequent_itemsets[1] = L1
print(f"L1: найдено {len(L1)} частых товаров")
for itemset, count in sorted(L1.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {set(itemset)}: {count} ({count / N * 100:.1f}%)")
if len(L1) > 10:
    print(f"  ... и еще {len(L1) - 10} товаров")
print()

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

    Lk = {candidate: count for candidate, count in candidate_counts.items()
          if count / N >= min_support}

    if not Lk:
        break

    frequent_itemsets[k] = Lk
    print(f"L{k}: найдено {len(Lk)} частых наборов")
    for itemset, count in sorted(Lk.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {set(itemset)}: {count} ({count / N * 100:.1f}%)")
    if len(Lk) > 10:
        print(f"  ... и еще {len(Lk) - 10} наборов")
    print()

    k += 1

rules = []

for size, itemsets in frequent_itemsets.items():
    if size < 2:
        continue

    for itemset, count in itemsets.items():
        itemset_list = list(itemset)

        # Генерируем все разбиения itemset на антецедент и консеквент
        for r in range(1, len(itemset_list)):
            for antecedent_tuple in combinations(itemset_list, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                # Достоверность: count(X → Y) / count(X)
                antecedent_count = frequent_itemsets[len(antecedent)][antecedent]
                confidence = count / antecedent_count

                if confidence >= min_confidence:
                    support = count / N

                    consequent_count = frequent_itemsets[len(consequent)][consequent]
                    consequent_support = consequent_count / N

                    # Лифт: confidence(X → Y) / support(Y)
                    lift = confidence / consequent_support if consequent_support > 0 else 0

                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })

# Сортировка правил
rules.sort(key=lambda r: r['support'], reverse=True)

# Вывод правил
print(f"\nВсего правил: {len(rules)}")
print(f"Порог поддержки: {min_support * 100}%")
print(f"Порог достоверности: {min_confidence * 100}%")

for rule in rules:
    ant_str = ', '.join(sorted(rule['antecedent']))
    con_str = ', '.join(sorted(rule['consequent']))
    print(f"{{{ant_str}}} → {{{con_str}}}  "
          f"[поддержка: {rule['support'] * 100:.1f}%, "
          f"достоверность: {rule['confidence'] * 100:.1f}%, "
          f"lift: {rule['lift']:.3f}]")