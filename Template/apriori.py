# Априорный алгоритм поиска шаблонов
from collections import defaultdict
from itertools import combinations

min_count = int(input("Введите критическую частоту (в %): ")) / 100

# Загрузка данных
with open("baskets.csv", 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# print("Датасет:")
products = []
c = 0
for line in lines:
    items = [item.strip() for item in line.split(',')] # Каждая строка разбивается на отдельные товары
    if items:
        products.append(items) # Получается список транзакций products
        c += 1
        # print(items)
print(f"Датасет загружен {c}\n")

print("Алгоритм (подсчет, фильтр, сортировка)")
frequent_itemsets = {}  # {размер: {набор: частота}}

item_counts = defaultdict(int)
for transaction in products:
    for item in transaction:
        item_counts[item] += 1 # Считаем, сколько раз встречается каждый товар во всех транзакциях.

L1 = {frozenset([item]): count for item, count in item_counts.items()
      if count / len(products) >= min_count}

if not L1:
    print("Нет товаров, удовлетворяющих критической частоте!")
    exit()

frequent_itemsets[1] = L1
print(f"L1: найдено {len(L1)} частых товаров")
for itemset, count in sorted(L1.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {set(itemset)}: {count} ({count / len(products) * 100:.1f}%)")
if len(L1) > 10:
    print(f"  ... и еще {len(L1) - 10} товаров")
print()


k = 2
while True:
    # Получаем частые наборы предыдущего размера
    prev_itemsets = list(frequent_itemsets[k - 1].keys())

    if len(prev_itemsets) < k:
        break

    # Генерация кандидатов размера k
    candidates = set()

    for i in range(len(prev_itemsets)):
        for j in range(i + 1, len(prev_itemsets)):
            union = set(prev_itemsets[i]) | set(prev_itemsets[j]) # Важная проверка подмножеств на размер k

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

    # Подсчет частот для кандидатов
    candidate_counts = defaultdict(int)

    for transaction in products:
        transaction_set = set(transaction)
        for candidate in candidates:
            if candidate.issubset(transaction_set):
                candidate_counts[candidate] += 1

    # Фильтруем по мин. частоте
    Lk = {candidate: count for candidate, count in candidate_counts.items()
          if count / len(products) >= min_count}

    if not Lk:
        break

    # Сохраняем частые наборы
    frequent_itemsets[k] = Lk
    print(f"L{k}: найдено {len(Lk)} частых наборов")

    # Выводим первые 10 для примера
    for itemset, count in sorted(Lk.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {set(itemset)}: {count} ({count / len(products) * 100:.1f}%)")
    if len(Lk) > 10:
        print(f"  ... и еще {len(Lk) - 10} наборов")
    print()

    k += 1

# Чем выше порог частоты, тем быстрее обработка, но меньше шаблонов находится