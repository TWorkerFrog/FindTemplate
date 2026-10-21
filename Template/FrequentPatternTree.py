# FP-Growth: поиск частых наборов через FP-дерево
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Ввод критической частоты
min_support = float(input("Введите минимальную поддержку (в %): ")) / 100

# Загрузка данных
with open("baskets.csv", 'r', encoding='cp1251') as f:
    lines = [line.strip() for line in f if line.strip()]

products = []
for line in lines:
    items = [item.strip() for item in line.split(',')]
    if items:
        products.append(items)
N = len(products)
min_count = N * min_support

print(f"Датасет загружен: {N} транзакций")
print(f"Минимальное вхождение: {min_count:.1f}\n")


# Узел FP-дерева
class FPNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


# Построение FP-дерева
def build_fp_tree(transactions, min_count):
    item_counts = Counter()
    for trans in transactions:
        item_counts.update(trans)

    frequent = {item: cnt for item, cnt in item_counts.items() if cnt >= min_count}
    if not frequent:
        return None, None

    header_table = {item: [cnt, None] for item, cnt in frequent.items()}
    root = FPNode(None, 0, None)

    for trans in transactions:
        filtered = [item for item in trans if item in frequent]
        filtered.sort(key=lambda x: frequent[x], reverse=True)

        current = root
        for item in filtered:
            if item in current.children:
                current.children[item].count += 1
            else:
                new_node = FPNode(item, 1, current)
                current.children[item] = new_node
                if header_table[item][1] is None:
                    header_table[item][1] = new_node
                else:
                    node = header_table[item][1]
                    while node.link is not None:
                        node = node.link
                    node.link = new_node
            current = current.children[item]

    return root, header_table


# Визуализация FP-дерева через matplotlib
def plot_fp_tree(root, header_table, max_depth=3, max_children=6):
    """
    Рисует FP-дерево. Ограничиваем глубину и число детей для читаемости.
    """
    fig, ax = plt.subplots(figsize=(20, 12))

    positions = {}
    counter = [0]

    def assign_positions(node, depth):
        if depth > max_depth:
            return None
        if not node.children or depth == max_depth:
            x = counter[0]
            counter[0] += 1
        else:
            children = list(node.children.values())[:max_children]
            xs = []
            for child in children:
                cx = assign_positions(child, depth + 1)
                if cx is not None:
                    xs.append(cx)
            x = sum(xs) / len(xs) if xs else counter[0]
        positions[node] = (x, -depth)
        return x

    assign_positions(root, 0)

    def draw(node, depth):
        if node not in positions:
            return
        x, y = positions[node]
        for child in list(node.children.values())[:max_children]:
            if child in positions:
                cx, cy = positions[child]
                ax.plot([x, cx], [y, cy], 'k-', lw=1, alpha=0.6)
                draw(child, depth + 1)

    draw(root, 0)

    for node, (x, y) in positions.items():
        if node.name is None:
            label = f"root\n({node.count})"
            color = '#cccccc'
        else:
            label = f"{node.name}\n({node.count})"
            color = '#a8d8ff'

        ax.text(x, y, label, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                          edgecolor='black', linewidth=1),
                fontsize=9)

    ax.set_title(f"FP-дерево (min_support={min_support * 100}%, "
                 f"глубина ≤ {max_depth})", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('fp_tree.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("FP-дерево сохранено: fp_tree.png")

    print("\nHeader table (товар: частота):")
    for item, (cnt, _) in sorted(header_table.items(), key=lambda x: -x[1][0]):
        print(f"  {item}: {cnt}")


# Запуск
root, header_table = build_fp_tree(products, min_count)

if header_table:
    plot_fp_tree(root, header_table, max_depth=2, max_children=4)
else:
    print("Нет частых товаров!")