# -*- coding: utf-8 -*-
"""
Визуализация результатов экспериментов по Apriori.
Строит два графика:
  1. Быстродействие при изменении порога поддержки (линия).
  2. Число частых наборов разной длины (группированные столбцы).
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ДАННЫЕ ИЗ ЭКСПЕРИМЕНТОВ
# ============================================================
# Пороги поддержки (в процентах) — по оси X
supports = [1, 3, 5, 10, 15]

# Количество частых наборов каждой длины
L1 = [76, 226, 376, 751, 1126]   # длина 1
L2 = [74, 36, 25, 7, 5]          # длина 2
L3 = [170, 19, 3, 0, 0]          # длина 3 (если 0 — столбца не будет)
L4 = [17, 0, 0, 0, 0]            # длина 4 (если 0 — столбца не будет)

# Время работы (в секундах). В таблице указано приблизительно, поэтому
# переведём в удобные значения. При 1% — 0.9-1.8 с, при 3% — 170 мс,
# при 5% — 90 мс, при 10% — 16 мс, при 15% — 15 мс.
times_seconds = [1.4, 0.170, 0.090, 0.016, 0.015]
times_ms = [t * 1000 for t in times_seconds]  # в миллисекундах, если нужно


# ============================================================
# ГРАФИК 1: БЫСТРОДЕЙСТВИЕ ПРИ ИЗМЕНЕНИИ ПОРОГА ПОДДЕРЖКИ
# ============================================================
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(supports, times_ms, marker='o', color='#4A90D9',
         linewidth=2, markersize=8, label='Время работы, мс')

ax1.set_xlabel('Порог поддержки, %', fontsize=12)
ax1.set_ylabel('Время работы, мс', fontsize=12)
ax1.set_title('1. Быстродействие при изменении порога поддержки',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

# Подписи точек значениями
for x, y in zip(supports, times_ms):
    ax1.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9)

# Ось X — категориальная, чтобы точки стояли ровно над подписями
ax1.set_xticks(supports)
ax1.set_xticklabels([f'{s}%' for s in supports])

plt.tight_layout()
plt.savefig('support_time.png', dpi=150)
plt.show()


# ============================================================
# ГРАФИК 2: ЧИСЛО ЧАСТЫХ НАБОРОВ РАЗНОЙ ДЛИНЫ
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

x = np.arange(len(supports))
width = 0.2  # ширина одного столбца

# Сдвиги для каждой серии, чтобы столбцы стояли рядом
offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

# Рисуем столбцы только там, где значения не нулевые
# (чтобы не было пустых столбцов, как на картинке)
bars1 = ax2.bar(x + offsets[0], L1, width, label='частота набора 1', color='#5B9BD5')
bars2 = ax2.bar(x + offsets[1], L2, width, label='частота набора 2', color='#ED7D31')
bars3 = ax2.bar(x + offsets[2], L3, width, label='частота набора 3', color='#FFC000')
bars4 = ax2.bar(x + offsets[3], L4, width, label='частота набора 4', color='#70AD47')

# Подписи значений над столбцами (только если > 0)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Порог поддержки, %', fontsize=12)
ax2.set_ylabel('Число наборов', fontsize=12)
ax2.set_title('2. Число частых наборов разной длины при изменении порога поддержки',
              fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{s}%' for s in supports])
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('support_counts.png', dpi=150)
plt.show()