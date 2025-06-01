from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
h36m_pts = [(3, 2), (2, 1), (1, 0), (0, 4), (4, 5), (5, 6), \
    (13, 12), (12, 11), (11, 8), (8, 14), (14, 15), (15, 16), \
    (8, 9), (9, 10), (8, 7), (7, 0)]

file_number = 795
file_path = f'file_{file_number}_2_withoutnorm.npy'
keypoints = np.load(file_path, allow_pickle=True)

# Исходные данные: d(t), d_i(t), t
# Задать начальный w0 (например, равномерный)
n = len(h36m_pts)

#skeleton_sums = []

import numpy as np
from scipy.optimize import minimize

# h36m_pts - пары индексов ключевых точек
# keypoints - np.array со всеми кадрами и координатами

n = len(h36m_pts)
T = len(keypoints)

# Предварительно считаем d(t; i,j) для всех t и всех рёбер
# shape: (T, n)
d_t_ij = np.zeros((T, n))
for t in range(T):
    frame = keypoints[t]
    for k, (i, j) in enumerate(h36m_pts):
        p_i = frame[i]
        p_j = frame[j]
        dist = np.linalg.norm(p_i - p_j)
        d_t_ij[t, k] = dist

d_t_ij = d_t_ij * 100

'''
plt.plot(d_t_ij)
plt.show()
'''
'''
plt.plot(d_t_ij[:, 10])
plt.show()
'''

# Функция J(tilde_w)
def objective(w):
    w = np.array(w)
    w = w / np.sum(w)  # для надёжности нормируем (опционально)
    
    # Вычисляем нормирующий знаменатель для каждого t: w^T d(t)
    denom = d_t_ij.dot(w)  # shape (T,)

    bar_norm = np.mean(denom)  # scalar

    # Предполагаем, что denom != 0 (проверить в реальном коде!)
    # Вычисляем normalized distances: shape (T,n)
    normalized = bar_norm * d_t_ij / denom[:, np.newaxis]
    
    # Вычисляем вариацию по времени для каждого ребра
    var_per_edge = np.var(normalized, axis=0)
    
    # Целевая функция — сумма вариаций по всем рёбрам

    return np.sum(var_per_edge)

# Ограничения: веса на симплексе
constraints = (
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
)

# Границы: w_i >= 0
bounds = [(0, None)] * n

# Начальное приближение
w0 = np.ones(n) / n

result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimization result:", result)
print("Optimal weights w:", result.x)

from sklearn.metrics import r2_score
plt.plot(d_t_ij.dot(result.x) / np.mean(d_t_ij.dot(result.x)))
plt.plot(d_t_ij.dot(np.ones(n) / n) / np.mean(d_t_ij.dot(np.ones(n) / n)))
plt.show()

print(r2_score(d_t_ij.dot(result.x) / np.mean(d_t_ij.dot(result.x)), d_t_ij.dot(np.ones(n) / n) / np.mean(d_t_ij.dot(np.ones(n) / n))))
