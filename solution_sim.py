import numpy as np


def solve(matrix):
    n = matrix.shape[0]

    # Сверхбыстрая версия
    x = np.random.normal(0, 0.1, n)

    # Всего 50 итераций
    for step in range(50):
        interaction = matrix @ x
        x = np.clip(x - 0.1 * interaction, -1.0, 1.0)

    solution = (x > 0).astype(np.float64)

    # Только одна попытка локального улучшения
    energy = solution @ matrix @ solution
    matrix_dot = matrix @ solution

    for i in np.random.choice(n, size=min(100, n), replace=False):
        delta = 1 - 2 * solution[i]
        if delta * (2 * matrix_dot[i] + delta * matrix[i, i]) < 0:
            solution[i] = 1 - solution[i]
            matrix_dot += delta * matrix[:, i]

    return solution