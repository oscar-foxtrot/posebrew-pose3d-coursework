import numpy as np
import matplotlib.pyplot as plt

file_1 = 107
file_2 = 110
file_grid = f"temp_seq/loss_grid_{file_1}_{file_2}.npy"

f1_grid = np.linspace(100, 5000, 50)
f2_grid = np.linspace(100, 5000, 50)

loss_grid = np.load(file_grid, allow_pickle=True)

clipping = 1

loss_grid = np.clip(loss_grid, 0, clipping)

F1, F2 = np.meshgrid(f1_grid, f2_grid)

plt.figure(figsize=(9, 7))

# filled contour
cont = plt.contourf(
    F1,
    F2,
    loss_grid.T,
    levels=50,
    cmap='gray',
    vmin=0,
    vmax=clipping
)

# contour lines (optional but makes it more readable)
plt.contour(
    F1,
    F2,
    loss_grid.T,
    levels=10,
    colors='black',
    linewidths=0.3,
    alpha=0.5
)

i, j = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
plt.scatter(f1_grid[i], f2_grid[j], color='red', s=40, label='minimum')

plt.colorbar(cont, label='Ошибка реконструкции')

plt.xlabel('Фокальное расстояние f1, в условных единицах')
plt.ylabel('Фокальное расстояние f2, в условных единицах')
#plt.title('Loss landscape over focal lengths')

plt.xlim(100, 5000)
plt.ylim(100, 5000)

plt.show()