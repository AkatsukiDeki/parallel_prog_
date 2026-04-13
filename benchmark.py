import numpy as np
import subprocess
import os
import re
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8] # Количество потоков для экспериментов
KERNEL_SIZE = 5
EXE_PATH = r"cmake-build-release\parallel_prog.exe"

if not os.path.exists(EXE_PATH):
    print(f" ОШИБКА: Файл {EXE_PATH} не найден. Соберите проект в Release.")
    exit(1)

results = {t: [] for t in THREADS}

print(" Запуск масштабного тестирования OpenMP...\n")

for N in SIZES:
    print(f"Генерация данных {N}x{N}...")
    input_matrix = np.random.rand(N, N).astype(np.float32)
    kernel = np.random.rand(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)

    with open("input.txt", 'w') as f:
        f.write(f"{N} {N}\n")
        np.savetxt(f, input_matrix, fmt='%.4f')

    with open("kernel.txt", 'w') as f:
        f.write(f"{KERNEL_SIZE} {KERNEL_SIZE}\n")
        np.savetxt(f, kernel, fmt='%.4f')

    for t in THREADS:
        print(f"  ⏳ Тестирование на {t} потоках...")

        # Заставляем OpenMP использовать конкретное число потоков
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t)

        result = subprocess.run([EXE_PATH], capture_output=True, text=True, env=env)

        match = re.search(r"Execution Time:\s*([0-9.,eE+-]+)\s*ms", result.stdout)
        if match:
            time_ms = float(match.group(1).replace(',', '.'))
            results[t].append(time_ms)
        else:
            print("    Ошибка C++:", result.stderr.strip())
            results[t].append(0)


plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'm', 'c']
for i, t in enumerate(THREADS):
    plt.plot(SIZES, results[t], marker='o', linestyle='-', color=colors[i % len(colors)],
             linewidth=2, markersize=6, label=f'{t} потоков')

plt.title('OpenMP: Время выполнения от размера матрицы и числа потоков', fontsize=14)
plt.xlabel('Размер матрицы (N x N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('omp_benchmark_plot.png', dpi=300, bbox_inches='tight')


print("\n === ТАБЛИЦА ДЛЯ README.md ===\n")
print("| Размер (N x N) | Объем задачи | Время (1 поток) | Время (2 потока) | Время (4 потока) | Время (8 потоков) |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for i, n in enumerate(SIZES):
    row = f"| {n} x {n} | {n**2} "
    for t in THREADS:
        row += f"| {results[t][i]:.2f} "
    row += "|"
    print(row)
