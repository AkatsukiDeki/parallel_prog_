import numpy as np
import subprocess
import os
import re
import matplotlib.pyplot as plt

# Настройки бенчмарка
SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCESSES = [1, 2, 4, 8] # MPI-процессы (не потоки!)
KERNEL_SIZE = 5
EXE_PATH = r"cmake-build-release\parallel_prog.exe"

if not os.path.exists(EXE_PATH):
    print(f" ОШИБКА: Файл {EXE_PATH} не найден. Соберите проект в Release.")
    exit(1)

results = {p: [] for p in PROCESSES}

print(" Запуск тестирования архитектуры MPI...\n")

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

    for p in PROCESSES:
        print(f"   MPI: {p} процессов...")

        # Запуск через mpiexec
        cmd = ["mpiexec", "-n", str(p), EXE_PATH]
        result = subprocess.run(cmd, capture_output=True, text=True)

        match = re.search(r"Execution Time:\s*([0-9.,eE+-]+)\s*ms", result.stdout)
        if match:
            time_ms = float(match.group(1).replace(',', '.'))
            results[p].append(time_ms)
        else:
            print("    Ошибка MPI:", result.stderr.strip() or result.stdout.strip())
            results[p].append(0)

# --- Построение графика ---
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'm']
for i, p in enumerate(PROCESSES):
    plt.plot(SIZES, results[p], marker='o', linestyle='-', color=colors[i % len(colors)],
             linewidth=2, markersize=6, label=f'{p} процессов')

plt.title('MPI: Время выполнения от размера матрицы и кол-ва процессов', fontsize=14)
plt.xlabel('Размер матрицы (N x N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('mpi_benchmark_plot.png', dpi=300, bbox_inches='tight')


print("\n === ТАБЛИЦА ДЛЯ README.md ===\n")
print("| Размер (N x N) | Время (1 процесс) | Время (2 процесса) | Время (4 процесса) | Время (8 процессов) |")
print("| :--- | :--- | :--- | :--- | :--- |")
for i, n in enumerate(SIZES):
    row = f"| {n} x {n} "
    for p in PROCESSES:
        row += f"| {results[p][i]:.2f} "
    row += "|"
    print(row)
