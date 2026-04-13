import numpy as np
import subprocess
import os
import re
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
KERNEL_SIZE = 5
EXE_PATH = r"cmake-build-release\parallel_prog.exe" 

times = []

if not os.path.exists(EXE_PATH):
    print(f" ОШИБКА: Файл {EXE_PATH} не найден. Соберите проект в Release.")
    exit(1)

print(" Запуск автоматического тестирования...\n")

for N in SIZES:
    print(f"⏳ Тестирование матрицы {N}x{N}...")

    # Генерируем данные
    input_matrix = np.random.rand(N, N).astype(np.float32)
    kernel = np.random.rand(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)

    with open("input.txt", 'w') as f:
        f.write(f"{N} {N}\n")
        np.savetxt(f, input_matrix, fmt='%.4f')

    with open("kernel.txt", 'w') as f:
        f.write(f"{KERNEL_SIZE} {KERNEL_SIZE}\n")
        np.savetxt(f, kernel, fmt='%.4f')

    result = subprocess.run([EXE_PATH], capture_output=True, text=True)

    match = re.search(r"Execution Time:\s*([0-9.,eE+-]+)\s*ms", result.stdout)
    if match:
        # Меняем возможную русскую запятую на точку, чтобы float() не упал
        time_str = match.group(1).replace(',', '.')
        time_ms = float(time_str)
        times.append(time_ms)
        print(f"   Время: {time_ms:.2f} мс")
    else:
        print("    Ошибка парсинга времени! Вот что на самом деле ответил C++:")
        print("   STDOUT (Вывод):", result.stdout.strip())
        print("   STDERR (Ошибки):", result.stderr.strip())
        times.append(0)

plt.figure(figsize=(10, 6))
plt.plot(SIZES, times, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.xlabel('Размер матрицы (N x N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')

print("| Размер матрицы (N x N) | Объем задачи (элементов) | Время выполнения (мс) |")
print("| :--- | :--- | :--- |")
for n, t in zip(SIZES, times):
    print(f"| {n} x {n} | {n**2} | {t:.2f} |")
