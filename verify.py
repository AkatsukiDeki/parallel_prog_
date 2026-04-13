import numpy as np
from scipy.signal import correlate2d
import subprocess
import os

# 1. Генерация исходных данных
N = 1000
K = 5
print(f"Generating data for {N}x{N} matrix...")

input_matrix = np.random.rand(N, N).astype(np.float32)
kernel = np.random.rand(K, K).astype(np.float32)

def save_matrix(filename, matrix):
    with open(filename, 'w') as f:
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        np.savetxt(f, matrix, fmt='%.6f')

save_matrix("input.txt", input_matrix)
save_matrix("kernel.txt", kernel)

# 2. Запуск УЖЕ скомпилированной программы
# Путь берем из вашего лога сборки CLion
exe_path = r"cmake-build-debug\parallel_prog.exe"

if not os.path.exists(exe_path):
    print(f"❌ ОШИБКА: Файл {exe_path} не найден!")
    print("Убедитесь, что вы нажали кнопку Build (молоток) в CLion.")
    exit(1)

print("Running C++ convolution...")
# Запускаем готовый .exe файл
result = subprocess.run([exe_path], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Ошибки C++:", result.stderr)

# 3. Верификация результатов с помощью Python
print("Verifying results using scipy...")
try:
    cpp_output = np.loadtxt("output.txt", skiprows=1, dtype=np.float32)
except FileNotFoundError:
    print("❌ ОШИБКА: Файл output.txt не создан. Программа на C++ упала или не отработала до конца.")
    exit(1)

expected_output = correlate2d(input_matrix, kernel, mode='same', boundary='fill', fillvalue=0)

is_correct = np.allclose(cpp_output, expected_output, rtol=1e-4, atol=1e-4)

if is_correct:
    print("✅ SUCCESS: C++ output matches Python/SciPy verification!")
else:
    print("❌ ERROR: Output mismatch detected.")
    diff = np.abs(cpp_output - expected_output)
    print(f"Max difference: {np.max(diff)}")