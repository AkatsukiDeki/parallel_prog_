#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

// Макрос для проверки ошибок CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// Вычислительное ядро, которое будет выполняться на видеокарте
__global__ void convolutionCUDA(const float* input, const float* kernel, float* output,
                                int N, int K_size, int k_half) {
    // Вычисляем глобальные 2D координаты текущего потока
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Защита от выхода за границы матрицы (если размер матрицы не кратен размеру блока)
    if (row < N && col < N) {
        float sum = 0.0f;

        for (int ki = 0; ki < K_size; ++ki) {
            for (int kj = 0; kj < K_size; ++kj) {
                int mi = row + ki - k_half;
                int mj = col + kj - k_half;

                if (mi >= 0 && mi < N && mj >= 0 && mj < N) {
                    sum += input[mi * N + mj] * kernel[ki * K_size + kj];
                }
            }
        }
        output[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    // Получаем размер блока (Block Size) из аргументов командной строки (по умолчанию 16)
    int BLOCK_SIZE = 16;
    if (argc > 1) {
        BLOCK_SIZE = std::stoi(argv[1]);
    }

    try {
        // 1. Чтение данных на хосте (CPU)
        std::ifstream fin("input.txt");
        if (!fin.is_open()) throw std::runtime_error("Cannot open input.txt");
        int N, tmp;
        fin >> N >> tmp;
        std::vector<float> h_input(N * N);
        for (int i = 0; i < N * N; ++i) fin >> h_input[i];

        std::ifstream fker("kernel.txt");
        if (!fker.is_open()) throw std::runtime_error("Cannot open kernel.txt");
        int K_size;
        fker >> K_size >> tmp;
        std::vector<float> h_kernel(K_size * K_size);
        for (int i = 0; i < K_size * K_size; ++i) fker >> h_kernel[i];

        int k_half = K_size / 2;
        std::vector<float> h_output(N * N, 0.0f);

        // 2. Выделение памяти на устройстве (GPU)
        float *d_input, *d_kernel, *d_output;
        cudaCheckError(cudaMalloc(&d_input, N * N * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_kernel, K_size * K_size * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_output, N * N * sizeof(float)));

        // 3. Копирование данных Host -> Device (PCIe шина)
        cudaCheckError(cudaMemcpy(d_input, h_input.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_kernel, h_kernel.data(), K_size * K_size * sizeof(float), cudaMemcpyHostToDevice));

        // 4. Настройка сетки потоков (Grid и Blocks)
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Настройка аппаратных таймеров GPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 5. Запуск ядра (Kernel Launch)
        cudaEventRecord(start);
        convolutionCUDA<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, N, K_size, k_half);
        cudaEventRecord(stop);

        // Ждем завершения расчетов и проверяем на ошибки (Kernel panic и т.д.)
        cudaCheckError(cudaPeekAtLastError());
        cudaCheckError(cudaDeviceSynchronize());

        // 6. Подсчет времени
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // 7. Копирование результатов Device -> Host
        cudaCheckError(cudaMemcpy(h_output.data(), d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        // 8. Очистка видеопамяти (обязательно!)
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 9. Вывод результатов
        std::ofstream fout("output.txt");
        fout << N << " " << N << "\n";
        // Пишем только первый элемент, чтобы не тратить время на диск, суть лабы не в I/O
        fout << h_output[0] << "\n";

        std::cout << "Execution Time: " << milliseconds << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}