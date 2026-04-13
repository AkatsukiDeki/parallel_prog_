#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>

// Структура для хранения матрицы в непрерывном блоке памяти
struct Matrix {
    int rows, cols;
    std::vector<float> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float& operator()(int r, int c) { return data[r * cols + c]; }
    const float& operator()(int r, int c) const { return data[r * cols + c]; }
};

// Чтение матрицы из текстового файла
Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + filename);

    int rows, cols;
    file >> rows >> cols;
    Matrix m(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        file >> m.data[i];
    }
    return m;
}

// Запись матрицы в файл
void writeMatrix(const std::string& filename, const Matrix& m) {
    std::ofstream file(filename);
    file << m.rows << " " << m.cols << "\n";
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            file << m(i, j) << (j == m.cols - 1 ? "" : " ");
        }
        file << "\n";
    }
}

// Рабочая функция для потока (обрабатывает заданный диапазон строк)
void convolutionWorker(const Matrix& input, const Matrix& kernel, Matrix& output, int start_row, int end_row) {
    int k_half_rows = kernel.rows / 2;
    int k_half_cols = kernel.cols / 2;

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            float sum = 0.0f;
            // Проход ядром по окрестности пикселя
            for (int ki = 0; ki < kernel.rows; ++ki) {
                for (int kj = 0; kj < kernel.cols; ++kj) {
                    int mi = i + ki - k_half_rows;
                    int mj = j + kj - k_half_cols;

                    // Zero-padding для границ
                    if (mi >= 0 && mi < input.rows && mj >= 0 && mj < input.cols) {
                        sum += input(mi, mj) * kernel(ki, kj);
                    }
                }
            }
            output(i, j) = sum;
        }
    }
}

int main() {
    try {
        Matrix input = readMatrix("input.txt");
        Matrix kernel = readMatrix("kernel.txt");
        Matrix output(input.rows, input.cols);

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // Fallback

        std::vector<std::thread> threads;
        int chunk_size = input.rows / num_threads;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Запуск потоков. Барьеры памяти release/acquire неявно создаются
        // конструктором std::thread и методом join().
        for (unsigned int i = 0; i < num_threads; ++i) {
            int start_row = i * chunk_size;
            int end_row = (i == num_threads - 1) ? input.rows : start_row + chunk_size;

            threads.emplace_back(convolutionWorker, std::cref(input), std::cref(kernel), std::ref(output), start_row, end_row);
        }

        // Синхронизация: ждем завершения всех потоков
        for (auto& t : threads) {
            t.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        writeMatrix("output.txt", output);

        std::cout << "Task Volume (Elements): " << input.rows * input.cols << "\n";
        std::cout << "Execution Time: " << elapsed.count() << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}