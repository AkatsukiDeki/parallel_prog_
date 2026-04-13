#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    float& operator()(int r, int c) { return data[r * cols + c]; }
    const float& operator()(int r, int c) const { return data[r * cols + c]; }
};

Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + filename);
    int rows, cols;
    file >> rows >> cols;
    Matrix m(rows, cols);
    for (int i = 0; i < rows * cols; ++i) file >> m.data[i];
    return m;
}

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

// Новая, чистая функция с OpenMP
void convolutionOpenMP(const Matrix& input, const Matrix& kernel, Matrix& output) {
    int k_half_rows = kernel.rows / 2;
    int k_half_cols = kernel.cols / 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel.rows; ++ki) {
                for (int kj = 0; kj < kernel.cols; ++kj) {
                    int mi = i + ki - k_half_rows;
                    int mj = j + kj - k_half_cols;
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

        // Печатаем количество потоков, которые реально будет использовать OpenMP
        // (Оно задается переменной окружения OMP_NUM_THREADS из Python-скрипта)
        std::cout << "Threads used: " << omp_get_max_threads() << "\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        convolutionOpenMP(input, kernel, output);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        writeMatrix("output.txt", output);
        std::cout << "Execution Time: " << elapsed.count() << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
