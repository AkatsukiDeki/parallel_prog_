#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>  
#include <algorithm>

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

int main(int argc, char** argv) {
    // Инициализация подсистемы MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, K_size;
    Matrix input(0, 0), kernel(0, 0);

    // Только Master (Rank 0) читает данные с диска
    if (rank == 0) {
        try {
            input = readMatrix("input.txt");
            kernel = readMatrix("kernel.txt");
            N = input.rows;
            K_size = kernel.rows;
        } catch (...) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Master рассылает размеры матриц всем Worker'ам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Выделяем память под данные на Worker'ах
    if (rank != 0) {
        input = Matrix(N, N);
        kernel = Matrix(K_size, K_size);
    }

    double start_time = MPI_Wtime();

    MPI_Bcast(input.data.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel.data.data(), K_size * K_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size;
    int remainder = N % size;

    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + std::min(rank, remainder);
    int end_row = start_row + my_rows;

    std::vector<float> local_output(my_rows * N, 0.0f);

    int k_half_rows = kernel.rows / 2;
    int k_half_cols = kernel.cols / 2;

    // Вычислительное ядро процессора
    for (int i = start_row; i < end_row; ++i) {
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
            local_output[(i - start_row) * N + j] = sum;
        }
    }

    // Подготовка массивов для MPI_Gatherv (Мастер должен знать, сколько элементов пришлет каждый)
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    if (rank == 0) {
        int current_displ = 0;
        for (int i = 0; i < size; ++i) {
            int p_rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = p_rows * N;
            displs[i] = current_displ;
            current_displ += recvcounts[i];
        }
    }

    Matrix final_output(0, 0);
    if (rank == 0) final_output = Matrix(N, N);

    MPI_Gatherv(local_output.data(), my_rows * N, MPI_FLOAT,
                rank == 0 ? final_output.data.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        writeMatrix("output.txt", final_output);
        std::cout << "Execution Time: " << (end_time - start_time) * 1000.0 << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
