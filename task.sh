#!/bin/bash
#SBATCH --job-name=mpi_lab3
#SBATCH --output=benchmark_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00

module purge
module load intel/mpi5

echo "=== Лабораторная работа №3: Полный цикл MPI ==="

for N in 200 400 800 1200 1600 2000; do
    echo "=========================================="
    echo "Генерация данных для матрицы ${N}x${N}..."
    python3 gen_data.py $N
    
    for P in 1 2 4 8; do
        echo -n "Матрица ${N}x${N} | Ядер: ${P} | "
        mpirun -np $P ./parallel_prog
    done
done
echo "=========================================="
echo "Тестирование завершено!"
