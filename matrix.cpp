#include<stdio.h>
#include<string.h>
#include <math.h>
#include <time.h>
#include<omp.h>
#include <mpi.h>
#include <malloc.h>

#define N 1000
#define THREADNUM 16

/**
 * @para A: 左矩阵，B: 右矩阵，C: 结果矩阵
 * 释放ABC变量所占堆区的空间
 */
void freeAll(int* A, int* B, int* C) {
	free(A); free(B); free(C);
}

/**
 * @para matrix: 矩阵，originN: 原本矩阵大小，n: 扩充后矩阵大小
 * 初始化n * n的矩阵，全都赋值为1
 */
void matrixInit(int* matrix, int originN, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i * n + j] = 1;
		}
	}
}

/**
 * @para matrix: 矩阵，row: 矩阵行数，row: 矩阵列数，n: 每一行的大小
 * 调试时使用，可以输出矩阵
 */
void printMatrix(int* matrix, int row, int col, int n) {
	int* p = matrix;
	for (int y = 0; y < row; y++){
		for (int x = 0; x < col; x++){
			printf("%d ", p[x]);
		}
		p = p + n;
		printf("\n");
	}
}

/**
 * @para A、B: 计算矩阵，C: 结果矩阵，MM: 行数，NN: 列数，PP: 扩充前的矩阵行
 * OMP计算矩阵乘法
 */
void  multiplicateMatrix(int* A, int* B, int* C, int MM, int PP, int NN) {
	// 并行for循环，普通矩阵乘法计算
#pragma omp parallel for
	for (int i = 0; i < MM; i++){
		for (int j = 0; j < NN; j++){
			int sum = 0;
			for (int k = 0; k < PP; k++){
				sum = sum + A[i * PP + k] * B[k * NN + j];
			}
			C[i * NN + j] = sum;
		}
	}
}

/**
 * @para n: 矩阵行列数
 * 确定最大行分块数，进而决定列分块数
 */
int factor(int n) {
	double temp = sqrt(n);
	for (int i = temp; i >= 1; --i) {
		if (n % i == 0) return i;
	}
}

int main() {
	int n = N;
	omp_set_num_threads(THREADNUM);

	clock_t start, end;

	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	int* A, * B, * C;

	if (size == 1) {

		A = (int*)malloc(n * n * sizeof(int));
		B = (int*)malloc(n * n * sizeof(int));
		C = (int*)malloc(n * n * sizeof(int));

		int originN = n;

		// 初始化
		matrixInit(A, originN, n);
		matrixInit(B, originN, n);
		printf("matrixA大小: %d * %d\n", originN, originN);
		printf("matrixB大小: %d * %d\n", originN, originN);

		// 开始计时
		start = clock();
		multiplicateMatrix(A, B, C, n, n, n);
		end = clock();

		printf("omp: \n");
		printf("threadnum: %d\n", THREADNUM);
		printf("matrixC: % d * % d\n", originN, originN);
		printf("time cost: %lfs\n\n\n", (double)(end - start) / CLOCKS_PER_SEC);
		// printMatrix(C, originN, originN, n);

		freeAll(A, B, C);
	}




	else {
		int originN = n;

		// 扩大矩阵至进程可以平均分配，0元素不影响运算
		// size - 1是0号主进程排除在外
		if (n % (size - 1) != 0) {
			n -= n % (size - 1);
			n += (size - 1);
		}

		// 扩大的量
		int addNum = n - originN;
		int temp = factor(size - 1);
		// 行可以分得的块数，保证行列分块最相近！
		int rowNum = (size - 1) / temp;
		// 列可以分得的块数
		int colNum = temp;

		// 分完块后每块的行列的大小
		int rowsPerBlock = n / rowNum;
		int colsPerBlock = n / colNum;

		if (rank == 0) {
			printf("rowNum = %d  ", rowNum);
			printf("colNum = %d  ", colNum);
			printf("rowsPerBlock = %d  ", rowsPerBlock);
			printf("colsPerBlock = %d  ", colsPerBlock);
			printf("n = %d  ", n);
			printf("addNum = %d  ", addNum);
			printf("size = %d\n", size);
		}

		// 主进程分配任务给其它进程、计算，并接受结果
		if (rank == 0) {
			// 初始化
			A = (int*)malloc(n * n * sizeof(int));
			B = (int*)malloc(n * n * sizeof(int));
			C = (int*)malloc(n * n * sizeof(int));
			matrixInit(A, originN, n);
			matrixInit(B, originN, n);

			// 开始计时
			start = clock();

			// 发送消息
			for (int i = 1; i < size; i++){
				// 每个进程要计算的开始行
				// 在第几行的块，乘rowsPerBlocks算起始行
				int startRow = ((i - 1) / colNum) * rowsPerBlock;
				// 每个进程要计算的开始列
				// 在某一行的第几个块，乘colsPerBlock算起始列
				int startCol = ((i - 1) % colNum) * colsPerBlock;

				// 首地址，发送字节数，数据类型，目标，消息标识符，通信域
				// 把要计算的块对应的行都发过去
				MPI_Send(&A[startRow * n], rowsPerBlock * n, MPI_INT, i, i, MPI_COMM_WORLD);

				// 把要计算的块对应的列都发过去
				for (int j = 0; j < n; j++){
					MPI_Send(&B[j * n + startCol], colsPerBlock, MPI_INT, i, i * n + j + size, MPI_COMM_WORLD);
				}
			}

			// 接收消息
			for (int i = 1; i < size; i++) {
				int startRow = ((i - 1) / colNum) * rowsPerBlock;
				int startCol = ((i - 1) % colNum) * colsPerBlock;

				// 第j行（遍历rows）第startCol列开始接收所有列的结果
				for (int j = startRow; j < startRow + rowsPerBlock; j++){
					MPI_Recv(&C[j * n + startCol], colsPerBlock, MPI_INT, i, rowsPerBlock * i + (j - startRow), MPI_COMM_WORLD, &status);
				}
			}

			end = clock();

			printf("omp + mpi\n");
			printf("threadnum: %d\n", THREADNUM);
			printf("matrixC: %d * %d\n", originN, originN);
			printf("time cost: %lfs\n", (double)(end - start) / CLOCKS_PER_SEC);
			// printMatrix(C, originN, originN, n);

			freeAll(A, B, C);
		}



		// 其它进程
		else {
			int* tempA, * tempB, * tempC;

			tempA = (int*)malloc(rowsPerBlock * n * sizeof(int));
			tempB = (int*)malloc(colsPerBlock * n * sizeof(int));
			tempC = (int*)malloc(rowsPerBlock * colsPerBlock * sizeof(int));

			// 接收行
			MPI_Recv(&tempA[0], rowsPerBlock * n, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);

			// 接收列
			for (int j = 0; j < n; j++){
				MPI_Recv(&tempB[j * colsPerBlock], colsPerBlock, MPI_INT, 0, rank * n + j + size, MPI_COMM_WORLD, &status);
			}

			multiplicateMatrix(tempA, tempB, tempC, rowsPerBlock, n - addNum, colsPerBlock);

			// 计算完成后发送回主线程
			for (int j = 0; j < rowsPerBlock; j++) {
				MPI_Send(&tempC[j * colsPerBlock], colsPerBlock, MPI_INT, 0, rowsPerBlock * rank + j, MPI_COMM_WORLD);
			}

			freeAll(tempA, tempB, tempC);
		}
	}

	MPI_Finalize();
}