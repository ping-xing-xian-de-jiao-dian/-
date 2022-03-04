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
 * @para A: �����B: �Ҿ���C: �������
 * �ͷ�ABC������ռ�����Ŀռ�
 */
void freeAll(int* A, int* B, int* C) {
	free(A); free(B); free(C);
}

/**
 * @para matrix: ����originN: ԭ�������С��n: ���������С
 * ��ʼ��n * n�ľ���ȫ����ֵΪ1
 */
void matrixInit(int* matrix, int originN, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i * n + j] = 1;
		}
	}
}

/**
 * @para matrix: ����row: ����������row: ����������n: ÿһ�еĴ�С
 * ����ʱʹ�ã������������
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
 * @para A��B: �������C: �������MM: ������NN: ������PP: ����ǰ�ľ�����
 * OMP�������˷�
 */
void  multiplicateMatrix(int* A, int* B, int* C, int MM, int PP, int NN) {
	// ����forѭ������ͨ����˷�����
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
 * @para n: ����������
 * ȷ������зֿ��������������зֿ���
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

		// ��ʼ��
		matrixInit(A, originN, n);
		matrixInit(B, originN, n);
		printf("matrixA��С: %d * %d\n", originN, originN);
		printf("matrixB��С: %d * %d\n", originN, originN);

		// ��ʼ��ʱ
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

		// ������������̿���ƽ�����䣬0Ԫ�ز�Ӱ������
		// size - 1��0���������ų�����
		if (n % (size - 1) != 0) {
			n -= n % (size - 1);
			n += (size - 1);
		}

		// �������
		int addNum = n - originN;
		int temp = factor(size - 1);
		// �п��ԷֵõĿ�������֤���зֿ��������
		int rowNum = (size - 1) / temp;
		// �п��ԷֵõĿ���
		int colNum = temp;

		// ������ÿ������еĴ�С
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

		// �����̷���������������̡����㣬�����ܽ��
		if (rank == 0) {
			// ��ʼ��
			A = (int*)malloc(n * n * sizeof(int));
			B = (int*)malloc(n * n * sizeof(int));
			C = (int*)malloc(n * n * sizeof(int));
			matrixInit(A, originN, n);
			matrixInit(B, originN, n);

			// ��ʼ��ʱ
			start = clock();

			// ������Ϣ
			for (int i = 1; i < size; i++){
				// ÿ������Ҫ����Ŀ�ʼ��
				// �ڵڼ��еĿ飬��rowsPerBlocks����ʼ��
				int startRow = ((i - 1) / colNum) * rowsPerBlock;
				// ÿ������Ҫ����Ŀ�ʼ��
				// ��ĳһ�еĵڼ����飬��colsPerBlock����ʼ��
				int startCol = ((i - 1) % colNum) * colsPerBlock;

				// �׵�ַ�������ֽ������������ͣ�Ŀ�꣬��Ϣ��ʶ����ͨ����
				// ��Ҫ����Ŀ��Ӧ���ж�����ȥ
				MPI_Send(&A[startRow * n], rowsPerBlock * n, MPI_INT, i, i, MPI_COMM_WORLD);

				// ��Ҫ����Ŀ��Ӧ���ж�����ȥ
				for (int j = 0; j < n; j++){
					MPI_Send(&B[j * n + startCol], colsPerBlock, MPI_INT, i, i * n + j + size, MPI_COMM_WORLD);
				}
			}

			// ������Ϣ
			for (int i = 1; i < size; i++) {
				int startRow = ((i - 1) / colNum) * rowsPerBlock;
				int startCol = ((i - 1) % colNum) * colsPerBlock;

				// ��j�У�����rows����startCol�п�ʼ���������еĽ��
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



		// ��������
		else {
			int* tempA, * tempB, * tempC;

			tempA = (int*)malloc(rowsPerBlock * n * sizeof(int));
			tempB = (int*)malloc(colsPerBlock * n * sizeof(int));
			tempC = (int*)malloc(rowsPerBlock * colsPerBlock * sizeof(int));

			// ������
			MPI_Recv(&tempA[0], rowsPerBlock * n, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);

			// ������
			for (int j = 0; j < n; j++){
				MPI_Recv(&tempB[j * colsPerBlock], colsPerBlock, MPI_INT, 0, rank * n + j + size, MPI_COMM_WORLD, &status);
			}

			multiplicateMatrix(tempA, tempB, tempC, rowsPerBlock, n - addNum, colsPerBlock);

			// ������ɺ��ͻ����߳�
			for (int j = 0; j < rowsPerBlock; j++) {
				MPI_Send(&tempC[j * colsPerBlock], colsPerBlock, MPI_INT, 0, rowsPerBlock * rank + j, MPI_COMM_WORLD);
			}

			freeAll(tempA, tempB, tempC);
		}
	}

	MPI_Finalize();
}