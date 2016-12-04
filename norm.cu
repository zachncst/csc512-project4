#include <cuda_runtime.h>

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include "device_launch_parameters.h"

// #include<sys/time.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 128
#define SIZE BLOCK_SIZE*BLOCK_SIZE*GRID_SIZE*GRID_SIZE

void checkresult(float *ref, float *in, float *out, float *mul, int width) {
	for (int i = 0; i < GRID_SIZE; i++) {
		for (int j = 0; j < GRID_SIZE; j++) {
			float sum = 0.0f;
			int start = j * BLOCK_SIZE * width + i * BLOCK_SIZE;
			for (int ii = 0; ii < BLOCK_SIZE; ii++) {
				for (int jj = 0; jj < BLOCK_SIZE; jj++) {
					sum += in[start + ii * width + jj] * mul[jj];
				}
			}
			for (int ii = 0; ii < BLOCK_SIZE; ii++) {
				for (int jj = 0; jj < BLOCK_SIZE; jj++) {
					if (jj % 2 == 0 && ii % 2 == 0)
						ref[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] = 2.0 * in[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] / sum;
					else if (jj % 2 == 1 && ii % 2 == 0)
						ref[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] = in[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] / sum;
					else if (jj % 2 == 1 && ii % 2 == 1)
						ref[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] = (-1.0) * in[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] / sum;
					else
						ref[(j * BLOCK_SIZE + jj) * width + i * BLOCK_SIZE + ii] = 0.0f;
				}
			}
		}
	}

	for (int i = 0; i < SIZE; i++) {
		if (abs(ref[i] - out[i]) > 1.e-6) {
			printf("Diff %f\n", abs(ref[i] - out[i]));
			printf("results checking failed at %d ref %f out %f\n", i, ref[i], out[i]);
			return;
		}
	}
	printf("results checking passed!\n");
}

__global__ void norm(float *in, float *out, float *mul, int width) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx >= width || ty >= SIZE / width) return;
	int start = blockIdx.x * blockDim.x * width + blockIdx.y * blockDim.y;
	float sum = 0.0f;

	for (int i = 0; i < BLOCK_SIZE; i++) {
		for (int j = 0; j < BLOCK_SIZE; j++) {
			sum += in[start + i * width + j] * mul[j];
		}
	}

	if (tx % 2 == 0 && ty % 2 == 0)
		out[tx * width + ty] = 2.0 * in[tx * width + ty] / sum;
	else if (tx % 2 == 1 && ty % 2 == 0)
		out[tx * width + ty] = in[tx * width + ty] / sum;
	else if (tx % 2 == 1 && ty % 2 == 1)
		out[tx * width + ty] = (-1.0) * in[tx * width + ty] / sum;
	else
		out[tx * width + ty] = 0.0f;
}

int main() {
	//float *hA_in = (float *)malloc(SIZE * sizeof(float));
	//float *hA_out = (float *)malloc(SIZE * sizeof(float));
	//float *hB_in = (float *)malloc(BLOCK_SIZE * sizeof(float));
	float *ref = (float *)malloc(SIZE * sizeof(float));
	float *hA_in, *hA_out, *hB_in;
	float *dA_in, *dA_out, *dB_in;

	cudaMallocHost((void**)&hA_in, SIZE * sizeof(float));
	cudaMallocHost((void**)&hA_out, SIZE * sizeof(float));
	cudaMallocHost((void**)&hB_in, BLOCK_SIZE * sizeof(float));

	srand(2016);

	for (int i = 0; i < SIZE; i++) {
		hA_in[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < BLOCK_SIZE; i++) {
		hB_in[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaMalloc((void **)&dA_in, SIZE * sizeof(float));
	cudaMalloc((void **)&dA_out, SIZE * sizeof(float));
	cudaMalloc((void **)&dB_in, BLOCK_SIZE * sizeof(float));

	cudaMemcpy(dA_in, hA_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB_in, hB_in, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(GRID_SIZE, GRID_SIZE, 1);
	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	norm << <grid, block >> > (dA_in, dA_out, dB_in, BLOCK_SIZE * GRID_SIZE);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("kernel time %fs\n", milliseconds);
	cudaMemcpy(hA_out, dA_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	checkresult(ref, hA_in, hA_out, hB_in, BLOCK_SIZE * GRID_SIZE);
}
