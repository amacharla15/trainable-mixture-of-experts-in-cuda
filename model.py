"""
Trainable Mixture of Experts in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_naive_kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    float temp = 0.0f;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    for (int i = 0; i < K; i++) {
        temp += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = temp;
}

