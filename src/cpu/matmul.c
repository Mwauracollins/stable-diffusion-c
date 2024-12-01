void matmul_forward(float* A, float* B, float* C, int rowsA, int columnsB, int commonDim) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsB; j++) {
            C[i * columnsB + j] = 0.0f;
            for (int k = 0; k < commonDim; k++) {
                C[i * columnsB + j] += A[i * commonDim + k] * B[k * columnsB + j];
            }
        }
    }
}