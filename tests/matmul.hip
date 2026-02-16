#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "ml/math.hpp"
#include "common/math.hpp"
#include <hip/hip_runtime.h>

TEST_CASE("Matrix Multiplication Correctness") {
    int M = 64, N = 64, K = 64;

    float alpha = 2;
    float beta = 3;
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N), result_cpu(M*N), result_gpu(M*N);

    randomize_matrix(h_A);
    randomize_matrix(h_B);
    randomize_matrix(h_C);

    float* d_A;
    float* d_B;
    float* d_C;

    int size_A = M*K*sizeof(float);
    int size_B = K*N*sizeof(float);
    int size_C = M*N*sizeof(float);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);
    
    SUBCASE("Naive Mode") {
        hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice);

        gpu_matmul(d_A, d_B, d_C, alpha, beta, M, N, K, MatMulMode::Naive);
        cpu_matmul(h_A, h_B, h_C, alpha, beta, M, N, K);

        hipMemcpy(result_gpu.data(), d_C, size_C, hipMemcpyDeviceToHost);

        for (int i=0; i<M*N; i++) {
            CHECK(
                result_gpu[i] == doctest::Approx(h_C[i]).epsilon(0.001)
            );
        }
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}