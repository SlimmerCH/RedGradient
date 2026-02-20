#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "blas/math.hpp"
#include "common/math.hpp"
#include <hip/hip_runtime.h>
#include "blas/kernels/utils.hpp"

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

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    
    SUBCASE("Naive Mode") {
        HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));

        gpu_matmul(d_A, d_B, d_C, alpha, beta, M, N, K, MatMulMode::Naive);
        cpu_matmul(h_A, h_B, h_C, alpha, beta, M, N, K);

        HIP_CHECK(hipMemcpy(result_gpu.data(), d_C, size_C, hipMemcpyDeviceToHost));

        for (int i=0; i<M*N; i++) {
            CHECK(
                result_gpu[i] == doctest::Approx(h_C[i]).epsilon(0.001)
            );
        }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}