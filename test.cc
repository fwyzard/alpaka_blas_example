#include "BlasCpu.h"
#include "BlasCuda.h"
#include "common.h"

#include <alpaka/alpaka.hpp>

#include <iomanip>
#include <iostream>
#include <random>

// Print a column-major matrix
template<int N, typename T>
void print(alpaka::BufCpu<T, Dim1D, Idx> const& M)
{
    assert(alpaka::getExtentProduct(M) == N * N);

    for(int row = 0; row < N; ++row)
    {
        for(int col = 0; col < N; ++col)
        {
            std::cout << std::fixed << std::setprecision(2) << std::setw(7) << M[col * N + row] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    constexpr int N = 4;

    // Host platform and device
    alpaka::PlatformCpu host_platform{};
    auto host = alpaka::getDevByIdx(host_platform, 0u);

    // Allocate matrices (column-major)
    auto A = alpaka::allocBuf<float, Idx>(host, Idx{N * N});
    auto B = alpaka::allocBuf<float, Idx>(host, Idx{N * N});
    auto C = alpaka::allocBuf<float, Idx>(host, Idx{N * N});

    // Fill A and B with random floats centered around 0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 10.0f);

    for(int i = 0; i < N * N; ++i)
    {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    std::cout << "Matrix A:\n";
    print<N>(A);
    std::cout << '\n';
    std::cout << "Matrix B:\n";
    print<N>(B);
    std::cout << '\n';

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    {
        alpaka::PlatformCudaRt platform;
        alpaka::DevCudaRt device = alpaka::getDevByIdx(platform, 0u);
        alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{device};

        auto A_d = alpaka::allocAsyncBuf<float, Idx>(queue, Idx{N * N});
        auto B_d = alpaka::allocAsyncBuf<float, Idx>(queue, Idx{N * N});
        auto C_d = alpaka::allocAsyncBuf<float, Idx>(queue, Idx{N * N});
        alpaka::memcpy(queue, A_d, A);
        alpaka::memcpy(queue, B_d, B);

        Blas<alpaka::TagGpuCudaRt> blas(queue);
        blas.gemm<4>(A_d, B_d, C_d);
        alpaka::memcpy(queue, C, C_d);

        alpaka::wait(queue);
        std::cout << "CUDA Matrix C = A × B:\n";
        print<N>(C);
        std::cout << '\n';
    }
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    {
        alpaka::PlatformCpu platform;
        alpaka::DevCpu device = alpaka::getDevByIdx(platform, 0u);
        alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{device};

        Blas<alpaka::TagCpuSerial> blas(queue);
        blas.gemm<4>(A, B, C);

        alpaka::wait(queue);
        std::cout << "CPU Matrix C = A × B:\n";
        print<N>(C);
        std::cout << '\n';
    }
#endif

    return 0;
}
