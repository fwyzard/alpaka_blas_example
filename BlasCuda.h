#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include "Blas.h"
#    include "common.h"

#    include <alpaka/alpaka.hpp>

#    include <cublas_v2.h>
#    include <cuda_runtime.h>

// Check for cuBLAS errors
#    define CUBLAS_CHECK(err)                                                                                         \
        if(err != CUBLAS_STATUS_SUCCESS)                                                                              \
        {                                                                                                             \
            std::cerr << "cuBLAS Error at line " << __LINE__ << "\n";                                                 \
            exit(EXIT_FAILURE);                                                                                       \
        }

class BlasCuda
{
public:
    BlasCuda(alpaka::QueueCudaRtNonBlocking& queue) : m_queue{queue}
    {
        CUBLAS_CHECK(cublasCreate(&m_handle));
        CUBLAS_CHECK(cublasSetStream(m_handle, m_queue.getNativeHandle()));
    }

    ~BlasCuda()
    {
        CUBLAS_CHECK(cublasDestroy(m_handle));
    }

    template<int N, typename T>
    inline void gemm(
        alpaka::BufCudaRt<T, Dim1D, Idx> const& A,
        alpaka::BufCudaRt<T, Dim1D, Idx> const& B,
        alpaka::BufCudaRt<T, Dim1D, Idx>& C)
    {
        assert(alpaka::getExtentProduct(A) == N * N);
        assert(alpaka::getExtentProduct(B) == N * N);
        assert(alpaka::getExtentProduct(C) == N * N);

        // Set alpha and beta for GEMM: C = alpha*A*B + beta*C
        const T alpha = 1;
        const T beta = 0;

        // Perform C = A × B using cuBLAS (column-major)
        CUBLAS_CHECK(cublasSgemm(
            m_handle,
            CUBLAS_OP_N, // Transpose A? No
            CUBLAS_OP_N, // Transpose B? No
            N, // m, n, k
            N,
            N,
            &alpha, // alpha
            A.data(), // A and its leading dimension
            N,
            B.data(), // B and its leading dimension
            N,
            &beta, // beta
            C.data(), // C and its leading dimension
            N));
    }

private:
    alpaka::QueueCudaRtNonBlocking m_queue;
    cublasHandle_t m_handle;
};

namespace traits
{

    template<>
    class Blas<alpaka::TagGpuCudaRt>
    {
    public:
        using Impl = BlasCuda;
    };

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
