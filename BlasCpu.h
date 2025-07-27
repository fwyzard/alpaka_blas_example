#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#    include "Blas.h"
#    include "common.h"

#    include <alpaka/alpaka.hpp>

#    include <cblas.h>

class BlasCpu
{
public:
    BlasCpu(alpaka::QueueCpuBlocking& queue [[maybe_unused]])
    {
    }

    template<int N, typename T>
    inline void gemm(
        alpaka::BufCpu<T, Dim1D, Idx> const& A,
        alpaka::BufCpu<T, Dim1D, Idx> const& B,
        alpaka::BufCpu<T, Dim1D, Idx>& C)
    {
        assert(alpaka::getExtentProduct(A) == N * N);
        assert(alpaka::getExtentProduct(B) == N * N);
        assert(alpaka::getExtentProduct(C) == N * N);

        // Set alpha and beta for GEMM: C = alpha*A*B + beta*C
        const T alpha = 1;
        const T beta = 0;

        // Perform C = A × B using OpenBLAS (column-major order)
        cblas_sgemm(
            CblasColMajor, // Layout
            CblasNoTrans, // Transpose A? No
            CblasNoTrans, // Transpose B? No
            N, // m, n, k
            N,
            N,
            alpha, // alpha
            A.data(), // A and its leading dimension
            N,
            B.data(), // B and its leading dimension
            N,
            beta, // beta
            C.data(), // C and its leading dimension
            N);
    }
};

namespace traits
{

    template<>
    class Blas<alpaka::TagCpuSerial>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuOmp2Blocks>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuOmp2Threads>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuTbbBlocks>
    {
    public:
        using Impl = BlasCpu;
    };

    template<>
    class Blas<alpaka::TagCpuThreads>
    {
    public:
        using Impl = BlasCpu;
    };

} // namespace traits

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
