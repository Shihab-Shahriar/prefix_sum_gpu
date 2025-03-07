#include <iostream>
#include <cub/cub.cuh>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int main()
{
    const int N = (1 << 28);
    double total_bytes = 2 * N * sizeof(int);
    double total_data_gb = total_bytes / (1 << 30);
    printf("Total data size: %f GB\n", total_data_gb);

    thrust::default_random_engine rng(42);
    thrust::uniform_int_distribution<int> dist(-5, 5);
    thrust::host_vector<int> h_vec(N);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]
                     { return dist(rng); });

    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_vector<int> d_prefix_sum(N);
    CUDA_CHECK(cudaGetLastError());

    // warmup
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_vec.begin(), d_prefix_sum.begin(), N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_vec.begin(), d_prefix_sum.begin(), N);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_time = 0.0f;
    const int NUM_ITERATIONS = 10;

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        // for now, kernels might reuse L2.

        CUDA_CHECK(cudaEventRecord(start));

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_vec.begin(), d_prefix_sum.begin(), N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;

        CUDA_CHECK(cudaGetLastError());

    }

    std::cout << "Average prefixsum time over "
              << NUM_ITERATIONS << " iterations: "
              << (total_time / NUM_ITERATIONS) << " ms" << std::endl;

    // Cleanup CUDA events.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double avg_time_sec = total_time / NUM_ITERATIONS / 1000.0;
    const double achieved_bandwith = total_data_gb / avg_time_sec;
    const double peak_bandwidth = 272.0; // GB/s for 4060
    //const double peak_bandwidth = 2039.0; // GB/s for A100 80GB SXM
    const double efficiency = (achieved_bandwith / peak_bandwidth) * 100.0;

    // speed of light analysis. this kernel is bottlenecked by memory bandwidth
    // But how close are we to that limit?
    std::cout << "Achieved bandwidth: " << achieved_bandwith << " GB/s" << std::endl;
    std::cout << "GPU Peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << "Efficiency: " << efficiency << "%" << std::endl;

    return 0;
}
