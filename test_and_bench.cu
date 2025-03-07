#include <iostream>
#include <curand.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <cuda/atomic>
#include <thrust/random.h>

#include "decoupled_lookback.cuh"

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

void launch_kernel()
{
    // 268M elements. +5 to test non-power of 2 sizes.
    const int N = (1 << 28) + 5;

    const int ITEMS_PER_THREAD = 16;
    const int threads = (N + ITEMS_PER_THREAD -1) / ITEMS_PER_THREAD;
    const int BLOCK_SIZE = 128;
    const int GRID_SIZE = (threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //assert(N % (BLOCK_SIZE * ITEMS_PER_THREAD) == 0);

    printf("GRID_SIZE: %d\n", GRID_SIZE);
    printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("Total threads: %d\n", GRID_SIZE * BLOCK_SIZE);
    printf("Total elements: %d\n", N);
    double total_bytes = 2 * N * sizeof(int);
    double total_data_gb = total_bytes / (1 << 30);
    printf("Total data size: %f GB\n", total_data_gb);

    // generate random data vector on host using thrust
    thrust::default_random_engine rng(42);
    thrust::uniform_int_distribution<int> dist(-5, 5);
    thrust::host_vector<int> h_vec(N);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]
                     { return dist(rng); });

    // use thrust to compute result for later comparison
    thrust::host_vector<int> h_prefix_sum_tmp(N);
    thrust::inclusive_scan(h_vec.begin(), h_vec.end(), h_prefix_sum_tmp.begin());

    // allocate necessary device memory
    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_vector<int> d_prefix_sum(N);

    using ULL = unsigned long long int;
    thrust::device_vector<ULL> block_results(GRID_SIZE);
    thrust::device_vector<unsigned int> tmp_bid(1);
    CUDA_CHECK(cudaGetLastError());

    // Warm-up run
    thrust::fill(block_results.begin(), block_results.end(), 0);
    thrust::fill(tmp_bid.begin(), tmp_bid.end(), 0);
    prefixsum<BLOCK_SIZE, ITEMS_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(&d_vec[0]),
        thrust::raw_pointer_cast(&d_prefix_sum[0]),
        thrust::raw_pointer_cast(&block_results[0]),
        thrust::raw_pointer_cast(&tmp_bid[0]),
        N);

    CUDA_CHECK(cudaGetLastError());

    printf("Warm-up run done.\n");

    // check for correctness against thrust
    thrust::host_vector<int> h_prefix_sum(d_prefix_sum);

    // print final sums as quick test
    printf("Final elems of both prefix arrays:\n");
    printf("%d %d\n", h_prefix_sum[N - 1], h_prefix_sum_tmp[N - 1]);

    bool all_equal = thrust::equal(h_prefix_sum.begin(), h_prefix_sum.end(),
                                   h_prefix_sum_tmp.begin());
    if (all_equal)
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results don't match!" << std::endl;
        int res = 0;
        for (int i = 0; i < N; i++)
        {
            if (h_prefix_sum[i] != h_prefix_sum_tmp[i])
            {
                std::cout << "Mismatch at index: " << i << " " << h_prefix_sum[i] << " " << h_prefix_sum_tmp[i] << std::endl;
                res++;
            }
            if (res > 10)
                break;
        }
        return;
    }
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_time = 0.0f;
    const int NUM_ITERATIONS = 10;

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        // for now, kernels might reuse L2.
        thrust::fill(block_results.begin(), block_results.end(), 0);
        thrust::fill(tmp_bid.begin(), tmp_bid.end(), 0);

        CUDA_CHECK(cudaEventRecord(start));

        prefixsum<BLOCK_SIZE, ITEMS_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(&d_vec[0]),
            thrust::raw_pointer_cast(&d_prefix_sum[0]),
            thrust::raw_pointer_cast(&block_results[0]),
            thrust::raw_pointer_cast(&tmp_bid[0]),
            N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;

        CUDA_CHECK(cudaGetLastError());

    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Average prefixsum time over "
              << NUM_ITERATIONS << " iterations: "
              << (total_time / NUM_ITERATIONS) << " ms" << std::endl;

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
}

int main()
{
    launch_kernel();
    return 0;
}
