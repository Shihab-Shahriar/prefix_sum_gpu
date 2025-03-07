#include <iostream>
#include <cub/cub.cuh>
#include <cuda/atomic>

/**
 * @brief Kernel to compute prefix sum of a vector using a decoupled lookback approach.
 *
 * The kernel consists of 3 stages:
 * 1. Compute the prefix sum of each block using cub::BlockScan.
 * 2. Decoupled Lookback: to efficiently compute sum of all previous blocks.
 * 3. Compute the final prefix sum of each block.
 *
 * @tparam BLOCK_SIZE Number of threads in a block.
 * @tparam ITEMS_PER_THREAD Number of items processed by each thread.
 *
 * @param d_vec Input vector of type int.
 * @param d_prefix_sum Output vector.
 * @param block_results Intermediate results of each block. Must be initialized to 0.
 * @param DCounter Counter used to assign blockID.
 * @param N Size of the input vector.
 */
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void prefixsum(const int *__restrict__ d_vec,
                          int *__restrict__ d_prefix_sum,
                          unsigned long long *__restrict__ block_results,
                          unsigned int *DCounter, int N)
{
    __shared__ unsigned int sbid;
    if (threadIdx.x == 0)
    {
        sbid = atomicAdd(DCounter, 1);
    }
    __syncthreads();

    const int blockId = sbid;

    int thread_data[ITEMS_PER_THREAD];

    // Stage 1: Compute prefix sum of each block
    using BlockLoadT = cub::BlockLoad<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;
    using WarpReduce = cub::WarpReduce<int>;
    using ULL = unsigned long long int;
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockScanT::TempStorage scan;
    } shmem;
    __shared__ typename WarpReduce::TempStorage warp_storage;

    int blockOffset = blockId * (BLOCK_SIZE * ITEMS_PER_THREAD);
    BlockLoadT(shmem.load).Load(d_vec + blockOffset, thread_data, N - blockOffset);
    __syncthreads();

    int block_total;
    BlockScanT(shmem.scan).InclusiveSum(thread_data, thread_data, block_total);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // pack block total and flag into a single 64-bit integer
        // flag = 1: block_total is sum of just this block
        // flag = 2: block_total is sum of current and all previous blocks
        const int flag = blockId == 0 ? 2 : 1;
        auto blockMsg = (static_cast<ULL>(block_total) << 32) | flag;

        cuda::atomic_ref<ULL, cuda::thread_scope_device> atom_ref(block_results[blockId]);
        atom_ref.exchange(blockMsg, cuda::memory_order_release);
    }

    // Stage 2: Decoupled Lookback

    const unsigned FULL_MASK = 0xffffffff;

    __shared__ int prevBlocksSum;
    int partial_sums = 0;

    if (blockId > 0)
    {
        /**
         * Only first warp will participate in the decoupled lookback. We
         * will begin polling the status of 32 blocks preceding the current,
         * wait untill all of them has updated their block results- either
         * with just that block's sum or with the final prefix sum (flag=2).
         * If we find all blocks have only published with their own sum, we
         * will sum those values up and move to the next 32 blocks, untill we
         * find a block that has computed it's prefix sum.
         */
        if (threadIdx.x < 32)
        {
            int block_offset = blockId - 1;

            int block_sum;

            while (block_offset >= 0)
            {
                // my_block: block this thread will poll
                int my_block = block_offset - threadIdx.x; 
                unsigned mask = __ballot_sync(FULL_MASK, my_block >= 0);

                // threads that point to invalid block index still need to participate in warpReduce
                block_sum = 0;
                int found_incl_sum; // did any thread find inclusive sum?
                int winner_tid = -1;
                if (my_block >= 0)
                {
                    cuda::atomic_ref<ULL, cuda::thread_scope_device>
                        aref(block_results[my_block]);

                    ULL b_msg;
                    bool stop_polling = false;
                    do
                    {
                        b_msg = aref.load(cuda::memory_order_acquire);
                        int myblock_flag = b_msg & 0xFFFFFFFF;

                        stop_polling = myblock_flag > 0;

                        found_incl_sum = __ballot_sync(mask, myblock_flag == 2);
                        if (found_incl_sum != 0)
                        {
                            winner_tid = __ffs(found_incl_sum) - 1;
                            stop_polling = threadIdx.x > winner_tid || stop_polling;
                        }
                    } while (__any_sync(mask, stop_polling == false));

                    block_sum = static_cast<int>(b_msg >> 32);

                    if (found_incl_sum != 0)
                    { // found prefix sum for a block
                        if (threadIdx.x > winner_tid)
                            block_sum = 0;
                    }
                }
                int cur_aggregate = WarpReduce(warp_storage).Sum(block_sum);

                // partial_sums is undefined for all threads except 0
                partial_sums += cur_aggregate;
                if (found_incl_sum != 0)
                    break; // "opportunistic early break"
                block_offset -= 32;
            } // while
            if (threadIdx.x == 0)
                prevBlocksSum = partial_sums;
        } // if (threadIdx.x < 32)
        __syncthreads();
    } // if (blockId > 0)

    // Stage 3: Compute final prefix sum of each block
    int prevSum = blockId == 0 ? 0 : prevBlocksSum; // load from shmem

    if (threadIdx.x == 0 && blockId > 0)
    { // ready to upload my prefix sum (flag=2)
        int prefix_sum = block_total + prevSum;
        auto newBlockMsg = (static_cast<ULL>(prefix_sum) << 32) | 2;

        cuda::atomic_ref<ULL, cuda::thread_scope_device> atom_ref(block_results[blockId]);
        atom_ref.exchange(newBlockMsg, cuda::memory_order_release);
    }

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        thread_data[i] += prevSum;
    }
    BlockStoreT(shmem.store).Store(d_prefix_sum + blockOffset, thread_data, N - blockOffset);
}
