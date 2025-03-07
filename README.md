A simple implementation of Decoupled Lookback prefix sum algorithm from NVidia. 

The algorithm employs two-level hierarchical parallelism. It begins by distributing segments of data to thread blocks, and consists of 3 stages:

1. **Block Scan**: Each thread block independently computes the prefix sum of it's segment and publishes the sum total. The communication _between threads_ happens through fast shared memory. Since our focus is on second stage, I simply use cub::BlockScan for this stage.
2. **Decoupled Lookback**: Each block tries to find the sum of it's previous threadblocks. Communication _between blocks_ has to happen through (slower) global memory.
3. **Computing Prefix Sum**: Each thread simply sums up their prefix sum (from stage 1) with the sum of all previous blocks (stage 2) and stores it.

You can find more details [(here)](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back). The code (`decoupled_lookbach.cuh`) has lots of comment too.

Despite the simplicity, the implementation is pretty fast. Prefix sum is bottlenecked by how fast data can be copied from to device memory. So a good measure of performance is the percentage of peak bandwidth achieved in processing data.

Using this measure, *for this specific testcase*, we even outperform Nvidia's own implementations from cub or thrust libraries. On A100 SXM GPU, this implementation processes upto 1499.4 GB data per second, 73.5% of peak bandwidth. `cub::DeviceScan` processes 1438.98 GB/s, 70.5% of peak bandwidth. On my local 4060, the values are 80% and 75% respectively. The figures observed for cub (and thrust) is a bit lower than what I expected, so any suggestion regarding the benchmark code is welcome.

To compile, correct the configurations in Makefile based on your system, then simply do `make` from command line. Optionally, update device's `peak_bandwidth` variable in each file for above metric. 

### Known Issues
+ Only supports int for now. Does NOT deal with overflow issues. 
+ Silently crashes when (BLOCK_SIZE * ITEMS_PER_THREAD) is too high.

