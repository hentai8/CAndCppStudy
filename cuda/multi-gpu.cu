#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
// This application demonstrates how to use CUDA API to use mutiple GPUs(4 Nvidia Geforce GTX 1080 ti)
// Function to add the elements of two arrays

// Mutiple-GPU Plan Structure
typedef struct
{
    // Host-side input data
    float *h_x, *h_y;

    // Result copied back from GPU
    float *h_yp;
    // Device buffers
    float *d_x, *d_y;

    // Stream for asynchronous command execution
    cudaStream_t stream;

} TGPUplan;

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *x, float *y)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int get_device_cores(int device_id);
void config_cuda(int device_id, int *grid_size, int *block_size, bool *is_inline_miner);
inline int get_sm_cores(int major, int minor);



int main(void)
{
    int N = 1 << 20; // 1M elements

    // Get the numble of CUDA-capble GPU

    // 获取一共有多少个GPU
    int N_GPU;
    cudaGetDeviceCount(&N_GPU);

    std::cout << "GPU COUNT:" << N_GPU << std::endl;
    // Arrange the task of each GPU
    int Np = (N + N_GPU - 1) / N_GPU;

    // Create GPU plans
    TGPUplan plan[N_GPU];

    // Initializing
    // 对于每个GPU，分别进行处理
    for (int i = 0; i < N_GPU; i++)
    {
        int id = i;
        int device_id = i;
        cudaSetDevice(i);
        cudaSetDevice(device_id);
        cudaStreamCreate(&plan[i].stream);
        int grid_size;
        int block_size;
        bool is_inline_miner;
        is_inline_miner = true;
        config_cuda(device_id, &grid_size, &block_size, &is_inline_miner);
        printf("Worker %d: device id %d, grid size %d, block size %d. Using %s kernel\n", id, device_id,
            grid_size, block_size, is_inline_miner ? "inline" : "reference");
        cudaMalloc((void **)&plan[i].d_x, Np * sizeof(float));
        cudaMalloc((void **)&plan[i].d_y, Np * sizeof(float));
        plan[i].h_x = (float *)malloc(Np * sizeof(float));
        plan[i].h_y = (float *)malloc(Np * sizeof(float));
        plan[i].h_yp = (float *)malloc(Np * sizeof(float));

        for (int j = 0; j < Np; j++)
        {
            plan[i].h_x[j] = 1.0f;
            plan[i].h_y[j] = 2.0f;
        }
    }

    int blockSize = 256;
    int numBlock = (Np + blockSize - 1) / blockSize;

    for (int i = 0; i < N_GPU; i++)
    {
        // Set device
        cudaSetDevice(i);

        // Copy input data from CPU
        cudaMemcpyAsync(plan[i].d_x, plan[i].h_x, Np * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);
        cudaMemcpyAsync(plan[i].d_y, plan[i].h_y, Np * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream);

        // Run the kernel function on GPU
        add<<<numBlock, blockSize, 0, plan[i].stream>>>(Np, plan[i].d_x, plan[i].d_y);

        // Read back GPU results
        cudaMemcpyAsync(plan[i].h_yp, plan[i].d_y, Np * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream);
    }

    // Process GPU results
    float y[N];
    for (int i = 0; i < N_GPU; i++)
    {
        // Set device
        cudaSetDevice(i);

        // Wait for all operations to finish
        cudaStreamSynchronize(plan[i].stream);

        // Get the final results
        for (int j = 0; j < Np; j++)
            if (Np * i + j < N)
                y[Np * i + j] = plan[i].h_yp[j];

        // shut down this GPU
        cudaFree(plan[i].d_x);
        cudaFree(plan[i].d_y);
        free(plan[i].h_x);
        free(plan[i].h_y);
        cudaStreamDestroy(plan[i].stream); // Destroy the stream
    }

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    return 0;
}

void config_cuda(int device_id, int *grid_size, int *block_size, bool *is_inline_miner)
{
    cudaSetDevice(device_id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    // If using a 2xxx or 3xxx card, use the new grid calc
    bool use_rtx_grid_bloc = ((props.major << 4) + props.minor) >= 0x75;

    // If compiling for windows, override the test and force the new calc
#ifdef _WIN32
    use_rtx_grid_bloc = true;
#endif

    // If compiling for linux, and we're not using the RTX grid block, force the original miner, otherwise use the inlined one
#ifdef __linux__
    *is_inline_miner = use_rtx_grid_bloc;
#else
    *is_inline_miner = true;
#endif
    // if(*is_inline_miner) {
    //     cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, inline_blake::blake3_hasher_mine);
    // } else {
    //     cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, ref_blake::blake3_hasher_mine);
    // }

    int cores_size = get_device_cores(device_id);
    if (use_rtx_grid_bloc)
    {
        *grid_size = props.multiProcessorCount * 2;
        *block_size = cores_size / *grid_size * 4;
    }
    else
    {
        *grid_size = cores_size / *block_size * 3 / 2;
    }
}

int get_device_cores(int device_id)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    int cores_size = get_sm_cores(props.major, props.minor) * props.multiProcessorCount;
    return cores_size;
}

// Beginning of GPU Architecture definitions
inline int get_sm_cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}