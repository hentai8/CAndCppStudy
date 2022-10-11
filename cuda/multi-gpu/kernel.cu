#define imin(a,b) (a<b?a:b)
extern const int N = 33 * 1024 * 1024;
extern const int threadsPerBlock = 256;
extern const int blockPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);
 
__global__ void dotProduct(float *a, float *b, float *c, int N)
{
	__shared__ float cache[threadsPerBlock];
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheIdx = threadIdx.x;
 
	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIdx] = temp;
 
	__syncthreads();
 
	int i = blockDim.x /2;
	while (i != 0)
	{
		if (cacheIdx < i)
		{
			cache[cacheIdx] += cache[cacheIdx+i];
		}
		__syncthreads();
		i /= 2;
	}
 
	if (cacheIdx == 0)
	{
		c[blockIdx.x] = cache[0];
	}
 
}
 
extern "C" void runDotProduct(float *dev_a, float *dev_b, float *dev_partial_c, int size)
{
	dotProduct<<<blockPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c, size);
}