// #include "main.h"
#include <stdio.h>
 
extern "C" void runDotProduct(float *dev_a, float *dev_b, float *dev_partial_c, int size);
 
void* worker(void *pvoidData)
{
	GPUPlan *plan = (GPUPlan*) pvoidData;
	HANDLE_ERROR(cudaSetDevice(plan->deviceID));
 
	int size = plan->size;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
 
	a = plan->a;
	b = plan->b;
	partial_c = (float*)malloc(blockPerGrid*sizeof(float));
 
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, size*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, size*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blockPerGrid*sizeof(float)));
 
	HANDLE_ERROR(cudaMemcpy(dev_a, a, size*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice));
 
	runDotProduct(dev_a, dev_b, dev_partial_c, size);
 
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
 
	c = 0;
	for (int i=0; i<blockPerGrid; i++)
	{
		c += partial_c[i];
	}
 
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));
 
	free(partial_c);
	plan->returnValue = c;
	return 0;
}
 
 
 
int main()
{
	//on two GPUs
	int i;
	int deviceCount;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
 
	if (deviceCount < 2)
	{
		printf("No more than 2 device with compute 1.0 or greater."
			"only %d devices found", deviceCount);
		return 0;
	}
 
	float *a = (float*)malloc(sizeof(float)*N);
	HANDLE_NULL(a);
	float *b = (float*)malloc(sizeof(float)*N);
	HANDLE_NULL(b);
 
	for (i=0; i<N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}
 
	GPUPlan plan[2];
	plan[0].deviceID = 0;
	plan[0].size = N/2;
	plan[0].a = a;
	plan[0].b = b;
 
	plan[1].deviceID = 1;
	plan[1].size = N/2;
	plan[1].a = a + N/2;
	plan[1].b = b + N/2;
 
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	float elapsedTime;
 
	HANDLE_ERROR(cudaEventRecord(start));
	
	CUTThread mythread1 = start_thread((CUT_THREADROUTINE)worker, &plan[0]);
	CUTThread mythread2 = start_thread((CUT_THREADROUTINE)worker, &plan[1]);
	//worker(&plan[1]);
 
	end_thread(mythread1);
	end_thread(mythread2);
 
	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));
 
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
		printf("Computing by 2 GPUs finished in %3.1f <ms>\n", elapsedTime);
 
	printf("value calculated: %f\n", plan[0].returnValue + plan[1].returnValue);
 
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	free(a);
	free(b);
 
	// on one GPU
	float *host_a;
	float *host_b;
	float *partial_c;
	host_a = (float*)malloc(N*sizeof(float));
	host_b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blockPerGrid*sizeof(float));
 
	for (int i=0; i<N; i++)
	{
		host_a[i] = i;
		host_b[i] = 2 * i;
	}
 
	float *dev_a, *dev_b, *dev_partial_c;
 
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blockPerGrid*sizeof(float)));
 
	HANDLE_ERROR(cudaMemcpy(dev_a, host_a, N*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, host_b, N*sizeof(float), cudaMemcpyHostToDevice));
 
	cudaEvent_t start1, stop1;
	HANDLE_ERROR(cudaEventCreate(&start1));
	HANDLE_ERROR(cudaEventCreate(&stop1));
 
	HANDLE_ERROR(cudaEventRecord(start1));
	runDotProduct(dev_a, dev_b, dev_partial_c, N);
	HANDLE_ERROR(cudaEventRecord(stop1));
	HANDLE_ERROR(cudaEventSynchronize(stop1));
 
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start1, stop1));
	printf("Computing by one GPU finished in %3.1f <ms>\n", elapsedTime);
 
	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
 
	float res = 0;
	for (int i=0; i<blockPerGrid; i++)
	{
		res += partial_c[i];
	}
 
	printf("value calculated: %f\n", res);
 
	HANDLE_ERROR(cudaEventDestroy(start1));
	HANDLE_ERROR(cudaEventDestroy(stop1));
 
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));
 
	free(host_a);
	free(host_b);
	free(partial_c);
 
	return 0;
}