#include <stdio.h>

__global__
void helloFromGPU()
{
  printf("Hello World from GPU!\n");
​}
int main(void) {
  printf("Hello World from CPU!\n");
  helloFromGPU<<<1, 2>>>();
  cudaDeviceReset();
  return 0;
}