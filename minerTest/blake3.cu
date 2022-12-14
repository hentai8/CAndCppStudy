#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <stdio.h>

#define mining_steps 10

__constant__ uint8_t c_PaddedMessage[256];

__global__ void gpu_hello_world() {
    printf("hello world from GPU\n");
}

__constant__ uint8_t c_sigma[7][16];
static const uint8_t sigma[7][16] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
	{3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
	{10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
	{12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
	{9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
	{11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

#define G(i, j, a, b, c, d) \
        a = a + b + m[c_sigma[i][j * 2]]; \
        d = d ^ a; \
        d = d >> 16 | d << 16; \
        c = c + d; \
        b = b ^ c; \
        b = b >> 12 | b << 20; \
        a = a + b + m[c_sigma[i][j * 2 + 1]]; \
        d = d ^ a; \
        d = d >> 8 | d << 24; \
        c = c + d; \
        b = b ^ c; \
        b = b >> 7 | b << 25;

#define ROUND(i, v) \
        G(i, 0, v[0], v[4], v[8],  v[12]) \
        G(i, 1, v[1], v[5], v[9],  v[13]) \
        G(i, 2, v[2], v[6], v[10], v[14]) \
        G(i, 3, v[3], v[7], v[11], v[15]) \
        G(i, 4, v[0], v[5], v[10], v[15]) \
        G(i, 5, v[1], v[6], v[11], v[12]) \
        G(i, 6, v[2], v[7], v[8],  v[13]) \
        G(i, 7, v[3], v[4], v[9],  v[14])




void initCuda(){
    int count;
    cudaGetDeviceCount(&count);
    printf("cuda count: %d\n", count);
}


__host__ void blake3_host_setBlock(void *pdata, int len) {
	cudaMemcpyToSymbol( c_PaddedMessage, pdata, 256, 0, cudaMemcpyHostToDevice);
}

__host__ void blake3_cpu_init()
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol( c_sigma,
			sigma,
			sizeof(sigma),
			0, cudaMemcpyHostToDevice);
}


 __device__ void blake3_compress(uint32_t *out, const uint32_t m[], const uint32_t h[], uint64_t t, uint32_t b, uint32_t d )
{

	uint32_t v[16];
	int i;

	v[0] = h[0];
	v[1] = h[1];
	v[2] = h[2];
	v[3] = h[3];
	v[4] = h[4];
	v[5] = h[5];
	v[6] = h[6];
	v[7] = h[7];
	v[8] = 0x6a09e667;
	v[9] = 0xbb67ae85;
	v[10] = 0x3c6ef372;
	v[11] = 0xa54ff53a;
	v[12] = 0;
	v[13] = 0;
	v[14] = b;
	v[15] = d;


	ROUND(0, v)
	ROUND(1, v)
	ROUND(2, v)
	ROUND(3, v)
	ROUND(4, v)
	ROUND(5, v)
	ROUND(6, v)

	if (d & 0x1000) {
		for (i = 8; i < 16; ++i)
			out[i] = v[i] ^ h[i - 8];
	}
	for (i = 0; i < 8; ++i)
		out[i] = v[i] ^ v[i + 8];

}

__device__ void load(uint32_t d[], const unsigned char s[]) {
        uint32_t *end;

        for (end = d + 16; d < end; ++d, s += 4) {
                *d = (uint32_t)s[0]       | (uint32_t)s[1] <<  8
                   | (uint32_t)s[2] << 16 | (uint32_t)s[3] << 24;
        }
}




__global__ void blake3_gpu_hash() {
	uint32_t hash_count = 0;
	uint32_t iv[] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
	};
	int i;

    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t *nonce = &iv[8];
    *nonce = (*nonce) / stride * stride + tid;

	uint32_t m[16];

	load(m, c_PaddedMessage);

	while (hash_count < mining_steps)
    {
		printf("nonce: %d", *nonce);
        hash_count += 1;
		*nonce += stride;
		blake3_compress(iv, m, iv, 0, 64, 1);

		printf("ROUND1:\n");
		for(i = 0; i < 8; i++)
			printf("%08x ", iv[i]);
		printf("\n");

		load(m, c_PaddedMessage+64);
		blake3_compress(iv, m, iv, 0, 64, 0);

		load(m, c_PaddedMessage+128);
		blake3_compress(iv, m, iv, 0, 64, 0);

		load(m, c_PaddedMessage+192);
		blake3_compress(iv, m, iv, 0, 16, 10);

		printf("ROUND4:\n");
		for(i = 0; i < 8; i++)
			printf("%08x ", iv[i]);
		printf("\n");
	}
}




int main(int argc, char * argv[]) {
    initCuda();
    uint8_t pdata[256] = {
	    0x00,0x00,0x00,0x00,0x03,0xb6,0xb4,0x45,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	    0x69,0xe2,0x63,0xe9,0x31,0xfa,0x1a,0x2a,0x4b,0x04,0x37,0xa8,0xef,0xf7,0x9f,0xfb,
	    0x7a,0x35,0x3b,0x63,0x84,0xa7,0xae,0xac,0x9f,0x90,0xac,0x12,0xae,0x48,0x11,0xef,
	    0x48,0xea,0x87,0x92,0x83,0xaa,0xaf,0x39,0x45,0xa4,0xcf,0xf1,0x5d,0x8e,0x3e,0x3b,
	    0x91,0x74,0x83,0x08,0x44,0x40,0x94,0x03,0xca,0xbc,0x17,0x3a,0x4e,0x38,0x0b,0x45,
	    0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xe2,0x48,0x4d,0x0b,0xf3,0x8f,0x29,0xef,
	    0xfd,0x63,0xef,0x9d,0x5a,0x61,0x20,0x2f,0x19,0x81,0x29,0x86,0x2b,0x12,0x84,0x51,
	    0x82,0xa4,0xca,0x77,0xaa,0x55,0x7a,0x4b,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	    0x00,0x00,0x00,0x00,0x00,0x1d,0x93,0xc7,0x28,0xe3,0x75,0x4d,0x5f,0xe7,0xe2,0x44,
	    0xd1,0xe6,0x58,0x96,0x3a,0x42,0x10,0x1f,0xfe,0x31,0x1e,0xff,0xb8,0xcf,0xa8,0x86,
	    0xe0,0x2a,0x72,0x83,0x83,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x77,0x35,0x94,0x00,
	    0x49,0x72,0x6f,0x6e,0x20,0x46,0x69,0x73,0x68,0x20,0x50,0x6f,0x6f,0x6c,0x2e,0x31,
	    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
    };

    blake3_cpu_init();
    blake3_host_setBlock(pdata, 256);
    blake3_gpu_hash<<<2,2>>>();

    gpu_hello_world<<<1,1>>>();
    cudaDeviceReset();

//    cudaDeciveSynchronize();
   return 0;
}

