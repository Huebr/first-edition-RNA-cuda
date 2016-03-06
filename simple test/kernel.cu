
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//macros
#define MAX_FILENAME_SIZE 256
#define MAX_TEST_SIZE 1000

//solve the RNA prediction problem
cudaError_t solverRNA(const char *,int *);


__device__ bool canPair(int base1,int base2) {
	bool case1, case2;
	case1 = (base1 == 67 && base2 == 71 ) || (base1 == 71 && base2 == 67);
	case2 = (base1 == 65 && base2 == 85) || (base1 == 85 && base2 == 65);
	return (case1||case2);
}
__global__ void solverKernel(int *dev_data,int*dev_memo,int size)
{
     int i,j,opt;
	 i = threadIdx.x;
     for(int k = 5 ; k < size ;k++){ 
	    if(i<size-k){
		    j = i + k;
			dev_memo[size*i + j] = dev_memo[size*i + (j - 1)];
			for (int t = i; t < j - 4; t++) {     //opt(i,j)=max(opt(i,j-1),1+opt(i,t-1)+opt(t+1,j-1))
				if (canPair(dev_data[t], dev_data[j])) {
					if (t == 0) {
						opt = 1 + dev_memo[size*(t + 1)+(j-1)];
					}
					else {
						opt = 1 + dev_memo[i*size+(t-1)] + dev_memo[size*(t+1)+(j-1)];
					}
					if (opt > dev_memo[size*i+j]) {
						dev_memo[i*size+j] = opt;
					}
				}
			}
		}
		__syncthreads();
	 }
}

int main()
{
    FILE *input;
	char *filename;
	char testRNA[MAX_TEST_SIZE];
	int result;
	cudaError_t cudaStatus;

	//Memory Allocation to file name
	filename = (char*)malloc(MAX_FILENAME_SIZE*sizeof(char));

	//Reading filename
	printf("Write name of input file : ");
	scanf("%s", filename);

	//Open File to read input test data
	input = fopen(filename, "r");

	//Testing input opening
	if (input == NULL) {
		printf("Error opening file, please try again.");
		return 1;
	}

	printf("\n\n---------------- Begin Tests --------------------\n\n");

	//Begin reading file and testing
	while (fscanf(input, "%s",testRNA)!=EOF) {

	    //launch solverRNA
	    cudaStatus = solverRNA(testRNA,&result);
        if (cudaStatus != cudaSuccess) {
           fprintf(stderr, "solverRNA failed!");
           return 1;
        }

		printf("%s : ", testRNA);
		printf("%d base pairs.\n",result);
	}

	printf("\n\n---------------- Ending Tests --------------------\n\n");


   
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	system("pause");
    return 0;
}

// Helper function for using CUDA to solve RNA prediction in parallel with function objective maximum number of bases
cudaError_t solverRNA(const char *data,int *result)
{
    int *dev_data = 0;//data in device
    int *dev_memo = 0;//memotable in device
	int *host_memo = 0;//memotable in host
	int *host_data = 0;
	int size = strlen(data);
	const int size_memo = size*size;
    cudaError_t cudaStatus;

	//convert string to array of integers
	host_data = (int*)malloc(size*sizeof(int));
	for(int i = 0;i < size ;++i) host_data[i]=(int)data[i];
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	// Allocate	CPU buffer to memoTable
	host_memo = (int *)calloc(size_memo,sizeof(int));

    // Allocate GPU buffer to memoTable
    cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_memo, size_memo*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_memo, host_memo , size_memo * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, host_data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	solverKernel <<< 1, size >>> (dev_data, dev_memo, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "solverKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching solverKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_memo, dev_memo, size_memo*sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	*result=host_memo[size-1];
Error:
    cudaFree(dev_memo);
    cudaFree(dev_data);
    free(host_data);
	free(host_memo);

    return cudaStatus;
}
