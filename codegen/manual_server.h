#ifndef __MANUAL_SERVER_H__
#define __MANUAL_SERVER_H__

#include "rpc.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>

int handle_cudaMallocHost(conn_t *conn);
int handle_cudaHostAlloc(conn_t *conn); 
int handle_cudaGraphAddKernelNode(conn_t *conn);
int handle_cudaGraphGetNodes(conn_t *conn);
int handle_cudaGraphAddMemAllocNode(conn_t *conn);
int handle_cudaDeviceGetGraphMemAttribute(conn_t *conn);
int handle_cudaGraphAddMemFreeNode(conn_t *conn);
int handle_cudaFree(conn_t *conn);
int handle_cudaMemcpy(conn_t *conn);
int handle_cudaMemcpyAsync(conn_t *conn);
int handle_cudaLaunchKernel(conn_t *conn);
int handle_cudaMallocManaged(conn_t *conn);
int handle___cudaRegisterVar(conn_t *conn);
int handle___cudaRegisterFunction(conn_t *conn);
int handle___cudaRegisterFatBinary(conn_t *conn);
int handle___cudaRegisterFatBinaryEnd(conn_t *conn);
int handle___cudaPushCallConfiguration(conn_t *conn);
int handle___cudaPopCallConfiguration(conn_t *conn);
int handle_cudaGraphAddHostNode(conn_t *conn);
int handle_cudaGraphAddMemcpyNode(conn_t *conn);
int handle_cudaGraphDestroy(conn_t *conn);
int handle_cudaLaunchKernel(conn_t *conn);
int handle_cublasLtMatmulPreferenceSetAttribute(conn_t *conn);
int handle_cublasLtMatmul(conn_t *conn);
int handle_cublasLtMatmulAlgoGetHeuristic(conn_t *conn);
int handle_cublasLtMatmulDescGetAttribute(conn_t *conn);
int handle_cublasLtMatmulDescSetAttribute(conn_t *conn);
int handle_cublasGemmBatchedEx(conn_t *conn);
int handle_cublasGemmBatchedEx_64(conn_t *conn);
int handle_cublasGemmEx_64(conn_t *conn);
int handle_cublasGemmEx(conn_t *conn);
int handle_cublasSgemmStridedBatched(conn_t *conn);
int handle_cublasSgemmStridedBatched_64(conn_t *conn);




#endif
