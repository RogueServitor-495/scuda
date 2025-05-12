#include <cublas_v2.h>
#include <cuda.h>
#include <nvml.h>
#include <cuda_runtime_api.h>

typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t, 
                                        int, int, int, const void*, const void*, cudaDataType_t, 
                                        int, const void*, cudaDataType_t, int, const void*, 
                                        void*, cudaDataType_t, int, cublasComputeType_t, 
                                        cublasGemmAlgo_t);

typedef cublasStatus_t (*cublasGemmBatchedEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
typedef cublasStatus_t (*cublasGemmStridedBatchedEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
typedef cublasStatus_t (*cublasGemmBatchedEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);



cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams* nodeParams);
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr);
cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream);
extern "C" void **__cudaRegisterFatBinary(void **fatCubin);
extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                struct CUstream_st *stream);
extern "C" unsigned __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                               size_t *sharedMem, void *stream);
extern "C" void __cudaInitModule(void **fatCubinHandle);
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const struct cudaHostNodeParams *pNodeParams);
cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemcpy3DParms *pCopyParams);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                              size_t *numNodes);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, 
                                        cublasLtMatmulPreferenceAttributes_t attr, 
                                        const void* buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, 
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, 
    const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, 
    const void* C, cublasLtMatrixLayout_t Cdesc, 
    void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, 
    void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);


cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, 
    cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, 
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, 
    cublasLtMatmulPreference_t preference, int requestedAlgoCount, 
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount);


cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, 
    cublasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes);
cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);

// cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
// cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount);


extern "C" void* dlopen(const char* filename, int flag);

