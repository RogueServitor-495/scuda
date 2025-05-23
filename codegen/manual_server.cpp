#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>

#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"
#include "ptx_fatbin.hpp"

extern int rpc_read(const conn_t *conn, void *data, const std::size_t size);
extern int rpc_read_end(const conn_t *conn);
extern int rpc_write_start_response(const conn_t *conn, const int request_id);
extern int rpc_write(const conn_t *conn, const void *data,
                     const std::size_t size);

void invoke_host_func(void *data);
void append_managed_ptr(const void *conn, void *srcPtr, void *dstPtr,
                        size_t size, cudaMemcpyKind kind, void *graph);
void maybe_destroy_graph_resources(void *graph);

FILE *__cudart_trace_output_stream = stdout;

int handle_cudaMemcpy(conn_t *conn) {
  int request_id;
  cudaError_t result;
  void *src;
  void *dst;
  void *host_data;
  std::size_t count;
  enum cudaMemcpyKind kind;
  int ret = -1;

  if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
      (kind != cudaMemcpyHostToDevice &&
       rpc_read(conn, &src, sizeof(void *)) < 0) ||
      (kind != cudaMemcpyDeviceToHost &&
       rpc_read(conn, &dst, sizeof(void *)) < 0) ||
      rpc_read(conn, &count, sizeof(size_t)) < 0)
    goto ERROR_0;

  switch (kind) {
  case cudaMemcpyDeviceToHost:
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_1;

    result = cudaMemcpy(host_data, src, count, kind);
    break;
  case cudaMemcpyHostToDevice:
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    if (rpc_read(conn, host_data, count) < 0)
      goto ERROR_1;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_1;

    result = cudaMemcpy(dst, host_data, count, kind);
    break;
  case cudaMemcpyDeviceToDevice:
    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpy(dst, src, count, kind);
    break;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (kind == cudaMemcpyDeviceToHost &&
       rpc_write(conn, host_data, count) < 0) ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  ret = 0;
ERROR_1:
  if (host_data != NULL)
    free((void *)host_data);
ERROR_0:
  return ret;
}

int handle_cudaMemcpyAsync(conn_t *conn) {
  int request_id;
  cudaError_t result;
  void *src;
  void *dst;
  void *host_data = NULL;
  std::size_t count;
  enum cudaMemcpyKind kind;
  int stream_null_check;
  cudaStream_t stream = 0;
  int ret = -1;

  if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
      rpc_read(conn, &stream_null_check, sizeof(int)) < 0 ||
      (stream_null_check == 0 &&
       rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0) ||
      (kind != cudaMemcpyHostToDevice &&
       rpc_read(conn, &src, sizeof(void *)) < 0) ||
      (kind != cudaMemcpyDeviceToHost &&
       rpc_read(conn, &dst, sizeof(void *)) < 0) ||
      rpc_read(conn, &count, sizeof(size_t)) < 0)
    goto ERROR_0;

  switch (kind) {
  case cudaMemcpyDeviceToHost:
    host_data = malloc(count);
    printf("[DEBUG] allocated size=%d at [%p]...\n", count, host_data);
    if (host_data == NULL)
      goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpyAsync(host_data, src, count, kind, stream);
    printf("[DEBUG] copy size=%d from device=[%p] to host...\n", count, src);
    break;
  case cudaMemcpyHostToDevice:
    host_data = malloc(count);
    printf("[DEBUG] allocated size=%d at [%p]...\n", count, host_data);
    if (host_data == NULL)
      goto ERROR_0;

    if (rpc_read(conn, host_data, count) < 0) {
      goto ERROR_0;
    }

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpyAsync(dst, host_data, count, kind, stream);
    printf("[DEBUG] copy size=%d from host to device=[%p]...\n", count, src);
    break;
  case cudaMemcpyDeviceToDevice:
  printf("[DEBUG] allocated a void pointer at [%p]...\n", host_data);
    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpyAsync(dst, src, count, kind, stream);
    printf("[DEBUG] copy size=%d from device=[%p] to device=[%p]...\n", count, src, dst);
    break;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (kind == cudaMemcpyDeviceToHost &&
       rpc_write(conn, host_data, count) < 0) ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0 ||
      (host_data != NULL && stream == 0 &&
       cudaStreamAddCallback(
           stream,
           [](cudaStream_t stream, cudaError_t status, void *ptr) {
             printf("[streamCallback] start to free memory at [%p]...\n", ptr);
             free(ptr);
             printf("[streamCallback] free host memory finished at [%p]...\n", ptr);
           },
           host_data, 0) != cudaSuccess))
    goto ERROR_0;

  ret = 0;
ERROR_0:
  return ret;
}

int handle_cudaGraphAddKernelNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode = nullptr; // Initialize to nullptr
  cudaGraph_t graph;
  void **args;
  cudaKernelNodeParams pNodeParams = {0};
  std::vector<cudaGraphNode_t> dependencies;
  const cudaKernelNodeParams *pNodeParams_null_check;
  int request_id;
  int num_args;
  int arg_size;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0) {
    printf("Failed to read numDependencies\n");
    return -1;
  }

  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0) {
    printf("Failed to read graph\n");
    return -1;
  }

  dependencies.resize(numDependencies);
  for (size_t i = 0; i < numDependencies; ++i) {
    if (rpc_read(conn, &dependencies[i], sizeof(cudaGraphNode_t)) < 0) {
      printf("Failed to read Dependency[%zu]\n", i);
      return -1;
    }
  }

  if (rpc_read(conn, &num_args, sizeof(int)) < 0) {
    printf("Failed to read arg count\n");
    return -1;
  }

  args = (void **)malloc(num_args * sizeof(void *));
  if (args == NULL)
    return -1;

  for (int i = 0; i < num_args; ++i) {
    if (rpc_read(conn, &arg_size, sizeof(int)) < 0)
      return -1;

    args[i] = malloc(arg_size);
    if (args[i] == NULL)
      return -1;

    if (rpc_read(conn, args[i], arg_size) < 0)
      return -1;
  }

  if (rpc_read(conn, &pNodeParams_null_check,
               sizeof(const cudaKernelNodeParams *)) < 0) {
    return -1;
  }

  if (pNodeParams_null_check) {
    if (rpc_read(conn, &pNodeParams,
                 sizeof(const struct cudaKernelNodeParams)) < 0) {
      return -1;
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  // make sure we write our kernel args properly
  pNodeParams.kernelParams = args;

  scuda_intercept_result = cudaGraphAddKernelNode(
      &pGraphNode, graph, dependencies.data(), numDependencies, &pNodeParams);

  if (scuda_intercept_result != cudaSuccess) {
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }

  return 0;
}

int handle_cudaLaunchKernel(conn_t *conn) {
  int request_id;
  cudaError_t result;
  const void *func;
  void **args;
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;
  int num_args;
  int arg_size;

  if (rpc_read(conn, &func, sizeof(const void *)) < 0 ||
      rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_read(conn, &num_args, sizeof(int)) < 0)
    goto ERROR_0;

  // Allocate memory for the arguments
  args = (void **)malloc(num_args * sizeof(void *));
  if (args == NULL)
    goto ERROR_0;

  for (int i = 0; i < num_args; ++i) {
    if (rpc_read(conn, &arg_size, sizeof(int)) < 0)
      goto ERROR_1;

    // Allocate memory for the argument
    args[i] = malloc(arg_size);
    if (args[i] == NULL)
      goto ERROR_1;

    // Read the actual argument data from the client
    printf("receiving arg: [%d/%d], size=%d ...\n", i, num_args, arg_size);
    if (rpc_read(conn, args[i], arg_size) < 0)
      goto ERROR_1;
  }
  printf("read args end...\n");

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1; 

  result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

  std::cout << "Launch kern result: " << result << std::endl;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  for (int i = 0; i < num_args; ++i)
    if (args[i] != NULL)
      free(args[i]);
  free(args);
ERROR_0:
  return -1;
}

std::unordered_map<void **, __cudaFatCudaBinary2 *> fat_binary_map;

extern "C" void **__cudaRegisterFatBinary(void *fatCubin);

int handle___cudaRegisterFatBinary(conn_t *conn) {
  __cudaFatCudaBinary2 *fatCubin =
      (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
  unsigned long long size;

  if (rpc_read(conn, fatCubin, sizeof(__cudaFatCudaBinary2)) < 0 ||
      rpc_read(conn, &size, sizeof(unsigned long long)) < 0)
    return -1;

  void *cubin = malloc(size);
  if (rpc_read(conn, cubin, size) < 0)
    return -1;

  fatCubin->text = (uint64_t)cubin;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  void **p = __cudaRegisterFatBinary(fatCubin);

  fat_binary_map[p] = fatCubin;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &p, sizeof(void **)) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubin);

int handle___cudaUnregisterFatBinary(conn_t *conn) {
  void **fatCubin;
  if (rpc_read(conn, &fatCubin, sizeof(void **)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  free((void *)fat_binary_map[fatCubin]->text);
  free(fat_binary_map[fatCubin]);
  fat_binary_map.erase(fatCubin);

  __cudaUnregisterFatBinary(fatCubin);

  return 0;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize);

int handle___cudaRegisterFunction(conn_t *conn) {
  void **fatCubinHandle;
  char *hostFun;
  size_t deviceFunLen;
  size_t deviceNameLen;
  char *deviceFun;
  char *deviceName;
  int thread_limit;
  uint8_t mask;
  uint3 tid, bid;
  dim3 bDim, gDim;
  int wSize;

  int request_id;

  if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0 ||
      rpc_read(conn, &hostFun, sizeof(const char *)) < 0 ||
      rpc_read(conn, &deviceFunLen, sizeof(size_t)) < 0)
    goto ERROR_0;

  deviceFun = (char *)malloc(deviceFunLen);
  if (rpc_read(conn, deviceFun, deviceFunLen) < 0 ||
      rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
    goto ERROR_1;

  deviceName = (char *)malloc(deviceNameLen);
  if (rpc_read(conn, deviceName, deviceNameLen) < 0 ||
      rpc_read(conn, &thread_limit, sizeof(int)) < 0 ||
      rpc_read(conn, &mask, sizeof(uint8_t)) < 0 ||
      (mask & 1 << 0 && rpc_read(conn, &tid, sizeof(uint3)) < 0) ||
      (mask & 1 << 1 && rpc_read(conn, &bid, sizeof(uint3)) < 0) ||
      (mask & 1 << 2 && rpc_read(conn, &bDim, sizeof(dim3)) < 0) ||
      (mask & 1 << 3 && rpc_read(conn, &gDim, sizeof(dim3)) < 0) ||
      (mask & 1 << 4 && rpc_read(conn, &wSize, sizeof(int)) < 0))
    goto ERROR_2;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_2;

  __cudaRegisterFunction(
      fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
      mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr,
      mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr,
      mask & 1 << 4 ? &wSize : nullptr);

  if (rpc_write_start_response(conn, request_id) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
ERROR_2:
  free((void *)deviceName);
ERROR_1:
  free((void *)deviceFun);
ERROR_0:
  return -1;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);

int handle___cudaRegisterFatBinaryEnd(conn_t *conn) {
  void **fatCubinHandle;

  if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  __cudaRegisterFatBinaryEnd(fatCubinHandle);

  if (rpc_write_start_response(conn, request_id) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

// Function pointer type for __cudaPushCallConfiguration
extern "C" unsigned int __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                    size_t sharedMem,
                                                    cudaStream_t stream);

int handle___cudaPushCallConfiguration(conn_t *conn) {
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;

  // Read the grid dimensions from the client
  if (rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  unsigned int result =
      __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  void *stream);

int handle___cudaPopCallConfiguration(conn_t *conn) {
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  cudaError_t result =
      __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

// typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

// void __cudaInitModule(void **fatCubinHandle) {
//   std::cerr << "calling __cudaInitModule" << std::endl;
// }

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);

int handle___cudaRegisterVar(conn_t *conn) {
  void **fatCubinHandle;
  char *hostVar;
  char *deviceAddress;
  char *deviceName;
  int ext;
  size_t size;
  int constant;
  int global;

  // Read the fatCubinHandle
  if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0) {
    std::cerr << "Failed reading fatCubinHandle" << std::endl;
    return -1;
  }

  // Read hostVar
  size_t hostVarLen;
  if (rpc_read(conn, &hostVarLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to read hostVar length" << std::endl;
    return -1;
  }
  hostVar = (char *)malloc(hostVarLen);
  if (rpc_read(conn, hostVar, hostVarLen) < 0) {
    std::cerr << "Failed to read hostVar" << std::endl;
    return -1;
  }

  // Read deviceAddress
  size_t deviceAddressLen;
  if (rpc_read(conn, &deviceAddressLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to read deviceAddress length" << std::endl;
    return -1;
  }
  deviceAddress = (char *)malloc(deviceAddressLen);
  if (rpc_read(conn, deviceAddress, deviceAddressLen) < 0) {
    std::cerr << "Failed to read deviceAddress" << std::endl;
    return -1;
  }

  // Read deviceName
  size_t deviceNameLen;
  if (rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to read deviceName length" << std::endl;
    return -1;
  }
  deviceName = (char *)malloc(deviceNameLen);
  if (rpc_read(conn, deviceName, deviceNameLen) < 0) {
    std::cerr << "Failed to read deviceName" << std::endl;
    return -1;
  }

  // Read ext, size, constant, global
  if (rpc_read(conn, &ext, sizeof(int)) < 0) {
    std::cerr << "Failed reading ext" << std::endl;
    return -1;
  }

  if (rpc_read(conn, &size, sizeof(size_t)) < 0) {
    std::cerr << "Failed reading size" << std::endl;
    return -1;
  }

  if (rpc_read(conn, &constant, sizeof(int)) < 0) {
    std::cerr << "Failed reading constant" << std::endl;
    return -1;
  }

  if (rpc_read(conn, &global, sizeof(int)) < 0) {
    std::cerr << "Failed reading global" << std::endl;
    return -1;
  }

  std::cout << "Received __cudaRegisterVar with deviceName: " << deviceName
            << std::endl;

  __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                    size, constant, global);

  // End request phase
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    std::cerr << "rpc_read_end failed" << std::endl;
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

int handle_cudaFree(conn_t *conn) {
  void *devPtr;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &devPtr, sizeof(void *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaFree(devPtr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaMallocManaged(conn_t *conn) {
  void *devPtr;
  size_t size;
  unsigned int flags;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &devPtr, sizeof(void *)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaMallocManaged(&devPtr, size, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &devPtr, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaMallocHost(conn_t *conn) { return 0; }

int handle_cudaHostAlloc(conn_t *conn) {return 0;}

int handle_cudaGraphGetNodes(conn_t *conn) {
  cudaGraph_t graph;
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaGraphGetNodes(graph, nodes, &numNodes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &nodes, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &numNodes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

typedef struct callBackData {
  conn_t *conn;
  void (*callback)(void *);
  void *data;
} callBackData_t;

int handle_cudaGraphAddHostNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaHostNodeParams *pNodeParams_null_check;
  struct cudaHostNodeParams pNodeParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  callBackData_t *hostFnData;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
          rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
          [=, &pDependencies]()
          -> bool {
        pDependencies.resize(numDependencies); // Resize the dependencies vector
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_read(conn, &pDependencies[i], sizeof(const cudaGraphNode_t)) <
              0) {
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_read(conn, &pNodeParams_null_check,
                          sizeof(const struct cudaHostNodeParams *)) < 0 ||
                 (pNodeParams_null_check &&
                  rpc_read(conn, &pNodeParams,
                           sizeof(const struct cudaHostNodeParams)) < 0) ||
                 false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  hostFnData = (callBackData_t *)malloc(sizeof(callBackData_t));
  // assign the previous function pointer so we can map back to it
  hostFnData->callback = pNodeParams.fn;
  hostFnData->data = pNodeParams.userData;
  hostFnData->conn = conn;

  pNodeParams.fn = invoke_host_func;
  pNodeParams.userData = hostFnData;

  scuda_intercept_result = cudaGraphAddHostNode(
      &pGraphNode, graph, pDependencies.data(), numDependencies, &pNodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemcpyNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaMemcpy3DParms *pCopyParams_null_check;
  struct cudaMemcpy3DParms pCopyParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
          [=, &pDependencies]()
          -> bool {
        pDependencies.resize(numDependencies); // Resize the dependencies vector
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_read(conn, &pDependencies[i], sizeof(const cudaGraphNode_t)) <
              0) {
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_read(conn, &pCopyParams_null_check,
                          sizeof(const struct cudaMemcpy3DParms *)) < 0 ||
                 (pCopyParams_null_check &&
                  rpc_read(conn, &pCopyParams,
                           sizeof(const struct cudaMemcpy3DParms)) < 0) ||
                 false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  append_managed_ptr(conn, (void *)pCopyParams.srcPtr.ptr,
                     (void *)pCopyParams.dstPtr.ptr, pCopyParams.extent.width,
                     pCopyParams.kind, graph);

  scuda_intercept_result = cudaGraphAddMemcpyNode(
      &pGraphNode, graph, pDependencies.data(), numDependencies, &pCopyParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphDestroy(conn_t *conn) {
  cudaGraph_t graph;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaGraphDestroy(graph);

  maybe_destroy_graph_resources(graph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemFreeNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraphNode_t pGraphNodeP;
  cudaGraph_t graph;
  cudaGraphNode_t *singleDependency = nullptr;
  std::vector<cudaGraphNode_t> pDependencies;
  void *dptr;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
    goto ERROR_0;

  if (numDependencies == 1) {
    if (rpc_read(conn, &pGraphNodeP, sizeof(cudaGraphNode_t)) < 0) {
      goto ERROR_1;
    }
  } else if (numDependencies > 1) {
    pDependencies.resize(numDependencies);
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_read(conn, &pDependencies[i], sizeof(cudaGraphNode_t)) < 0) {
        goto ERROR_0;
      }
    }
  }

  if (rpc_read(conn, &dptr, sizeof(void *)) < 0)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;

  scuda_intercept_result = cudaGraphAddMemFreeNode(
      &pGraphNode, graph,
      (numDependencies == 1) ? &pGraphNodeP : pDependencies.data(),
      numDependencies, dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  if (singleDependency)
    free(singleDependency);

  return 0;

ERROR_1:
  if (singleDependency)
    free(singleDependency);
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemAllocNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraphNode_t pGraphNodeP;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaMemAllocNodeParams *nodeParams_null_check;
  struct cudaMemAllocNodeParams nodeParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
      rpc_read(conn, &nodeParams_null_check,
               sizeof(struct cudaMemAllocNodeParams *)) < 0 ||
      (nodeParams_null_check &&
       rpc_read(conn, &nodeParams, sizeof(struct cudaMemAllocNodeParams)) <
           0) ||
      false)
    goto ERROR_0;

  if (numDependencies == 1) {
    if (rpc_read(conn, &pGraphNodeP, sizeof(cudaGraphNode_t)) < 0) {
      goto ERROR_0;
    }
  } else if (numDependencies > 1) {
    pDependencies.resize(numDependencies);
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_read(conn, &pDependencies[i], sizeof(cudaGraphNode_t)) < 0) {
        goto ERROR_0;
      }
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaGraphAddMemAllocNode(
      &pGraphNode, graph,
      (numDependencies == 1) ? &pGraphNodeP : pDependencies.data(),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &nodeParams.dptr, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaDeviceGetGraphMemAttribute(conn_t *conn) {
  int device;
  enum cudaGraphMemAttributeType attr;
  void *value;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &device, sizeof(int)) < 0 ||
      rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaDeviceGetGraphMemAttribute(device, attr, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cublasLtMatmulPreferenceSetAttribute(conn_t *conn)
{
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceAttributes_t attr;
    size_t sizeInBytes;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    size_t workspaceSize;
    if (
        rpc_read(conn, &pref, sizeof(cublasLtMatmulPreference_t)) < 0 ||
        rpc_read(conn, &attr, sizeof(cublasLtMatmulPreferenceAttributes_t)) < 0 ||
        rpc_read(conn, &workspaceSize, sizeof(size_t)) < 0 ||
        rpc_read(conn, &sizeInBytes, sizeof(size_t)) < 0 ||
        false)
        goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    scuda_intercept_result = cublasLtMatmulPreferenceSetAttribute(pref, attr, &workspaceSize, sizeInBytes);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cublasLtMatmul(conn_t *conn)
{
    cublasLtHandle_t lightHandle;
    cublasLtMatmulDesc_t computeDesc;
    void* alpha = malloc(sizeof(const void*));
    const void* A;
    cublasLtMatrixLayout_t Adesc;
    const void* B;
    cublasLtMatrixLayout_t Bdesc;
    void* beta = malloc(sizeof(const void*));
    const void* C;
    cublasLtMatrixLayout_t Cdesc;
    void* D;
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatmulAlgo_t algo;
    // void* algo = (void *)malloc(sizeof(cublasLtMatmulAlgo_t));
    void* workspace;
    size_t workspaceSizeInBytes;
    cudaStream_t stream;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &lightHandle, sizeof(cublasLtHandle_t)) < 0 ||
        rpc_read(conn, &computeDesc, sizeof(cublasLtMatmulDesc_t)) < 0)
        goto ERROR_0;

    // parse the bytes of alpha and beta
    // size_t maxBytes = 8;
    // size_t nBytes;
    cudaDataType_t scaleType;
    cublasLtMatmulDescGetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                  &scaleType, sizeof(scaleType), nullptr);
    // nBytes = dataTypeSize(scaleType);


    if (
        rpc_read(conn, alpha, sizeof(const void*)) < 0 ||
        rpc_read(conn, &A, sizeof(const void*)) < 0 ||
        rpc_read(conn, &Adesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &B, sizeof(const void*)) < 0 ||
        rpc_read(conn, &Bdesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, beta, sizeof(const void*)) < 0 ||
        rpc_read(conn, &C, sizeof(const void*)) < 0 ||
        rpc_read(conn, &Cdesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &D, sizeof(void*)) < 0 ||
        rpc_read(conn, &Ddesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &algo, sizeof(cublasLtMatmulAlgo_t)) < 0 ||
        rpc_read(conn, &workspace, sizeof(void*)) < 0 ||
        rpc_read(conn, &workspaceSizeInBytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
        false)
        goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    // scuda_intercept_result = cublasLtMatmul(lightHandle, computeDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
    scuda_intercept_result = cublasLtMatmul(lightHandle, computeDesc, 
      alpha, A, Adesc, 
      B, Bdesc, beta, 
      C, Cdesc, 
      D, Ddesc, 
      &algo, workspace, workspaceSizeInBytes, stream);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cublasLtMatmulAlgoGetHeuristic(conn_t *conn)
{
    cublasLtHandle_t lightHandle;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatmulPreference_t preference;
    int requestedAlgoCount;
    cublasLtMatmulHeuristicResult_t heuristicResultsArray;
    int returnAlgoCount;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &lightHandle, sizeof(cublasLtHandle_t)) < 0 ||
        rpc_read(conn, &operationDesc, sizeof(cublasLtMatmulDesc_t)) < 0 ||
        rpc_read(conn, &Adesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &Bdesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &Cdesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &Ddesc, sizeof(cublasLtMatrixLayout_t)) < 0 ||
        rpc_read(conn, &preference, sizeof(cublasLtMatmulPreference_t)) < 0 ||
        rpc_read(conn, &requestedAlgoCount, sizeof(int)) < 0 ||
        false)
        goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    scuda_intercept_result = cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, &heuristicResultsArray, &returnAlgoCount);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &returnAlgoCount, sizeof(int)) < 0 ||
        rpc_write(conn, &heuristicResultsArray, returnAlgoCount * sizeof(cublasLtMatmulHeuristicResult_t)) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        // rpc_write(conn, &heuristicResultsArray.algo, sizeof(cublasLtMatmulAlgo_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cublasLtMatmulDescGetAttribute(conn_t *conn)
{
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescAttributes_t attr;
    void* buf;
    size_t sizeInBytes;
    size_t* sizeWrittenCheck;
    size_t sizeWritten;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &matmulDesc, sizeof(cublasLtMatmulDesc_t)) < 0 ||
        rpc_read(conn, &attr, sizeof(cublasLtMatmulDescAttributes_t)) < 0 ||
        false)
        goto ERROR_0;
    std::cout << "writtenCheck size:" << sizeof(sizeWrittenCheck) << std::endl;
    if(        rpc_read(conn, &sizeInBytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &sizeWrittenCheck, sizeof(sizeWrittenCheck)) < 0 ||
        (sizeWrittenCheck && rpc_read(conn, sizeWrittenCheck, sizeof(size_t)) < 0) ||
        false)
        goto ERROR_1;
        
    buf = (void*)malloc(sizeInBytes);
    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_1;
    scuda_intercept_result = cublasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes, &sizeWritten);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, buf, sizeInBytes) < 0 ||
        (sizeWrittenCheck && rpc_write(conn, &sizeWritten, sizeof(size_t)) < 0) ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) buf);
ERROR_0:
    return -1;
}

int handle_cublasLtMatmulDescSetAttribute(conn_t *conn)
{
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescAttributes_t attr;
    void* buf;
    size_t sizeInBytes;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &matmulDesc, sizeof(cublasLtMatmulDesc_t)) < 0 ||
        rpc_read(conn, &attr, sizeof(cublasLtMatmulDescAttributes_t)) < 0 ||
        rpc_read(conn, &sizeInBytes, sizeof(size_t)) < 0 ||
        false)
        goto ERROR_0;
    buf = malloc(sizeInBytes);
    if (rpc_read(conn, buf, sizeInBytes) < 0){
      goto ERROR_0;
    }
 
    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    scuda_intercept_result = cublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);
        if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cublasGemmBatchedEx(conn_t *conn)
{
    int batchCount;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    void* alpha_null_check;
    void* alpha;
    void** Aarray = nullptr;
    cudaDataType Atype;
    int lda;
    void** Barray = nullptr;
    cudaDataType Btype;
    int ldb;
    void* beta_null_check;
    void* beta;
    void** Carray = nullptr;
    cudaDataType Ctype;
    int ldc;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algo;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &batchCount, sizeof(int)) < 0 ||
        rpc_read(conn, &handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_read(conn, &transa, sizeof(cublasOperation_t)) < 0 ||
        rpc_read(conn, &transb, sizeof(cublasOperation_t)) < 0 ||
        rpc_read(conn, &m, sizeof(int)) < 0 ||
        rpc_read(conn, &n, sizeof(int)) < 0 ||
        rpc_read(conn, &k, sizeof(int)) < 0 ||
        rpc_read(conn, &alpha_null_check, sizeof(const void*)) < 0 ||
        (alpha_null_check && rpc_read(conn, &alpha, sizeof(const void*)) < 0) ||
        false)
        goto ERROR_0;
    Aarray = (void**)malloc(batchCount * sizeof(void*));
    if( rpc_read(conn, Aarray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Atype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &lda, sizeof(int)) < 0 ||
        false)
        goto ERROR_0;
    Barray = (void**)malloc(batchCount * sizeof(void*));
    if(rpc_read(conn, Barray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Btype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &ldb, sizeof(int)) < 0 ||
        rpc_read(conn, &beta_null_check, sizeof(const void*)) < 0 ||
        (beta_null_check && rpc_read(conn, &beta, sizeof(const void*)) < 0) ||
        false)
        goto ERROR_0;
    Carray = (void**)malloc(batchCount * sizeof(void*));
    if(rpc_read(conn, Carray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Ctype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &ldc, sizeof(int)) < 0 ||
        rpc_read(conn, &computeType, sizeof(cublasComputeType_t)) < 0 ||
        rpc_read(conn, &algo, sizeof(cublasGemmAlgo_t)) < 0 ||
        false)
        goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    scuda_intercept_result = cublasGemmBatchedEx_64(handle, transa, transb, m, n, k, &alpha, Aarray, Atype, lda, Barray, Btype, ldb, &beta, Carray, Ctype, ldc, batchCount, computeType, algo);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    free(Aarray);
    free(Barray);
    free(Carray);
    return 0;
ERROR_0:
    return -1;
}


int handle_cublasGemmBatchedEx_64(conn_t *conn)
{
    int64_t batchCount;
    cublasHandle_t handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int64_t m;
    int64_t n;
    int64_t k;
    void* alpha_null_check;
    void* alpha;
    void** Aarray = nullptr;
    cudaDataType Atype;
    int64_t lda;
    void** Barray = nullptr;
    cudaDataType Btype;
    int64_t ldb;
    void* beta_null_check;
    void* beta;
    void** Carray = nullptr;
    cudaDataType Ctype;
    int64_t ldc;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algo;
    int request_id;
    cublasStatus_t scuda_intercept_result;
    if (
        rpc_read(conn, &batchCount, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_read(conn, &transa, sizeof(cublasOperation_t)) < 0 ||
        rpc_read(conn, &transb, sizeof(cublasOperation_t)) < 0 ||
        rpc_read(conn, &m, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &n, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &k, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &alpha_null_check, sizeof(const void*)) < 0 ||
        (alpha_null_check && rpc_read(conn, &alpha, sizeof(const void*)) < 0) ||
        false)
        goto ERROR_0;
    Aarray = (void**)malloc(batchCount * sizeof(void*));
    if( rpc_read(conn, Aarray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Atype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &lda, sizeof(int64_t)) < 0 ||
        false)
        goto ERROR_0;
    Barray = (void**)malloc(batchCount * sizeof(void*));
    if(rpc_read(conn, Barray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Btype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &ldb, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &beta_null_check, sizeof(const void*)) < 0 ||
        (beta_null_check && rpc_read(conn, &beta, sizeof(const void*)) < 0) ||
        false)
        goto ERROR_0;
    Carray = (void**)malloc(batchCount * sizeof(void*));
    if(rpc_read(conn, Carray, batchCount * sizeof(void*)) < 0 ||
        rpc_read(conn, &Ctype, sizeof(cudaDataType)) < 0 ||
        rpc_read(conn, &ldc, sizeof(int64_t)) < 0 ||
        rpc_read(conn, &computeType, sizeof(cublasComputeType_t)) < 0 ||
        rpc_read(conn, &algo, sizeof(cublasGemmAlgo_t)) < 0 ||
        false)
        goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
        goto ERROR_0;
    scuda_intercept_result = cublasGemmBatchedEx_64(handle, transa, transb, m, n, k, &alpha, Aarray, Atype, lda, Barray, Btype, ldb, &beta, Carray, Ctype, ldc, batchCount, computeType, algo);

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &scuda_intercept_result, sizeof(cublasStatus_t)) < 0 ||
        rpc_write_end(conn) < 0)
        goto ERROR_0;

    free(Aarray);
    free(Barray);
    free(Carray);
    return 0;
ERROR_0:
    return -1;
}

