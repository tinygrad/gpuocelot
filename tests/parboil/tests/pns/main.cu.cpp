# 1 "/tmp/tmpxft_00006e97_00000000-1_main.cudafe1.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/tmp/tmpxft_00006e97_00000000-1_main.cudafe1.cpp"
# 1 "main.cu"
# 233 "/usr/include/c++/4.3/x86_64-linux-gnu/bits/c++config.h" 3
namespace std __attribute__((visibility("default"))) {
# 245 "/usr/include/c++/4.3/x86_64-linux-gnu/bits/c++config.h" 3
}
# 46 "/usr/local/cuda/bin/../include/device_types.h"
# 152 "/usr/lib/gcc/x86_64-linux-gnu/4.3.3/include/stddef.h" 3
typedef long ptrdiff_t;
# 214 "/usr/lib/gcc/x86_64-linux-gnu/4.3.3/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 1 3
# 80 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 3
# 1 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 1 3
# 53 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
# 1 "/usr/local/cuda/bin/../include/host_defines.h" 1 3
# 54 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 2 3
# 1 "/usr/local/cuda/bin/../include/builtin_types.h" 1 3
# 42 "/usr/local/cuda/bin/../include/builtin_types.h" 3
# 1 "/usr/local/cuda/bin/../include/device_types.h" 1 3
# 46 "/usr/local/cuda/bin/../include/device_types.h" 3
enum cudaRoundMode
{
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
};
# 43 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/driver_types.h" 1 3
# 90 "/usr/local/cuda/bin/../include/driver_types.h" 3
enum cudaError
{
  cudaSuccess = 0,
  cudaErrorMissingConfiguration = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorPriorLaunchFailure = 5,
  cudaErrorLaunchTimeout = 6,
  cudaErrorLaunchOutOfResources = 7,
  cudaErrorInvalidDeviceFunction = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidValue = 11,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorMapBufferObjectFailed = 14,
  cudaErrorUnmapBufferObjectFailed = 15,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorAddressOfConstant = 22,
  cudaErrorTextureFetchFailed = 23,
  cudaErrorTextureNotBound = 24,
  cudaErrorSynchronizationError = 25,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorMixedDeviceExecution = 28,
  cudaErrorCudartUnloading = 29,
  cudaErrorUnknown = 30,
  cudaErrorNotYetImplemented = 31,
  cudaErrorMemoryValueTooLarge = 32,
  cudaErrorInvalidResourceHandle = 33,
  cudaErrorNotReady = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorSetOnActiveProcess = 36,
  cudaErrorNoDevice = 38,
  cudaErrorStartupFailure = 0x7f,
  cudaErrorApiFailureBase = 10000
};





enum cudaChannelFormatKind
{
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3,
};





struct cudaChannelFormatDesc
{
  int x;
  int y;
  int z;
  int w;
  enum cudaChannelFormatKind f;
};





struct cudaArray;





enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};





struct cudaPitchedPtr
{
  void *ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};





struct cudaExtent
{
  size_t width;
  size_t height;
  size_t depth;
};





struct cudaPos
{
  size_t x;
  size_t y;
  size_t z;
};





struct cudaMemcpy3DParms
{
  struct cudaArray *srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;

  struct cudaArray *dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;

  struct cudaExtent extent;
  enum cudaMemcpyKind kind;
};





struct cudaFuncAttributes
{
   size_t sharedSizeBytes;
   size_t constSizeBytes;
   size_t localSizeBytes;
   int maxThreadsPerBlock;
   int numRegs;
   int __cudaReserved[8];
};





enum cudaComputeMode
{
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2
};






struct cudaDeviceProp
{
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int __cudaReserved[36];
};
# 318 "/usr/local/cuda/bin/../include/driver_types.h" 3
typedef enum cudaError cudaError_t;





typedef int cudaStream_t;





typedef int cudaEvent_t;
# 44 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/texture_types.h" 1 3
# 54 "/usr/local/cuda/bin/../include/texture_types.h" 3
enum cudaTextureAddressMode
{
  cudaAddressModeWrap,
  cudaAddressModeClamp
};


enum cudaTextureFilterMode
{
  cudaFilterModePoint,
  cudaFilterModeLinear
};


enum cudaTextureReadMode
{
  cudaReadModeElementType,
  cudaReadModeNormalizedFloat
};


struct textureReference
{
  int normalized;
  enum cudaTextureFilterMode filterMode;
  enum cudaTextureAddressMode addressMode[3];
  struct cudaChannelFormatDesc channelDesc;
  int __cudaReserved[16];
};
# 45 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/vector_types.h" 1 3
# 54 "/usr/local/cuda/bin/../include/vector_types.h" 3
struct char1
{
  signed char x;
};


struct uchar1
{
  unsigned char x;
};


struct char2
{
  signed char x, y;
};


struct uchar2
{
  unsigned char x, y;
};


struct char3
{
  signed char x, y, z;
};


struct uchar3
{
  unsigned char x, y, z;
};


struct char4
{
  signed char x, y, z, w;
};


struct uchar4
{
  unsigned char x, y, z, w;
};


struct short1
{
  short x;
};


struct ushort1
{
  unsigned short x;
};


struct short2
{
  short x, y;
};


struct ushort2
{
  unsigned short x, y;
};


struct short3
{
  short x, y, z;
};


struct ushort3
{
  unsigned short x, y, z;
};


struct short4
{
  short x, y, z, w;
};


struct ushort4
{
  unsigned short x, y, z, w;
};


struct int1
{
  int x;
};


struct uint1
{
  unsigned int x;
};


struct int2
{
  int x, y;
};


struct uint2
{
  unsigned int x, y;
};


struct int3
{
  int x, y, z;
};


struct uint3
{
  unsigned int x, y, z;
};


struct int4
{
  int x, y, z, w;
};


struct uint4
{
  unsigned int x, y, z, w;
};


struct long1
{
  long int x;
};


struct ulong1
{
  unsigned long x;
};


struct



      

                                             long2
{
  long int x, y;
};


struct



      

                                                      ulong2
{
  unsigned long int x, y;
};
# 262 "/usr/local/cuda/bin/../include/vector_types.h" 3
struct float1
{
  float x;
};


struct float2
{
  float x, y;
};


struct float3
{
  float x, y, z;
};


struct float4
{
  float x, y, z, w;
};


struct longlong1
{
  long long int x;
};


struct ulonglong1
{
  unsigned long long int x;
};


struct longlong2
{
  long long int x, y;
};


struct ulonglong2
{
  unsigned long long int x, y;
};


struct double1
{
  double x;
};


struct double2
{
  double x, y;
};
# 328 "/usr/local/cuda/bin/../include/vector_types.h" 3
typedef struct char1 char1;

typedef struct uchar1 uchar1;

typedef struct char2 char2;

typedef struct uchar2 uchar2;

typedef struct char3 char3;

typedef struct uchar3 uchar3;

typedef struct char4 char4;

typedef struct uchar4 uchar4;

typedef struct short1 short1;

typedef struct ushort1 ushort1;

typedef struct short2 short2;

typedef struct ushort2 ushort2;

typedef struct short3 short3;

typedef struct ushort3 ushort3;

typedef struct short4 short4;

typedef struct ushort4 ushort4;

typedef struct int1 int1;

typedef struct uint1 uint1;

typedef struct int2 int2;

typedef struct uint2 uint2;

typedef struct int3 int3;

typedef struct uint3 uint3;

typedef struct int4 int4;

typedef struct uint4 uint4;

typedef struct long1 long1;

typedef struct ulong1 ulong1;

typedef struct long2 long2;

typedef struct ulong2 ulong2;

typedef struct long3 long3;

typedef struct ulong3 ulong3;

typedef struct long4 long4;

typedef struct ulong4 ulong4;

typedef struct float1 float1;

typedef struct float2 float2;

typedef struct float3 float3;

typedef struct float4 float4;

typedef struct longlong1 longlong1;

typedef struct ulonglong1 ulonglong1;

typedef struct longlong2 longlong2;

typedef struct ulonglong2 ulonglong2;

typedef struct double1 double1;

typedef struct double2 double2;
# 419 "/usr/local/cuda/bin/../include/vector_types.h" 3
typedef struct dim3 dim3;


struct dim3
{
    unsigned int x, y, z;

    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

};
# 45 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 55 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 2 3
# 80 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern "C" {
# 89 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
extern cudaError_t cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
extern cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
extern cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
extern cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
# 102 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaMallocHost(void **ptr, size_t size);
extern cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
extern cudaError_t cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height );
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaFreeHost(void *ptr);
extern cudaError_t cudaFreeArray(struct cudaArray *array);

extern cudaError_t cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags);
extern cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
# 120 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset , enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind );







extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);







extern cudaError_t cudaMemset(void *devPtr, int value, size_t count);
extern cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);







extern cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
extern cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);







extern cudaError_t cudaGetDeviceCount(int *count);
extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
extern cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
extern cudaError_t cudaSetDevice(int device);
extern cudaError_t cudaGetDevice(int *device);
extern cudaError_t cudaSetValidDevices(int *device_arr, int len);
extern cudaError_t cudaSetDeviceFlags( int flags );







extern cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size );
extern cudaError_t cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch);
extern cudaError_t cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
extern cudaError_t cudaUnbindTexture(const struct textureReference *texref);
extern cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
extern cudaError_t cudaGetTextureReference(const struct textureReference **texref, const char *symbol);







extern cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
extern struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);







extern cudaError_t cudaGetLastError(void);
extern const char* cudaGetErrorString(cudaError_t error);







extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem , cudaStream_t stream );
extern cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
extern cudaError_t cudaLaunch(const char *entry);
extern cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func);







extern cudaError_t cudaStreamCreate(cudaStream_t *pStream);
extern cudaError_t cudaStreamDestroy(cudaStream_t stream);
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);
extern cudaError_t cudaStreamQuery(cudaStream_t stream);







extern cudaError_t cudaEventCreate(cudaEvent_t *event);
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern cudaError_t cudaEventQuery(cudaEvent_t event);
extern cudaError_t cudaEventSynchronize(cudaEvent_t event);
extern cudaError_t cudaEventDestroy(cudaEvent_t event);
extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);







extern cudaError_t cudaSetDoubleForDevice(double *d);
extern cudaError_t cudaSetDoubleForHost(double *d);







extern cudaError_t cudaThreadExit(void);
extern cudaError_t cudaThreadSynchronize(void);







extern cudaError_t cudaDriverGetVersion(int *driverVersion);
extern cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);


}
# 81 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2 3
# 1 "/usr/local/cuda/bin/../include/crt/storage_class.h" 1 3
# 82 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2 3
# 216 "/usr/lib/gcc/x86_64-linux-gnu/4.3.3/include/stddef.h" 2 3
# 90 "/usr/local/cuda/bin/../include/driver_types.h"
# 138 "/usr/local/cuda/bin/../include/driver_types.h"
# 150 "/usr/local/cuda/bin/../include/driver_types.h"
# 163 "/usr/local/cuda/bin/../include/driver_types.h"
# 169 "/usr/local/cuda/bin/../include/driver_types.h"
# 181 "/usr/local/cuda/bin/../include/driver_types.h"
# 193 "/usr/local/cuda/bin/../include/driver_types.h"
# 204 "/usr/local/cuda/bin/../include/driver_types.h"
# 215 "/usr/local/cuda/bin/../include/driver_types.h"
# 233 "/usr/local/cuda/bin/../include/driver_types.h"
# 247 "/usr/local/cuda/bin/../include/driver_types.h"
# 259 "/usr/local/cuda/bin/../include/driver_types.h"
# 318 "/usr/local/cuda/bin/../include/driver_types.h"
# 324 "/usr/local/cuda/bin/../include/driver_types.h"
# 330 "/usr/local/cuda/bin/../include/driver_types.h"
# 54 "/usr/local/cuda/bin/../include/texture_types.h"
# 61 "/usr/local/cuda/bin/../include/texture_types.h"
# 68 "/usr/local/cuda/bin/../include/texture_types.h"
# 75 "/usr/local/cuda/bin/../include/texture_types.h"
# 54 "/usr/local/cuda/bin/../include/vector_types.h"
# 60 "/usr/local/cuda/bin/../include/vector_types.h"
# 66 "/usr/local/cuda/bin/../include/vector_types.h"
# 72 "/usr/local/cuda/bin/../include/vector_types.h"
# 78 "/usr/local/cuda/bin/../include/vector_types.h"
# 84 "/usr/local/cuda/bin/../include/vector_types.h"
# 90 "/usr/local/cuda/bin/../include/vector_types.h"
# 96 "/usr/local/cuda/bin/../include/vector_types.h"
# 102 "/usr/local/cuda/bin/../include/vector_types.h"
# 108 "/usr/local/cuda/bin/../include/vector_types.h"
# 114 "/usr/local/cuda/bin/../include/vector_types.h"
# 120 "/usr/local/cuda/bin/../include/vector_types.h"
# 126 "/usr/local/cuda/bin/../include/vector_types.h"
# 132 "/usr/local/cuda/bin/../include/vector_types.h"
# 138 "/usr/local/cuda/bin/../include/vector_types.h"
# 144 "/usr/local/cuda/bin/../include/vector_types.h"
# 150 "/usr/local/cuda/bin/../include/vector_types.h"
# 156 "/usr/local/cuda/bin/../include/vector_types.h"
# 162 "/usr/local/cuda/bin/../include/vector_types.h"
# 168 "/usr/local/cuda/bin/../include/vector_types.h"
# 174 "/usr/local/cuda/bin/../include/vector_types.h"
# 180 "/usr/local/cuda/bin/../include/vector_types.h"
# 186 "/usr/local/cuda/bin/../include/vector_types.h"
# 192 "/usr/local/cuda/bin/../include/vector_types.h"
# 198 "/usr/local/cuda/bin/../include/vector_types.h"
# 204 "/usr/local/cuda/bin/../include/vector_types.h"
# 216 "/usr/local/cuda/bin/../include/vector_types.h"
# 228 "/usr/local/cuda/bin/../include/vector_types.h"
# 262 "/usr/local/cuda/bin/../include/vector_types.h"
# 268 "/usr/local/cuda/bin/../include/vector_types.h"
# 274 "/usr/local/cuda/bin/../include/vector_types.h"
# 280 "/usr/local/cuda/bin/../include/vector_types.h"
# 286 "/usr/local/cuda/bin/../include/vector_types.h"
# 292 "/usr/local/cuda/bin/../include/vector_types.h"
# 298 "/usr/local/cuda/bin/../include/vector_types.h"
# 304 "/usr/local/cuda/bin/../include/vector_types.h"
# 310 "/usr/local/cuda/bin/../include/vector_types.h"
# 316 "/usr/local/cuda/bin/../include/vector_types.h"
# 328 "/usr/local/cuda/bin/../include/vector_types.h"
# 330 "/usr/local/cuda/bin/../include/vector_types.h"
# 332 "/usr/local/cuda/bin/../include/vector_types.h"
# 334 "/usr/local/cuda/bin/../include/vector_types.h"
# 336 "/usr/local/cuda/bin/../include/vector_types.h"
# 338 "/usr/local/cuda/bin/../include/vector_types.h"
# 340 "/usr/local/cuda/bin/../include/vector_types.h"
# 342 "/usr/local/cuda/bin/../include/vector_types.h"
# 344 "/usr/local/cuda/bin/../include/vector_types.h"
# 346 "/usr/local/cuda/bin/../include/vector_types.h"
# 348 "/usr/local/cuda/bin/../include/vector_types.h"
# 350 "/usr/local/cuda/bin/../include/vector_types.h"
# 352 "/usr/local/cuda/bin/../include/vector_types.h"
# 354 "/usr/local/cuda/bin/../include/vector_types.h"
# 356 "/usr/local/cuda/bin/../include/vector_types.h"
# 358 "/usr/local/cuda/bin/../include/vector_types.h"
# 360 "/usr/local/cuda/bin/../include/vector_types.h"
# 362 "/usr/local/cuda/bin/../include/vector_types.h"
# 364 "/usr/local/cuda/bin/../include/vector_types.h"
# 366 "/usr/local/cuda/bin/../include/vector_types.h"
# 368 "/usr/local/cuda/bin/../include/vector_types.h"
# 370 "/usr/local/cuda/bin/../include/vector_types.h"
# 372 "/usr/local/cuda/bin/../include/vector_types.h"
# 374 "/usr/local/cuda/bin/../include/vector_types.h"
# 376 "/usr/local/cuda/bin/../include/vector_types.h"
# 378 "/usr/local/cuda/bin/../include/vector_types.h"
# 380 "/usr/local/cuda/bin/../include/vector_types.h"
# 382 "/usr/local/cuda/bin/../include/vector_types.h"
# 384 "/usr/local/cuda/bin/../include/vector_types.h"
# 386 "/usr/local/cuda/bin/../include/vector_types.h"
# 388 "/usr/local/cuda/bin/../include/vector_types.h"
# 390 "/usr/local/cuda/bin/../include/vector_types.h"
# 392 "/usr/local/cuda/bin/../include/vector_types.h"
# 394 "/usr/local/cuda/bin/../include/vector_types.h"
# 396 "/usr/local/cuda/bin/../include/vector_types.h"
# 398 "/usr/local/cuda/bin/../include/vector_types.h"
# 400 "/usr/local/cuda/bin/../include/vector_types.h"
# 402 "/usr/local/cuda/bin/../include/vector_types.h"
# 404 "/usr/local/cuda/bin/../include/vector_types.h"
# 406 "/usr/local/cuda/bin/../include/vector_types.h"
# 408 "/usr/local/cuda/bin/../include/vector_types.h"
# 410 "/usr/local/cuda/bin/../include/vector_types.h"
# 419 "/usr/local/cuda/bin/../include/vector_types.h"
# 422 "/usr/local/cuda/bin/../include/vector_types.h"
# 89 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent);
extern "C" cudaError_t cudaMalloc3DArray(cudaArray **, const cudaChannelFormatDesc *, cudaExtent);
extern "C" cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent);
extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *);
extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t);
# 102 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc(void **, size_t);
extern "C" cudaError_t cudaMallocHost(void **, size_t);
extern "C" cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
extern "C" cudaError_t cudaMallocArray(cudaArray **, const cudaChannelFormatDesc *, size_t, size_t = (1));
extern "C" cudaError_t cudaFree(void *);
extern "C" cudaError_t cudaFreeHost(void *);
extern "C" cudaError_t cudaFreeArray(cudaArray *);

extern "C" cudaError_t cudaHostAlloc(void **, size_t, unsigned);
extern "C" cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned);
# 120 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpyToArray(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpyFromArray(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
extern "C" cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpy2DFromArray(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind);
extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
extern "C" cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyHostToDevice);
extern "C" cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyDeviceToHost);
# 137 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 152 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset(void *, int, size_t);
extern "C" cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t);
# 161 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolAddress(void **, const char *);
extern "C" cudaError_t cudaGetSymbolSize(size_t *, const char *);
# 170 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceCount(int *);
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
extern "C" cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *);
extern "C" cudaError_t cudaSetDevice(int);
extern "C" cudaError_t cudaGetDevice(int *);
extern "C" cudaError_t cudaSetValidDevices(int *, int);
extern "C" cudaError_t cudaSetDeviceFlags(int);
# 184 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t = (((2147483647) * 2U) + 1U));
extern "C" cudaError_t cudaBindTexture2D(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t, size_t, size_t);
extern "C" cudaError_t cudaBindTextureToArray(const textureReference *, const cudaArray *, const cudaChannelFormatDesc *);
extern "C" cudaError_t cudaUnbindTexture(const textureReference *);
extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *, const textureReference *);
extern "C" cudaError_t cudaGetTextureReference(const textureReference **, const char *);
# 197 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, const cudaArray *);
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind);
# 206 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetLastError();
extern "C" const char *cudaGetErrorString(cudaError_t);
# 215 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = (0), cudaStream_t = (0));
extern "C" cudaError_t cudaSetupArgument(const void *, size_t, size_t);
extern "C" cudaError_t cudaLaunch(const char *);
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const char *);
# 226 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *);
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t);
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t cudaStreamQuery(cudaStream_t);
# 237 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *);
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, int);
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t cudaEventQuery(cudaEvent_t);
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t);
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
# 251 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForDevice(double *);
extern "C" cudaError_t cudaSetDoubleForHost(double *);
# 260 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadExit();
extern "C" cudaError_t cudaThreadSynchronize();
# 269 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaDriverGetVersion(int *);
extern "C" cudaError_t cudaRuntimeGetVersion(int *);
# 93 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<class T> inline cudaChannelFormatDesc cudaCreateChannelDesc()
{
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> ()
{
auto int e = (((int)sizeof(char)) * 8);




return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);

}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> ()
{
auto int e = (((int)sizeof(signed char)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> ()
{
auto int e = (((int)sizeof(unsigned char)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> ()
{
auto int e = (((int)sizeof(signed char)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> ()
{
auto int e = (((int)sizeof(unsigned char)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> ()
{
auto int e = (((int)sizeof(signed char)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> ()
{
auto int e = (((int)sizeof(unsigned char)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> ()
{
auto int e = (((int)sizeof(signed char)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> ()
{
auto int e = (((int)sizeof(unsigned char)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> ()
{
auto int e = (((int)sizeof(short)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> ()
{
auto int e = (((int)sizeof(unsigned short)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> ()
{
auto int e = (((int)sizeof(short)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> ()
{
auto int e = (((int)sizeof(unsigned short)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> ()
{
auto int e = (((int)sizeof(short)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> ()
{
auto int e = (((int)sizeof(unsigned short)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> ()
{
auto int e = (((int)sizeof(short)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> ()
{
auto int e = (((int)sizeof(unsigned short)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> ()
{
auto int e = (((int)sizeof(int)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> ()
{
auto int e = (((int)sizeof(unsigned)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> ()
{
auto int e = (((int)sizeof(int)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> ()
{
auto int e = (((int)sizeof(unsigned)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> ()
{
auto int e = (((int)sizeof(int)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> ()
{
auto int e = (((int)sizeof(unsigned)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> ()
{
auto int e = (((int)sizeof(int)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> ()
{
auto int e = (((int)sizeof(unsigned)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}
# 337 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> ()
{
auto int e = (((int)sizeof(float)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> ()
{
auto int e = (((int)sizeof(float)) * 8);

return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> ()
{
auto int e = (((int)sizeof(float)) * 8);

return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> ()
{
auto int e = (((int)sizeof(float)) * 8);

return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}
# 54 "/usr/local/cuda/bin/../include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
{
auto cudaPitchedPtr s;

(s.ptr) = d;
(s.pitch) = p;
(s.xsize) = xsz;
(s.ysize) = ysz;

return s;
}

static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
{
auto cudaPos p;

(p.x) = x;
(p.y) = y;
(p.z) = z;

return p;
}

static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
auto cudaExtent e;

(e.width) = w;
(e.height) = h;
(e.depth) = d;

return e;
}
# 54 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char1 make_char1(signed char x)
{
auto char1 t; (t.x) = x; return t;
}

static inline uchar1 make_uchar1(unsigned char x)
{
auto uchar1 t; (t.x) = x; return t;
}

static inline char2 make_char2(signed char x, signed char y)
{
auto char2 t; (t.x) = x; (t.y) = y; return t;
}

static inline uchar2 make_uchar2(unsigned char x, unsigned char y)
{
auto uchar2 t; (t.x) = x; (t.y) = y; return t;
}

static inline char3 make_char3(signed char x, signed char y, signed char z)
{
auto char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
auto uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
auto char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
auto uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline short1 make_short1(short x)
{
auto short1 t; (t.x) = x; return t;
}

static inline ushort1 make_ushort1(unsigned short x)
{
auto ushort1 t; (t.x) = x; return t;
}

static inline short2 make_short2(short x, short y)
{
auto short2 t; (t.x) = x; (t.y) = y; return t;
}

static inline ushort2 make_ushort2(unsigned short x, unsigned short y)
{
auto ushort2 t; (t.x) = x; (t.y) = y; return t;
}

static inline short3 make_short3(short x, short y, short z)
{
auto short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
auto ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline short4 make_short4(short x, short y, short z, short w)
{
auto short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
auto ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline int1 make_int1(int x)
{
auto int1 t; (t.x) = x; return t;
}

static inline uint1 make_uint1(unsigned x)
{
auto uint1 t; (t.x) = x; return t;
}

static inline int2 make_int2(int x, int y)
{
auto int2 t; (t.x) = x; (t.y) = y; return t;
}

static inline uint2 make_uint2(unsigned x, unsigned y)
{
auto uint2 t; (t.x) = x; (t.y) = y; return t;
}

static inline int3 make_int3(int x, int y, int z)
{
auto int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z)
{
auto uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline int4 make_int4(int x, int y, int z, int w)
{
auto int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w)
{
auto uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline long1 make_long1(long x)
{
auto long1 t; (t.x) = x; return t;
}

static inline ulong1 make_ulong1(unsigned long x)
{
auto ulong1 t; (t.x) = x; return t;
}

static inline long2 make_long2(long x, long y)
{
auto long2 t; (t.x) = x; (t.y) = y; return t;
}

static inline ulong2 make_ulong2(unsigned long x, unsigned long y)
{
auto ulong2 t; (t.x) = x; (t.y) = y; return t;
}
# 218 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline float1 make_float1(float x)
{
auto float1 t; (t.x) = x; return t;
}

static inline float2 make_float2(float x, float y)
{
auto float2 t; (t.x) = x; (t.y) = y; return t;
}

static inline float3 make_float3(float x, float y, float z)
{
auto float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
}

static inline float4 make_float4(float x, float y, float z, float w)
{
auto float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
}

static inline longlong1 make_longlong1(long long x)
{
auto longlong1 t; (t.x) = x; return t;
}

static inline ulonglong1 make_ulonglong1(unsigned long long x)
{
auto ulonglong1 t; (t.x) = x; return t;
}

static inline longlong2 make_longlong2(long long x, long long y)
{
auto longlong2 t; (t.x) = x; (t.y) = y; return t;
}

static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y)
{
auto ulonglong2 t; (t.x) = x; (t.y) = y; return t;
}

static inline double1 make_double1(double x)
{
auto double1 t; (t.x) = x; return t;
}

static inline double2 make_double2(double x, double y)
{
auto double2 t; (t.x) = x; (t.y) = y; return t;
}
# 31 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __u_char; }
extern "C" { typedef unsigned short __u_short; }
extern "C" { typedef unsigned __u_int; }
extern "C" { typedef unsigned long __u_long; }


extern "C" { typedef signed char __int8_t; }
extern "C" { typedef unsigned char __uint8_t; }
extern "C" { typedef signed short __int16_t; }
extern "C" { typedef unsigned short __uint16_t; }
extern "C" { typedef signed int __int32_t; }
extern "C" { typedef unsigned __uint32_t; }

extern "C" { typedef signed long __int64_t; }
extern "C" { typedef unsigned long __uint64_t; }
# 53 "/usr/include/bits/types.h" 3
extern "C" { typedef long __quad_t; }
extern "C" { typedef unsigned long __u_quad_t; }
# 134 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __dev_t; }
extern "C" { typedef unsigned __uid_t; }
extern "C" { typedef unsigned __gid_t; }
extern "C" { typedef unsigned long __ino_t; }
extern "C" { typedef unsigned long __ino64_t; }
extern "C" { typedef unsigned __mode_t; }
extern "C" { typedef unsigned long __nlink_t; }
extern "C" { typedef long __off_t; }
extern "C" { typedef long __off64_t; }
extern "C" { typedef int __pid_t; }
extern "C" { typedef struct __fsid_t { int __val[2]; } __fsid_t; }
extern "C" { typedef long __clock_t; }
extern "C" { typedef unsigned long __rlim_t; }
extern "C" { typedef unsigned long __rlim64_t; }
extern "C" { typedef unsigned __id_t; }
extern "C" { typedef long __time_t; }
extern "C" { typedef unsigned __useconds_t; }
extern "C" { typedef long __suseconds_t; }

extern "C" { typedef int __daddr_t; }
extern "C" { typedef long __swblk_t; }
extern "C" { typedef int __key_t; }


extern "C" { typedef int __clockid_t; }


extern "C" { typedef void *__timer_t; }


extern "C" { typedef long __blksize_t; }




extern "C" { typedef long __blkcnt_t; }
extern "C" { typedef long __blkcnt64_t; }


extern "C" { typedef unsigned long __fsblkcnt_t; }
extern "C" { typedef unsigned long __fsblkcnt64_t; }


extern "C" { typedef unsigned long __fsfilcnt_t; }
extern "C" { typedef unsigned long __fsfilcnt64_t; }

extern "C" { typedef long __ssize_t; }



extern "C" { typedef __off64_t __loff_t; }
extern "C" { typedef __quad_t *__qaddr_t; }
extern "C" { typedef char *__caddr_t; }


extern "C" { typedef long __intptr_t; }


extern "C" { typedef unsigned __socklen_t; }
# 61 "/usr/include/time.h" 3
extern "C" { typedef __clock_t clock_t; }
# 77 "/usr/include/time.h" 3
extern "C" { typedef __time_t time_t; }
# 93 "/usr/include/time.h" 3
extern "C" { typedef __clockid_t clockid_t; }
# 105 "/usr/include/time.h" 3
extern "C" { typedef __timer_t timer_t; }
# 121 "/usr/include/time.h" 3
extern "C" { struct timespec {

__time_t tv_sec;
long tv_nsec;
}; }
# 134 "/usr/include/time.h" 3
extern "C" { struct tm {

int tm_sec;
int tm_min;
int tm_hour;
int tm_mday;
int tm_mon;
int tm_year;
int tm_wday;
int tm_yday;
int tm_isdst;


long tm_gmtoff;
const char *tm_zone;




}; }
# 162 "/usr/include/time.h" 3
extern "C" { struct itimerspec {

timespec it_interval;
timespec it_value;
}; }


struct sigevent;
# 175 "/usr/include/time.h" 3
extern "C" { typedef __pid_t pid_t; }
# 184 "/usr/include/time.h" 3
extern "C" __attribute__((__weak__)) clock_t clock() throw();


extern "C" time_t time(time_t *) throw();


extern "C" double difftime(time_t, time_t) throw() __attribute__((__const__));



extern "C" time_t mktime(tm *) throw();
# 200 "/usr/include/time.h" 3
extern "C" size_t strftime(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__) throw();
# 208 "/usr/include/time.h" 3
extern "C" char *strptime(const char *__restrict__, const char *__restrict__, tm *) throw();
# 40 "/usr/include/xlocale.h" 3
extern "C" { typedef
# 28 "/usr/include/xlocale.h" 3
struct __locale_struct {


struct locale_data *__locales[13];


const unsigned short *__ctype_b;
const int *__ctype_tolower;
const int *__ctype_toupper;


const char *__names[13];
} *__locale_t; }
# 218 "/usr/include/time.h" 3
extern "C" size_t strftime_l(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__, __locale_t) throw();




extern "C" char *strptime_l(const char *__restrict__, const char *__restrict__, tm *, __locale_t) throw();
# 232 "/usr/include/time.h" 3
extern "C" tm *gmtime(const time_t *) throw();



extern "C" tm *localtime(const time_t *) throw();
# 242 "/usr/include/time.h" 3
extern "C" tm *gmtime_r(const time_t *__restrict__, tm *__restrict__) throw();




extern "C" tm *localtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 254 "/usr/include/time.h" 3
extern "C" char *asctime(const tm *) throw();


extern "C" char *ctime(const time_t *) throw();
# 265 "/usr/include/time.h" 3
extern "C" char *asctime_r(const tm *__restrict__, char *__restrict__) throw();



extern "C" char *ctime_r(const time_t *__restrict__, char *__restrict__) throw();
# 275 "/usr/include/time.h" 3
extern "C" { extern char *__tzname[2]; }
extern "C" { extern int __daylight; }
extern "C" { extern long __timezone; }




extern "C" { extern char *tzname[2]; }



extern "C" void tzset() throw();



extern "C" { extern int daylight; }
extern "C" { extern long timezone; }
# 297 "/usr/include/time.h" 3
extern "C" int stime(const time_t *) throw();
# 312 "/usr/include/time.h" 3
extern "C" time_t timegm(tm *) throw();


extern "C" time_t timelocal(tm *) throw();


extern "C" int dysize(int) throw() __attribute__((__const__));
# 327 "/usr/include/time.h" 3
extern "C" int nanosleep(const timespec *, timespec *);




extern "C" int clock_getres(clockid_t, timespec *) throw();


extern "C" int clock_gettime(clockid_t, timespec *) throw();


extern "C" int clock_settime(clockid_t, const timespec *) throw();
# 346 "/usr/include/time.h" 3
extern "C" int clock_nanosleep(clockid_t, int, const timespec *, timespec *);




extern "C" int clock_getcpuclockid(pid_t, clockid_t *) throw();




extern "C" int timer_create(clockid_t, sigevent *__restrict__, timer_t *__restrict__) throw();




extern "C" int timer_delete(timer_t) throw();


extern "C" int timer_settime(timer_t, int, const itimerspec *__restrict__, itimerspec *__restrict__) throw();




extern "C" int timer_gettime(timer_t, itimerspec *) throw();



extern "C" int timer_getoverrun(timer_t) throw();
# 389 "/usr/include/time.h" 3
extern "C" { extern int getdate_err; }
# 398 "/usr/include/time.h" 3
extern "C" tm *getdate(const char *);
# 412 "/usr/include/time.h" 3
extern "C" int getdate_r(const char *__restrict__, tm *__restrict__);
# 38 "/usr/include/string.h" 3
extern "C" { extern inline __attribute__((__weak__)) void *memcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" { extern inline void *memmove(void *, const void *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 51 "/usr/include/string.h" 3
extern "C" void *memccpy(void *__restrict__, const void *__restrict__, int, size_t) throw();
# 59 "/usr/include/string.h" 3
extern "C" { extern inline __attribute__((__weak__)) void *memset(void *, int, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" int memcmp(const void *, const void *, size_t) throw() __attribute__((__pure__));



extern "C" void *memchr(const void *, int, size_t) throw() __attribute__((__pure__));
# 73 "/usr/include/string.h" 3
extern "C" void *rawmemchr(const void *, int) throw() __attribute__((__pure__));



extern "C" void *memrchr(const void *, int, size_t) throw() __attribute__((__pure__));
# 84 "/usr/include/string.h" 3
extern "C" { extern inline char *strcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline char *strncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" { extern inline char *strcat(char *__restrict__, const char *__restrict__) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline char *strncat(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" int strcmp(const char *, const char *) throw() __attribute__((__pure__));


extern "C" int strncmp(const char *, const char *, size_t) throw() __attribute__((__pure__));



extern "C" int strcoll(const char *, const char *) throw() __attribute__((__pure__));


extern "C" size_t strxfrm(char *__restrict__, const char *__restrict__, size_t) throw();
# 121 "/usr/include/string.h" 3
extern "C" int strcoll_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__));


extern "C" size_t strxfrm_l(char *, const char *, size_t, __locale_t) throw();
# 130 "/usr/include/string.h" 3
extern "C" char *strdup(const char *) throw() __attribute__((__malloc__));
# 138 "/usr/include/string.h" 3
extern "C" char *strndup(const char *, size_t) throw() __attribute__((__malloc__));
# 167 "/usr/include/string.h" 3
extern "C" char *strchr(const char *, int) throw() __attribute__((__pure__));


extern "C" char *strrchr(const char *, int) throw() __attribute__((__pure__));
# 177 "/usr/include/string.h" 3
extern "C" char *strchrnul(const char *, int) throw() __attribute__((__pure__));
# 184 "/usr/include/string.h" 3
extern "C" size_t strcspn(const char *, const char *) throw() __attribute__((__pure__));



extern "C" size_t strspn(const char *, const char *) throw() __attribute__((__pure__));


extern "C" char *strpbrk(const char *, const char *) throw() __attribute__((__pure__));


extern "C" char *strstr(const char *, const char *) throw() __attribute__((__pure__));




extern "C" char *strtok(char *__restrict__, const char *__restrict__) throw();
# 205 "/usr/include/string.h" 3
extern "C" char *__strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw();




extern "C" char *strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw();
# 217 "/usr/include/string.h" 3
extern "C" char *strcasestr(const char *, const char *) throw() __attribute__((__pure__));
# 225 "/usr/include/string.h" 3
extern "C" void *memmem(const void *, size_t, const void *, size_t) throw() __attribute__((__pure__));
# 231 "/usr/include/string.h" 3
extern "C" void *__mempcpy(void *__restrict__, const void *__restrict__, size_t) throw();


extern "C" { extern inline void *mempcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 242 "/usr/include/string.h" 3
extern "C" size_t strlen(const char *) throw() __attribute__((__pure__));
# 249 "/usr/include/string.h" 3
extern "C" size_t strnlen(const char *, size_t) throw() __attribute__((__pure__));
# 256 "/usr/include/string.h" 3
extern "C" char *strerror(int) throw();
# 281 "/usr/include/string.h" 3
extern "C" char *strerror_r(int, char *, size_t) throw();
# 288 "/usr/include/string.h" 3
extern "C" char *strerror_l(int, __locale_t) throw();
# 294 "/usr/include/string.h" 3
extern "C" void __bzero(void *, size_t) throw();



extern "C" { extern inline void bcopy(const void *, void *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" { extern inline void bzero(void *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" int bcmp(const void *, const void *, size_t) throw() __attribute__((__pure__));



extern "C" char *index(const char *, int) throw() __attribute__((__pure__));



extern "C" char *rindex(const char *, int) throw() __attribute__((__pure__));




extern "C" int ffs(int) throw() __attribute__((__const__));




extern "C" int ffsl(long) throw() __attribute__((__const__));

extern "C" int ffsll(long long) throw() __attribute__((__const__));
# 331 "/usr/include/string.h" 3
extern "C" int strcasecmp(const char *, const char *) throw() __attribute__((__pure__));



extern "C" int strncasecmp(const char *, const char *, size_t) throw() __attribute__((__pure__));
# 342 "/usr/include/string.h" 3
extern "C" int strcasecmp_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__));



extern "C" int strncasecmp_l(const char *, const char *, size_t, __locale_t) throw() __attribute__((__pure__));
# 354 "/usr/include/string.h" 3
extern "C" char *strsep(char **__restrict__, const char *__restrict__) throw();
# 361 "/usr/include/string.h" 3
extern "C" int strverscmp(const char *, const char *) throw() __attribute__((__pure__));



extern "C" char *strsignal(int) throw();


extern "C" char *__stpcpy(char *__restrict__, const char *__restrict__) throw();

extern "C" { extern inline char *stpcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" char *__stpncpy(char *__restrict__, const char *__restrict__, size_t) throw();


extern "C" { extern inline char *stpncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" char *strfry(char *) throw();


extern "C" void *memfrob(void *, size_t) throw();
# 393 "/usr/include/string.h" 3
extern "C" char *basename(const char *) throw();
# 23 "/usr/include/bits/string3.h" 3
extern "C" void __warn_memset_zero_len();
# 49 "/usr/include/bits/string3.h" 3
extern "C" { inline __attribute__((__weak__)) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __len) throw()

{
return __builtin___memcpy_chk(__dest, __src, __len, __builtin_object_size(__dest, 0));
} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void *memmove(void *__restrict__ __dest, const void *__restrict__ __src, size_t __len) throw()

{
return __builtin___memmove_chk(__dest, __src, __len, __builtin_object_size(__dest, 0));
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __len) throw()

{
return __builtin___mempcpy_chk(__dest, __src, __len, __builtin_object_size(__dest, 0));
} }
# 78 "/usr/include/bits/string3.h" 3
extern "C" { inline __attribute__((__weak__)) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void *memset(void *__dest, int __ch, size_t __len) throw()
{
if ((__builtin_constant_p(__len)) && (__len == (0)))
{
__warn_memset_zero_len();
return __dest;
}
return __builtin___memset_chk(__dest, __ch, __len, __builtin_object_size(__dest, 0));
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void bcopy(const void *__restrict__ __src, void *__restrict__ __dest, size_t __len) throw()

{
(void)__builtin___memmove_chk(__dest, __src, __len, __builtin_object_size(__dest, 0));
} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) void bzero(void *__dest, size_t __len) throw()
{
(void)__builtin___memset_chk(__dest, '\000', __len, __builtin_object_size(__dest, 0));
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
{
return __builtin___strcpy_chk(__dest, __src, __builtin_object_size(__dest, 2 > 1));
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
{
return __builtin___stpcpy_chk(__dest, __src, __builtin_object_size(__dest, 2 > 1));
} }




extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __len) throw()

{
return __builtin___strncpy_chk(__dest, __src, __len, __builtin_object_size(__dest, 2 > 1));
} }


extern "C" char *__stpncpy_chk(char *, const char *, size_t, size_t) throw();

extern "C" char *__stpncpy_alias(char *, const char *, size_t) throw() __asm__("stpncpy");




extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *stpncpy(char *__dest, const char *__src, size_t __n) throw()
{
if ((__builtin_object_size(__dest, 2 > 1) != ((size_t)(-1))) && ((!(__builtin_constant_p(__n))) || (__n <= __builtin_object_size(__dest, 2 > 1)))) {

return __stpncpy_chk(__dest, __src, __n, __builtin_object_size(__dest, 2 > 1)); }
return __stpncpy_alias(__dest, __src, __n);
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
{
return __builtin___strcat_chk(__dest, __src, __builtin_object_size(__dest, 2 > 1));
} }



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __len) throw()

{
return __builtin___strncat_chk(__dest, __src, __len, __builtin_object_size(__dest, 2 > 1));
} }
# 56 "/usr/local/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((__weak__)) clock_t clock() throw();


extern "C" { extern inline __attribute__((__weak__)) void *memset(void *, int, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline __attribute__((__weak__)) void *memcpy(void *, const void *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 65 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int min(int, int);

extern "C" __attribute__((__weak__)) unsigned umin(unsigned, unsigned);

extern "C" __attribute__((__weak__)) float fminf(float, float) throw();

extern "C" __attribute__((__weak__)) double fmin(double, double) throw();


extern "C" __attribute__((__weak__)) int max(int, int);

extern "C" __attribute__((__weak__)) unsigned umax(unsigned, unsigned);

extern "C" __attribute__((__weak__)) float fmaxf(float, float) throw();

extern "C" __attribute__((__weak__)) double fmax(double, double) throw();


extern "C" __attribute__((__weak__)) double sin(double) throw();

extern "C" __attribute__((__weak__)) float sinf(float) throw();


extern "C" __attribute__((__weak__)) double cos(double) throw();

extern "C" __attribute__((__weak__)) float cosf(float) throw();


extern "C" __attribute__((__weak__)) void sincos(double, double *, double *) throw();

extern "C" __attribute__((__weak__)) void sincosf(float, float *, float *) throw();


extern "C" __attribute__((__weak__)) double tan(double) throw();

extern "C" __attribute__((__weak__)) float tanf(float) throw();


extern "C" __attribute__((__weak__)) double sqrt(double) throw();

extern "C" __attribute__((__weak__)) float sqrtf(float) throw();


extern "C" __attribute__((__weak__)) double rsqrt(double);

extern "C" __attribute__((__weak__)) float rsqrtf(float);


extern "C" __attribute__((__weak__)) double exp2(double) throw();

extern "C" __attribute__((__weak__)) float exp2f(float) throw();


extern "C" __attribute__((__weak__)) double exp10(double) throw();

extern "C" __attribute__((__weak__)) float exp10f(float) throw();


extern "C" __attribute__((__weak__)) double expm1(double) throw();

extern "C" __attribute__((__weak__)) float expm1f(float) throw();


extern "C" __attribute__((__weak__)) double log2(double) throw();

extern "C" __attribute__((__weak__)) float log2f(float) throw();


extern "C" __attribute__((__weak__)) double log10(double) throw();

extern "C" __attribute__((__weak__)) float log10f(float) throw();


extern "C" __attribute__((__weak__)) double log(double) throw();

extern "C" __attribute__((__weak__)) float logf(float) throw();


extern "C" __attribute__((__weak__)) double log1p(double) throw();

extern "C" __attribute__((__weak__)) float log1pf(float) throw();


extern "C" __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double exp(double) throw();

extern "C" __attribute__((__weak__)) float expf(float) throw();


extern "C" __attribute__((__weak__)) double cosh(double) throw();

extern "C" __attribute__((__weak__)) float coshf(float) throw();


extern "C" __attribute__((__weak__)) double sinh(double) throw();

extern "C" __attribute__((__weak__)) float sinhf(float) throw();


extern "C" __attribute__((__weak__)) double tanh(double) throw();

extern "C" __attribute__((__weak__)) float tanhf(float) throw();


extern "C" __attribute__((__weak__)) double acosh(double) throw();

extern "C" __attribute__((__weak__)) float acoshf(float) throw();


extern "C" __attribute__((__weak__)) double asinh(double) throw();

extern "C" __attribute__((__weak__)) float asinhf(float) throw();


extern "C" __attribute__((__weak__)) double atanh(double) throw();

extern "C" __attribute__((__weak__)) float atanhf(float) throw();


extern "C" __attribute__((__weak__)) double ldexp(double, int) throw();

extern "C" __attribute__((__weak__)) float ldexpf(float, int) throw();


extern "C" __attribute__((__weak__)) double logb(double) throw();

extern "C" __attribute__((__weak__)) float logbf(float) throw();


extern "C" __attribute__((__weak__)) int ilogb(double) throw();

extern "C" __attribute__((__weak__)) int ilogbf(float) throw();


extern "C" __attribute__((__weak__)) double scalbn(double, int) throw();

extern "C" __attribute__((__weak__)) float scalbnf(float, int) throw();


extern "C" __attribute__((__weak__)) double scalbln(double, long) throw();

extern "C" __attribute__((__weak__)) float scalblnf(float, long) throw();


extern "C" __attribute__((__weak__)) double frexp(double, int *) throw();

extern "C" __attribute__((__weak__)) float frexpf(float, int *) throw();


extern "C" __attribute__((__weak__)) double round(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) long lround(double) throw();

extern "C" __attribute__((__weak__)) long lroundf(float) throw();


extern "C" __attribute__((__weak__)) long long llround(double) throw();

extern "C" __attribute__((__weak__)) long long llroundf(float) throw();


extern "C" __attribute__((__weak__)) double rint(double) throw();

extern "C" __attribute__((__weak__)) float rintf(float) throw();


extern "C" __attribute__((__weak__)) long lrint(double) throw();

extern "C" __attribute__((__weak__)) long lrintf(float) throw();


extern "C" __attribute__((__weak__)) long long llrint(double) throw();

extern "C" __attribute__((__weak__)) long long llrintf(float) throw();


extern "C" __attribute__((__weak__)) double nearbyint(double) throw();

extern "C" __attribute__((__weak__)) float nearbyintf(float) throw();


extern "C" __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double fdim(double, double) throw();

extern "C" __attribute__((__weak__)) float fdimf(float, float) throw();


extern "C" __attribute__((__weak__)) double atan2(double, double) throw();

extern "C" __attribute__((__weak__)) float atan2f(float, float) throw();


extern "C" __attribute__((__weak__)) double atan(double) throw();

extern "C" __attribute__((__weak__)) float atanf(float) throw();


extern "C" __attribute__((__weak__)) double asin(double) throw();

extern "C" __attribute__((__weak__)) float asinf(float) throw();


extern "C" __attribute__((__weak__)) double acos(double) throw();

extern "C" __attribute__((__weak__)) float acosf(float) throw();


extern "C" __attribute__((__weak__)) double hypot(double, double) throw();

extern "C" __attribute__((__weak__)) float hypotf(float, float) throw();


extern "C" __attribute__((__weak__)) double cbrt(double) throw();

extern "C" __attribute__((__weak__)) float cbrtf(float) throw();


extern "C" __attribute__((__weak__)) double pow(double, double) throw();

extern "C" __attribute__((__weak__)) float powf(float, float) throw();


extern "C" __attribute__((__weak__)) double modf(double, double *) throw();

extern "C" __attribute__((__weak__)) float modff(float, float *) throw();


extern "C" __attribute__((__weak__)) double fmod(double, double) throw();

extern "C" __attribute__((__weak__)) float fmodf(float, float) throw();


extern "C" __attribute__((__weak__)) double remainder(double, double) throw();

extern "C" __attribute__((__weak__)) float remainderf(float, float) throw();


extern "C" __attribute__((__weak__)) double remquo(double, double, int *) throw();

extern "C" __attribute__((__weak__)) float remquof(float, float, int *) throw();


extern "C" __attribute__((__weak__)) double erf(double) throw();

extern "C" __attribute__((__weak__)) float erff(float) throw();


extern "C" __attribute__((__weak__)) double erfinv(double) throw();

extern "C" __attribute__((__weak__)) float erfinvf(float) throw();


extern "C" __attribute__((__weak__)) double erfc(double) throw();

extern "C" __attribute__((__weak__)) float erfcf(float) throw();


extern "C" __attribute__((__weak__)) double erfcinv(double) throw();

extern "C" __attribute__((__weak__)) float erfcinvf(float) throw();


extern "C" __attribute__((__weak__)) double lgamma(double) throw();

extern "C" __attribute__((__weak__)) float lgammaf(float) throw();


extern "C" __attribute__((__weak__)) double tgamma(double) throw();

extern "C" __attribute__((__weak__)) float tgammaf(float) throw();


extern "C" __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__));
# 400 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__));

extern "C" { extern inline __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }




extern "C" { extern inline __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }


extern "C" __attribute__((__weak__)) double fma(double, double, double) throw();

extern "C" __attribute__((__weak__)) float fmaf(float, float, float) throw();
# 31 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef float float_t; }
extern "C" { typedef double double_t; }
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double acos(double) throw(); extern "C" double __acos(double) throw();

extern "C" __attribute__((__weak__)) double asin(double) throw(); extern "C" double __asin(double) throw();

extern "C" __attribute__((__weak__)) double atan(double) throw(); extern "C" double __atan(double) throw();

extern "C" __attribute__((__weak__)) double atan2(double, double) throw(); extern "C" double __atan2(double, double) throw();


extern "C" __attribute__((__weak__)) double cos(double) throw(); extern "C" double __cos(double) throw();

extern "C" __attribute__((__weak__)) double sin(double) throw(); extern "C" double __sin(double) throw();

extern "C" __attribute__((__weak__)) double tan(double) throw(); extern "C" double __tan(double) throw();




extern "C" __attribute__((__weak__)) double cosh(double) throw(); extern "C" double __cosh(double) throw();

extern "C" __attribute__((__weak__)) double sinh(double) throw(); extern "C" double __sinh(double) throw();

extern "C" __attribute__((__weak__)) double tanh(double) throw(); extern "C" double __tanh(double) throw();




extern "C" __attribute__((__weak__)) void sincos(double, double *, double *) throw(); extern "C" void __sincos(double, double *, double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double acosh(double) throw(); extern "C" double __acosh(double) throw();

extern "C" __attribute__((__weak__)) double asinh(double) throw(); extern "C" double __asinh(double) throw();

extern "C" __attribute__((__weak__)) double atanh(double) throw(); extern "C" double __atanh(double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double exp(double) throw(); extern "C" double __exp(double) throw();


extern "C" __attribute__((__weak__)) double frexp(double, int *) throw(); extern "C" double __frexp(double, int *) throw();


extern "C" __attribute__((__weak__)) double ldexp(double, int) throw(); extern "C" double __ldexp(double, int) throw();


extern "C" __attribute__((__weak__)) double log(double) throw(); extern "C" double __log(double) throw();


extern "C" __attribute__((__weak__)) double log10(double) throw(); extern "C" double __log10(double) throw();


extern "C" __attribute__((__weak__)) double modf(double, double *) throw(); extern "C" double __modf(double, double *) throw();




extern "C" __attribute__((__weak__)) double exp10(double) throw(); extern "C" double __exp10(double) throw();

extern "C" double pow10(double) throw(); extern "C" double __pow10(double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double expm1(double) throw(); extern "C" double __expm1(double) throw();


extern "C" __attribute__((__weak__)) double log1p(double) throw(); extern "C" double __log1p(double) throw();


extern "C" __attribute__((__weak__)) double logb(double) throw(); extern "C" double __logb(double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double exp2(double) throw(); extern "C" double __exp2(double) throw();


extern "C" __attribute__((__weak__)) double log2(double) throw(); extern "C" double __log2(double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double pow(double, double) throw(); extern "C" double __pow(double, double) throw();


extern "C" __attribute__((__weak__)) double sqrt(double) throw(); extern "C" double __sqrt(double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double hypot(double, double) throw(); extern "C" double __hypot(double, double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double cbrt(double) throw(); extern "C" double __cbrt(double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__)); extern "C" double __ceil(double) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__)); extern "C" double __fabs(double) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__)); extern "C" double __floor(double) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) double fmod(double, double) throw(); extern "C" double __fmod(double, double) throw();




extern "C" __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinf(double) throw() __attribute__((__const__));


extern "C" int finite(double) throw() __attribute__((__const__));


extern "C" double drem(double, double) throw(); extern "C" double __drem(double, double) throw();



extern "C" double significand(double) throw(); extern "C" double __significand(double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__)); extern "C" double __copysign(double, double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__)); extern "C" double __nan(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__));



extern "C" int isnan(double) throw() __attribute__((__const__));


extern "C" double j0(double) throw(); extern "C" double __j0(double) throw();
extern "C" double j1(double) throw(); extern "C" double __j1(double) throw();
extern "C" double jn(int, double) throw(); extern "C" double __jn(int, double) throw();
extern "C" double y0(double) throw(); extern "C" double __y0(double) throw();
extern "C" double y1(double) throw(); extern "C" double __y1(double) throw();
extern "C" double yn(int, double) throw(); extern "C" double __yn(int, double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double erf(double) throw(); extern "C" double __erf(double) throw();
extern "C" __attribute__((__weak__)) double erfc(double) throw(); extern "C" double __erfc(double) throw();
extern "C" __attribute__((__weak__)) double lgamma(double) throw(); extern "C" double __lgamma(double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double tgamma(double) throw(); extern "C" double __tgamma(double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" double gamma(double) throw(); extern "C" double __gamma(double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" double lgamma_r(double, int *) throw(); extern "C" double __lgamma_r(double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double rint(double) throw(); extern "C" double __rint(double) throw();


extern "C" __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__)); extern "C" double __nextafter(double, double) throw() __attribute__((__const__));

extern "C" double nexttoward(double, long double) throw() __attribute__((__const__)); extern "C" double __nexttoward(double, long double) throw() __attribute__((__const__));



extern "C" __attribute__((__weak__)) double remainder(double, double) throw(); extern "C" double __remainder(double, double) throw();



extern "C" __attribute__((__weak__)) double scalbn(double, int) throw(); extern "C" double __scalbn(double, int) throw();



extern "C" __attribute__((__weak__)) int ilogb(double) throw(); extern "C" int __ilogb(double) throw();




extern "C" __attribute__((__weak__)) double scalbln(double, long) throw(); extern "C" double __scalbln(double, long) throw();



extern "C" __attribute__((__weak__)) double nearbyint(double) throw(); extern "C" double __nearbyint(double) throw();



extern "C" __attribute__((__weak__)) double round(double) throw() __attribute__((__const__)); extern "C" double __round(double) throw() __attribute__((__const__));



extern "C" __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__)); extern "C" double __trunc(double) throw() __attribute__((__const__));




extern "C" __attribute__((__weak__)) double remquo(double, double, int *) throw(); extern "C" double __remquo(double, double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lrint(double) throw(); extern "C" long __lrint(double) throw();
extern "C" __attribute__((__weak__)) long long llrint(double) throw(); extern "C" long long __llrint(double) throw();



extern "C" __attribute__((__weak__)) long lround(double) throw(); extern "C" long __lround(double) throw();
extern "C" __attribute__((__weak__)) long long llround(double) throw(); extern "C" long long __llround(double) throw();



extern "C" __attribute__((__weak__)) double fdim(double, double) throw(); extern "C" double __fdim(double, double) throw();


extern "C" __attribute__((__weak__)) double fmax(double, double) throw(); extern "C" double __fmax(double, double) throw();


extern "C" __attribute__((__weak__)) double fmin(double, double) throw(); extern "C" double __fmin(double, double) throw();



extern "C" int __fpclassify(double) throw() __attribute__((__const__));



extern "C" { extern inline __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }




extern "C" __attribute__((__weak__)) double fma(double, double, double) throw(); extern "C" double __fma(double, double, double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" double scalb(double, double) throw(); extern "C" double __scalb(double, double) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float acosf(float) throw(); extern "C" float __acosf(float) throw();

extern "C" __attribute__((__weak__)) float asinf(float) throw(); extern "C" float __asinf(float) throw();

extern "C" __attribute__((__weak__)) float atanf(float) throw(); extern "C" float __atanf(float) throw();

extern "C" __attribute__((__weak__)) float atan2f(float, float) throw(); extern "C" float __atan2f(float, float) throw();


extern "C" __attribute__((__weak__)) float cosf(float) throw();

extern "C" __attribute__((__weak__)) float sinf(float) throw();

extern "C" __attribute__((__weak__)) float tanf(float) throw();




extern "C" __attribute__((__weak__)) float coshf(float) throw(); extern "C" float __coshf(float) throw();

extern "C" __attribute__((__weak__)) float sinhf(float) throw(); extern "C" float __sinhf(float) throw();

extern "C" __attribute__((__weak__)) float tanhf(float) throw(); extern "C" float __tanhf(float) throw();




extern "C" __attribute__((__weak__)) void sincosf(float, float *, float *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float acoshf(float) throw(); extern "C" float __acoshf(float) throw();

extern "C" __attribute__((__weak__)) float asinhf(float) throw(); extern "C" float __asinhf(float) throw();

extern "C" __attribute__((__weak__)) float atanhf(float) throw(); extern "C" float __atanhf(float) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float expf(float) throw();


extern "C" __attribute__((__weak__)) float frexpf(float, int *) throw(); extern "C" float __frexpf(float, int *) throw();


extern "C" __attribute__((__weak__)) float ldexpf(float, int) throw(); extern "C" float __ldexpf(float, int) throw();


extern "C" __attribute__((__weak__)) float logf(float) throw();


extern "C" __attribute__((__weak__)) float log10f(float) throw();


extern "C" __attribute__((__weak__)) float modff(float, float *) throw(); extern "C" float __modff(float, float *) throw();




extern "C" __attribute__((__weak__)) float exp10f(float) throw();

extern "C" float pow10f(float) throw(); extern "C" float __pow10f(float) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float expm1f(float) throw(); extern "C" float __expm1f(float) throw();


extern "C" __attribute__((__weak__)) float log1pf(float) throw(); extern "C" float __log1pf(float) throw();


extern "C" __attribute__((__weak__)) float logbf(float) throw(); extern "C" float __logbf(float) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float exp2f(float) throw(); extern "C" float __exp2f(float) throw();


extern "C" __attribute__((__weak__)) float log2f(float) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float powf(float, float) throw();


extern "C" __attribute__((__weak__)) float sqrtf(float) throw(); extern "C" float __sqrtf(float) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float hypotf(float, float) throw(); extern "C" float __hypotf(float, float) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float cbrtf(float) throw(); extern "C" float __cbrtf(float) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__)); extern "C" float __ceilf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__)); extern "C" float __fabsf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__)); extern "C" float __floorf(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) float fmodf(float, float) throw(); extern "C" float __fmodf(float, float) throw();




extern "C" __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinff(float) throw() __attribute__((__const__));


extern "C" int finitef(float) throw() __attribute__((__const__));


extern "C" float dremf(float, float) throw(); extern "C" float __dremf(float, float) throw();



extern "C" float significandf(float) throw(); extern "C" float __significandf(float) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__)); extern "C" float __copysignf(float, float) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__)); extern "C" float __nanf(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__));



extern "C" int isnanf(float) throw() __attribute__((__const__));


extern "C" float j0f(float) throw(); extern "C" float __j0f(float) throw();
extern "C" float j1f(float) throw(); extern "C" float __j1f(float) throw();
extern "C" float jnf(int, float) throw(); extern "C" float __jnf(int, float) throw();
extern "C" float y0f(float) throw(); extern "C" float __y0f(float) throw();
extern "C" float y1f(float) throw(); extern "C" float __y1f(float) throw();
extern "C" float ynf(int, float) throw(); extern "C" float __ynf(int, float) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float erff(float) throw(); extern "C" float __erff(float) throw();
extern "C" __attribute__((__weak__)) float erfcf(float) throw(); extern "C" float __erfcf(float) throw();
extern "C" __attribute__((__weak__)) float lgammaf(float) throw(); extern "C" float __lgammaf(float) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float tgammaf(float) throw(); extern "C" float __tgammaf(float) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" float gammaf(float) throw(); extern "C" float __gammaf(float) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" float lgammaf_r(float, int *) throw(); extern "C" float __lgammaf_r(float, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float rintf(float) throw(); extern "C" float __rintf(float) throw();


extern "C" __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__)); extern "C" float __nextafterf(float, float) throw() __attribute__((__const__));

extern "C" float nexttowardf(float, long double) throw() __attribute__((__const__)); extern "C" float __nexttowardf(float, long double) throw() __attribute__((__const__));



extern "C" __attribute__((__weak__)) float remainderf(float, float) throw(); extern "C" float __remainderf(float, float) throw();



extern "C" __attribute__((__weak__)) float scalbnf(float, int) throw(); extern "C" float __scalbnf(float, int) throw();



extern "C" __attribute__((__weak__)) int ilogbf(float) throw(); extern "C" int __ilogbf(float) throw();




extern "C" __attribute__((__weak__)) float scalblnf(float, long) throw(); extern "C" float __scalblnf(float, long) throw();



extern "C" __attribute__((__weak__)) float nearbyintf(float) throw(); extern "C" float __nearbyintf(float) throw();



extern "C" __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__)); extern "C" float __roundf(float) throw() __attribute__((__const__));



extern "C" __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__)); extern "C" float __truncf(float) throw() __attribute__((__const__));




extern "C" __attribute__((__weak__)) float remquof(float, float, int *) throw(); extern "C" float __remquof(float, float, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lrintf(float) throw(); extern "C" long __lrintf(float) throw();
extern "C" __attribute__((__weak__)) long long llrintf(float) throw(); extern "C" long long __llrintf(float) throw();



extern "C" __attribute__((__weak__)) long lroundf(float) throw(); extern "C" long __lroundf(float) throw();
extern "C" __attribute__((__weak__)) long long llroundf(float) throw(); extern "C" long long __llroundf(float) throw();



extern "C" __attribute__((__weak__)) float fdimf(float, float) throw(); extern "C" float __fdimf(float, float) throw();


extern "C" __attribute__((__weak__)) float fmaxf(float, float) throw(); extern "C" float __fmaxf(float, float) throw();


extern "C" __attribute__((__weak__)) float fminf(float, float) throw(); extern "C" float __fminf(float, float) throw();



extern "C" int __fpclassifyf(float) throw() __attribute__((__const__));



extern "C" { extern inline __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }




extern "C" __attribute__((__weak__)) float fmaf(float, float, float) throw(); extern "C" float __fmaf(float, float, float) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" float scalbf(float, float) throw(); extern "C" float __scalbf(float, float) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acosl(long double) throw(); extern "C" long double __acosl(long double) throw();

extern "C" long double asinl(long double) throw(); extern "C" long double __asinl(long double) throw();

extern "C" long double atanl(long double) throw(); extern "C" long double __atanl(long double) throw();

extern "C" long double atan2l(long double, long double) throw(); extern "C" long double __atan2l(long double, long double) throw();


extern "C" long double cosl(long double) throw(); extern "C" long double __cosl(long double) throw();

extern "C" long double sinl(long double) throw(); extern "C" long double __sinl(long double) throw();

extern "C" long double tanl(long double) throw(); extern "C" long double __tanl(long double) throw();




extern "C" long double coshl(long double) throw(); extern "C" long double __coshl(long double) throw();

extern "C" long double sinhl(long double) throw(); extern "C" long double __sinhl(long double) throw();

extern "C" long double tanhl(long double) throw(); extern "C" long double __tanhl(long double) throw();




extern "C" void sincosl(long double, long double *, long double *) throw(); extern "C" void __sincosl(long double, long double *, long double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acoshl(long double) throw(); extern "C" long double __acoshl(long double) throw();

extern "C" long double asinhl(long double) throw(); extern "C" long double __asinhl(long double) throw();

extern "C" long double atanhl(long double) throw(); extern "C" long double __atanhl(long double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expl(long double) throw(); extern "C" long double __expl(long double) throw();


extern "C" long double frexpl(long double, int *) throw(); extern "C" long double __frexpl(long double, int *) throw();


extern "C" long double ldexpl(long double, int) throw(); extern "C" long double __ldexpl(long double, int) throw();


extern "C" long double logl(long double) throw(); extern "C" long double __logl(long double) throw();


extern "C" long double log10l(long double) throw(); extern "C" long double __log10l(long double) throw();


extern "C" long double modfl(long double, long double *) throw(); extern "C" long double __modfl(long double, long double *) throw();




extern "C" long double exp10l(long double) throw(); extern "C" long double __exp10l(long double) throw();

extern "C" long double pow10l(long double) throw(); extern "C" long double __pow10l(long double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expm1l(long double) throw(); extern "C" long double __expm1l(long double) throw();


extern "C" long double log1pl(long double) throw(); extern "C" long double __log1pl(long double) throw();


extern "C" long double logbl(long double) throw(); extern "C" long double __logbl(long double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" long double exp2l(long double) throw(); extern "C" long double __exp2l(long double) throw();


extern "C" long double log2l(long double) throw(); extern "C" long double __log2l(long double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" long double powl(long double, long double) throw(); extern "C" long double __powl(long double, long double) throw();


extern "C" long double sqrtl(long double) throw(); extern "C" long double __sqrtl(long double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" long double hypotl(long double, long double) throw(); extern "C" long double __hypotl(long double, long double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" long double cbrtl(long double) throw(); extern "C" long double __cbrtl(long double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ceill(long double) throw() __attribute__((__const__)); extern "C" long double __ceill(long double) throw() __attribute__((__const__));


extern "C" long double fabsl(long double) throw() __attribute__((__const__)); extern "C" long double __fabsl(long double) throw() __attribute__((__const__));


extern "C" long double floorl(long double) throw() __attribute__((__const__)); extern "C" long double __floorl(long double) throw() __attribute__((__const__));


extern "C" long double fmodl(long double, long double) throw(); extern "C" long double __fmodl(long double, long double) throw();




extern "C" __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__));


extern "C" __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinfl(long double) throw() __attribute__((__const__));


extern "C" int finitel(long double) throw() __attribute__((__const__));


extern "C" long double dreml(long double, long double) throw(); extern "C" long double __dreml(long double, long double) throw();



extern "C" long double significandl(long double) throw(); extern "C" long double __significandl(long double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" long double copysignl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __copysignl(long double, long double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nanl(const char *) throw() __attribute__((__const__)); extern "C" long double __nanl(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__));



extern "C" int isnanl(long double) throw() __attribute__((__const__));


extern "C" long double j0l(long double) throw(); extern "C" long double __j0l(long double) throw();
extern "C" long double j1l(long double) throw(); extern "C" long double __j1l(long double) throw();
extern "C" long double jnl(int, long double) throw(); extern "C" long double __jnl(int, long double) throw();
extern "C" long double y0l(long double) throw(); extern "C" long double __y0l(long double) throw();
extern "C" long double y1l(long double) throw(); extern "C" long double __y1l(long double) throw();
extern "C" long double ynl(int, long double) throw(); extern "C" long double __ynl(int, long double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" long double erfl(long double) throw(); extern "C" long double __erfl(long double) throw();
extern "C" long double erfcl(long double) throw(); extern "C" long double __erfcl(long double) throw();
extern "C" long double lgammal(long double) throw(); extern "C" long double __lgammal(long double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tgammal(long double) throw(); extern "C" long double __tgammal(long double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" long double gammal(long double) throw(); extern "C" long double __gammal(long double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" long double lgammal_r(long double, int *) throw(); extern "C" long double __lgammal_r(long double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" long double rintl(long double) throw(); extern "C" long double __rintl(long double) throw();


extern "C" long double nextafterl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nextafterl(long double, long double) throw() __attribute__((__const__));

extern "C" long double nexttowardl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nexttowardl(long double, long double) throw() __attribute__((__const__));



extern "C" long double remainderl(long double, long double) throw(); extern "C" long double __remainderl(long double, long double) throw();



extern "C" long double scalbnl(long double, int) throw(); extern "C" long double __scalbnl(long double, int) throw();



extern "C" int ilogbl(long double) throw(); extern "C" int __ilogbl(long double) throw();




extern "C" long double scalblnl(long double, long) throw(); extern "C" long double __scalblnl(long double, long) throw();



extern "C" long double nearbyintl(long double) throw(); extern "C" long double __nearbyintl(long double) throw();



extern "C" long double roundl(long double) throw() __attribute__((__const__)); extern "C" long double __roundl(long double) throw() __attribute__((__const__));



extern "C" long double truncl(long double) throw() __attribute__((__const__)); extern "C" long double __truncl(long double) throw() __attribute__((__const__));




extern "C" long double remquol(long double, long double, int *) throw(); extern "C" long double __remquol(long double, long double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" long lrintl(long double) throw(); extern "C" long __lrintl(long double) throw();
extern "C" long long llrintl(long double) throw(); extern "C" long long __llrintl(long double) throw();



extern "C" long lroundl(long double) throw(); extern "C" long __lroundl(long double) throw();
extern "C" long long llroundl(long double) throw(); extern "C" long long __llroundl(long double) throw();



extern "C" long double fdiml(long double, long double) throw(); extern "C" long double __fdiml(long double, long double) throw();


extern "C" long double fmaxl(long double, long double) throw(); extern "C" long double __fmaxl(long double, long double) throw();


extern "C" long double fminl(long double, long double) throw(); extern "C" long double __fminl(long double, long double) throw();



extern "C" int __fpclassifyl(long double) throw() __attribute__((__const__));



extern "C" { extern inline __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }




extern "C" long double fmal(long double, long double, long double) throw(); extern "C" long double __fmal(long double, long double, long double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalbl(long double, long double) throw(); extern "C" long double __scalbl(long double, long double) throw();
# 157 "/usr/include/math.h" 3
extern "C" { extern int signgam; }
# 199 "/usr/include/math.h" 3
enum __cuda_FP_NAN {
FP_NAN,

FP_INFINITE,

FP_ZERO,

FP_SUBNORMAL,

FP_NORMAL

};
# 291 "/usr/include/math.h" 3
extern "C" { typedef
# 285 "/usr/include/math.h" 3
enum {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_
} _LIB_VERSION_TYPE; }




extern "C" { extern _LIB_VERSION_TYPE _LIB_VERSION; }
# 307 "/usr/include/math.h" 3
extern "C" { struct __exception {




int type;
char *name;
double arg1;
double arg2;
double retval;
}; }


extern "C" int matherr(__exception *) throw();
# 36 "/usr/include/bits/mathinline.h" 3
extern "C" { inline __attribute__((__weak__)) __attribute__((__gnu_inline__)) __attribute__((__const__)) int __signbitf(float __x) throw()
{
auto union { float __f; int __i; } __u = {__f: __x};
return (__u.__i) < 0;
} }

extern "C" { inline __attribute__((__weak__)) __attribute__((__gnu_inline__)) __attribute__((__const__)) int __signbit(double __x) throw()
{
auto union { double __d; int __i[2]; } __u = {__d: __x};
return ((__u.__i)[1]) < 0;
} }

extern "C" { inline __attribute__((__weak__)) __attribute__((__gnu_inline__)) __attribute__((__const__)) int __signbitl(long double __x) throw()
{
auto union { long double __l; int __i[3]; } __u = {__l: __x};
return (((__u.__i)[2]) & 32768) != 0;
} }
# 67 "/usr/include/bits/waitstatus.h" 3
extern "C" { union wait {

int w_status;

struct {

unsigned __w_termsig:7;
unsigned __w_coredump:1;
unsigned __w_retcode:8;
unsigned:16;
# 84 "/usr/include/bits/waitstatus.h" 3
} __wait_terminated;

struct {

unsigned __w_stopval:8;
unsigned __w_stopsig:8;
unsigned:16;
# 97 "/usr/include/bits/waitstatus.h" 3
} __wait_stopped;
}; }
# 102 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 99 "/usr/include/stdlib.h" 3
struct div_t {
int quot;
int rem;
} div_t; }
# 110 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 107 "/usr/include/stdlib.h" 3
struct ldiv_t {
long quot;
long rem;
} ldiv_t; }
# 122 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 119 "/usr/include/stdlib.h" 3
struct lldiv_t {
long long quot;
long long rem;
} lldiv_t; }
# 140 "/usr/include/stdlib.h" 3
extern "C" size_t __ctype_get_mb_cur_max() throw();




extern "C" { extern inline double atof(const char *) throw() __attribute__((__pure__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline int atoi(const char *) throw() __attribute__((__pure__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline long atol(const char *) throw() __attribute__((__pure__)) __attribute__((__gnu_inline__)); }
# 158 "/usr/include/stdlib.h" 3
extern "C" { extern inline long long atoll(const char *) throw() __attribute__((__pure__)) __attribute__((__gnu_inline__)); }
# 165 "/usr/include/stdlib.h" 3
extern "C" double strtod(const char *__restrict__, char **__restrict__) throw();
# 173 "/usr/include/stdlib.h" 3
extern "C" float strtof(const char *__restrict__, char **__restrict__) throw();


extern "C" long double strtold(const char *__restrict__, char **__restrict__) throw();
# 184 "/usr/include/stdlib.h" 3
extern "C" long strtol(const char *__restrict__, char **__restrict__, int) throw();



extern "C" unsigned long strtoul(const char *__restrict__, char **__restrict__, int) throw();
# 196 "/usr/include/stdlib.h" 3
extern "C" long long strtoq(const char *__restrict__, char **__restrict__, int) throw();




extern "C" unsigned long long strtouq(const char *__restrict__, char **__restrict__, int) throw();
# 210 "/usr/include/stdlib.h" 3
extern "C" long long strtoll(const char *__restrict__, char **__restrict__, int) throw();




extern "C" unsigned long long strtoull(const char *__restrict__, char **__restrict__, int) throw();
# 240 "/usr/include/stdlib.h" 3
extern "C" long strtol_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();



extern "C" unsigned long strtoul_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 250 "/usr/include/stdlib.h" 3
extern "C" long long strtoll_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 256 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();




extern "C" double strtod_l(const char *__restrict__, char **__restrict__, __locale_t) throw();



extern "C" float strtof_l(const char *__restrict__, char **__restrict__, __locale_t) throw();



extern "C" long double strtold_l(const char *__restrict__, char **__restrict__, __locale_t) throw();
# 279 "/usr/include/stdlib.h" 3
extern "C" { inline __attribute__((__pure__)) __attribute__((__gnu_inline__)) double atof(const char *__nptr) throw()
{
return strtod(__nptr, (char **)__null);
} }

extern "C" { inline __attribute__((__pure__)) __attribute__((__gnu_inline__)) int atoi(const char *__nptr) throw()
{
return (int)strtol(__nptr, (char **)__null, 10);
} }

extern "C" { inline __attribute__((__pure__)) __attribute__((__gnu_inline__)) long atol(const char *__nptr) throw()
{
return strtol(__nptr, (char **)__null, 10);
} }
# 298 "/usr/include/stdlib.h" 3
extern "C" { inline __attribute__((__pure__)) __attribute__((__gnu_inline__)) long long atoll(const char *__nptr) throw()
{
return strtoll(__nptr, (char **)__null, 10);
} }
# 311 "/usr/include/stdlib.h" 3
extern "C" char *l64a(long) throw();


extern "C" long a64l(const char *) throw() __attribute__((__pure__));
# 35 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_char u_char; }
extern "C" { typedef __u_short u_short; }
extern "C" { typedef __u_int u_int; }
extern "C" { typedef __u_long u_long; }
extern "C" { typedef __quad_t quad_t; }
extern "C" { typedef __u_quad_t u_quad_t; }
extern "C" { typedef __fsid_t fsid_t; }




extern "C" { typedef __loff_t loff_t; }



extern "C" { typedef __ino_t ino_t; }
# 57 "/usr/include/sys/types.h" 3
extern "C" { typedef __ino64_t ino64_t; }




extern "C" { typedef __dev_t dev_t; }




extern "C" { typedef __gid_t gid_t; }




extern "C" { typedef __mode_t mode_t; }




extern "C" { typedef __nlink_t nlink_t; }




extern "C" { typedef __uid_t uid_t; }
# 88 "/usr/include/sys/types.h" 3
extern "C" { typedef __off_t off_t; }
# 95 "/usr/include/sys/types.h" 3
extern "C" { typedef __off64_t off64_t; }
# 105 "/usr/include/sys/types.h" 3
extern "C" { typedef __id_t id_t; }




extern "C" { typedef __ssize_t ssize_t; }
# 116 "/usr/include/sys/types.h" 3
extern "C" { typedef __daddr_t daddr_t; }
extern "C" { typedef __caddr_t caddr_t; }
# 123 "/usr/include/sys/types.h" 3
extern "C" { typedef __key_t key_t; }
# 137 "/usr/include/sys/types.h" 3
extern "C" { typedef __useconds_t useconds_t; }



extern "C" { typedef __suseconds_t suseconds_t; }
# 151 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long ulong; }
extern "C" { typedef unsigned short ushort; }
extern "C" { typedef unsigned uint; }
# 195 "/usr/include/sys/types.h" 3
extern "C" { typedef signed char int8_t; }
extern "C" { typedef short int16_t; }
extern "C" { typedef int int32_t; }
extern "C" { typedef long int64_t; }


extern "C" { typedef unsigned char u_int8_t; }
extern "C" { typedef unsigned short u_int16_t; }
extern "C" { typedef unsigned u_int32_t; }
extern "C" { typedef unsigned long u_int64_t; }

extern "C" { typedef int register_t; }
# 24 "/usr/include/bits/sigset.h" 3
extern "C" { typedef int __sig_atomic_t; }
# 32 "/usr/include/bits/sigset.h" 3
extern "C" { typedef
# 30 "/usr/include/bits/sigset.h" 3
struct __sigset_t {
unsigned long __val[((1024) / ((8) * sizeof(unsigned long)))];
} __sigset_t; }
# 38 "/usr/include/sys/select.h" 3
extern "C" { typedef __sigset_t sigset_t; }
# 69 "/usr/include/bits/time.h" 3
extern "C" { struct timeval {

__time_t tv_sec;
__suseconds_t tv_usec;
}; }
# 55 "/usr/include/sys/select.h" 3
extern "C" { typedef long __fd_mask; }
# 78 "/usr/include/sys/select.h" 3
extern "C" { typedef
# 68 "/usr/include/sys/select.h" 3
struct fd_set {



__fd_mask fds_bits[((1024) / ((8) * sizeof(__fd_mask)))];
# 78 "/usr/include/sys/select.h" 3
} fd_set; }
# 85 "/usr/include/sys/select.h" 3
extern "C" { typedef __fd_mask fd_mask; }
# 109 "/usr/include/sys/select.h" 3
extern "C" int select(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, timeval *__restrict__);
# 121 "/usr/include/sys/select.h" 3
extern "C" int pselect(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, const timespec *__restrict__, const __sigset_t *__restrict__);
# 31 "/usr/include/sys/sysmacros.h" 3
extern "C" { extern inline unsigned gnu_dev_major(unsigned long long) throw() __attribute__((__gnu_inline__)); }


extern "C" { extern inline unsigned gnu_dev_minor(unsigned long long) throw() __attribute__((__gnu_inline__)); }


extern "C" { extern inline unsigned long long gnu_dev_makedev(unsigned, unsigned) throw() __attribute__((__gnu_inline__)); }
# 43 "/usr/include/sys/sysmacros.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) unsigned gnu_dev_major(unsigned long long __dev) throw()
{
return ((__dev >> 8) & (4095)) | (((unsigned)(__dev >> 32)) & (~4095));
} }


extern "C" { inline __attribute__((__gnu_inline__)) unsigned gnu_dev_minor(unsigned long long __dev) throw()
{
return (__dev & (255)) | (((unsigned)(__dev >> 12)) & (~255));
} }


extern "C" { inline __attribute__((__gnu_inline__)) unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
{
return (((__minor & (255)) | ((__major & (4095)) << 8)) | (((unsigned long long)(__minor & (~255))) << 12)) | (((unsigned long long)(__major & (~4095))) << 32);


} }
# 228 "/usr/include/sys/types.h" 3
extern "C" { typedef __blksize_t blksize_t; }
# 235 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt_t blkcnt_t; }



extern "C" { typedef __fsblkcnt_t fsblkcnt_t; }



extern "C" { typedef __fsfilcnt_t fsfilcnt_t; }
# 262 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt64_t blkcnt64_t; }
extern "C" { typedef __fsblkcnt64_t fsblkcnt64_t; }
extern "C" { typedef __fsfilcnt64_t fsfilcnt64_t; }
# 50 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned long pthread_t; }
# 57 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 54 "/usr/include/bits/pthreadtypes.h" 3
union pthread_attr_t {
char __size[56];
long __align;
} pthread_attr_t; }
# 65 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 61 "/usr/include/bits/pthreadtypes.h" 3
struct __pthread_internal_list {

__pthread_internal_list *__prev;
__pthread_internal_list *__next;
} __pthread_list_t; }
# 104 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 77 "/usr/include/bits/pthreadtypes.h" 3
union pthread_mutex_t {
struct __pthread_mutex_s {

int __lock;
unsigned __count;
int __owner;

unsigned __nusers;



int __kind;

int __spins;
__pthread_list_t __list;
# 101 "/usr/include/bits/pthreadtypes.h" 3
} __data;
char __size[40];
long __align;
} pthread_mutex_t; }
# 110 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 107 "/usr/include/bits/pthreadtypes.h" 3
union pthread_mutexattr_t {
char __size[4];
int __align;
} pthread_mutexattr_t; }
# 130 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 116 "/usr/include/bits/pthreadtypes.h" 3
union pthread_cond_t {

struct {
int __lock;
unsigned __futex;
unsigned long long __total_seq;
unsigned long long __wakeup_seq;
unsigned long long __woken_seq;
void *__mutex;
unsigned __nwaiters;
unsigned __broadcast_seq;
} __data;
char __size[48];
long long __align;
} pthread_cond_t; }
# 136 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 133 "/usr/include/bits/pthreadtypes.h" 3
union pthread_condattr_t {
char __size[4];
int __align;
} pthread_condattr_t; }



extern "C" { typedef unsigned pthread_key_t; }



extern "C" { typedef int pthread_once_t; }
# 189 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 151 "/usr/include/bits/pthreadtypes.h" 3
union pthread_rwlock_t {


struct {
int __lock;
unsigned __nr_readers;
unsigned __readers_wakeup;
unsigned __writer_wakeup;
unsigned __nr_readers_queued;
unsigned __nr_writers_queued;
int __writer;
int __shared;
unsigned long __pad1;
unsigned long __pad2;


unsigned __flags;
} __data;
# 187 "/usr/include/bits/pthreadtypes.h" 3
char __size[56];
long __align;
} pthread_rwlock_t; }
# 195 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 192 "/usr/include/bits/pthreadtypes.h" 3
union pthread_rwlockattr_t {
char __size[8];
long __align;
} pthread_rwlockattr_t; }
# 201 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef volatile int pthread_spinlock_t; }
# 210 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 207 "/usr/include/bits/pthreadtypes.h" 3
union pthread_barrier_t {
char __size[32];
long __align;
} pthread_barrier_t; }
# 216 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 213 "/usr/include/bits/pthreadtypes.h" 3
union pthread_barrierattr_t {
char __size[4];
int __align;
} pthread_barrierattr_t; }
# 327 "/usr/include/stdlib.h" 3
extern "C" long random() throw();


extern "C" void srandom(unsigned) throw();
# 336 "/usr/include/stdlib.h" 3
extern "C" char *initstate(unsigned, char *, size_t) throw();




extern "C" char *setstate(char *) throw();
# 349 "/usr/include/stdlib.h" 3
extern "C" { struct random_data {

int32_t *fptr;
int32_t *rptr;
int32_t *state;
int rand_type;
int rand_deg;
int rand_sep;
int32_t *end_ptr;
}; }

extern "C" int random_r(random_data *__restrict__, int32_t *__restrict__) throw();


extern "C" int srandom_r(unsigned, random_data *) throw();


extern "C" int initstate_r(unsigned, char *__restrict__, size_t, random_data *__restrict__) throw();




extern "C" int setstate_r(char *__restrict__, random_data *__restrict__) throw();
# 380 "/usr/include/stdlib.h" 3
extern "C" int rand() throw();

extern "C" void srand(unsigned) throw();




extern "C" int rand_r(unsigned *) throw();
# 395 "/usr/include/stdlib.h" 3
extern "C" double drand48() throw();
extern "C" double erand48(unsigned short [3]) throw();


extern "C" long lrand48() throw();
extern "C" long nrand48(unsigned short [3]) throw();



extern "C" long mrand48() throw();
extern "C" long jrand48(unsigned short [3]) throw();



extern "C" void srand48(long) throw();
extern "C" unsigned short *seed48(unsigned short [3]) throw();

extern "C" void lcong48(unsigned short [7]) throw();
# 418 "/usr/include/stdlib.h" 3
extern "C" { struct drand48_data {

unsigned short __x[3];
unsigned short __old_x[3];
unsigned short __c;
unsigned short __init;
unsigned long long __a;
}; }


extern "C" int drand48_r(drand48_data *__restrict__, double *__restrict__) throw();

extern "C" int erand48_r(unsigned short [3], drand48_data *__restrict__, double *__restrict__) throw();




extern "C" int lrand48_r(drand48_data *__restrict__, long *__restrict__) throw();


extern "C" int nrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw();
# 444 "/usr/include/stdlib.h" 3
extern "C" int mrand48_r(drand48_data *__restrict__, long *__restrict__) throw();


extern "C" int jrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw();
# 453 "/usr/include/stdlib.h" 3
extern "C" int srand48_r(long, drand48_data *) throw();


extern "C" int seed48_r(unsigned short [3], drand48_data *) throw();


extern "C" int lcong48_r(unsigned short [7], drand48_data *) throw();
# 471 "/usr/include/stdlib.h" 3
extern "C" void *malloc(size_t) throw() __attribute__((__malloc__));

extern "C" void *calloc(size_t, size_t) throw() __attribute__((__malloc__));
# 485 "/usr/include/stdlib.h" 3
extern "C" void *realloc(void *, size_t) throw();


extern "C" void free(void *) throw();




extern "C" void cfree(void *) throw();
# 33 "/usr/include/alloca.h" 3
extern "C" void *alloca(size_t) throw();
# 502 "/usr/include/stdlib.h" 3
extern "C" void *valloc(size_t) throw() __attribute__((__malloc__));




extern "C" int posix_memalign(void **, size_t, size_t) throw();
# 513 "/usr/include/stdlib.h" 3
extern "C" void abort() throw() __attribute__((__noreturn__));



extern "C" int atexit(void (*)(void)) throw();
# 523 "/usr/include/stdlib.h" 3
extern "C" int on_exit(void (*)(int, void *), void *) throw();
# 531 "/usr/include/stdlib.h" 3
extern "C" void exit(int) throw() __attribute__((__noreturn__));
# 538 "/usr/include/stdlib.h" 3
extern "C" void _Exit(int) throw() __attribute__((__noreturn__));
# 545 "/usr/include/stdlib.h" 3
extern "C" char *getenv(const char *) throw();




extern "C" char *__secure_getenv(const char *) throw();
# 557 "/usr/include/stdlib.h" 3
extern "C" int putenv(char *) throw();
# 563 "/usr/include/stdlib.h" 3
extern "C" int setenv(const char *, const char *, int) throw();



extern "C" int unsetenv(const char *) throw();
# 574 "/usr/include/stdlib.h" 3
extern "C" int clearenv() throw();
# 583 "/usr/include/stdlib.h" 3
extern "C" char *mktemp(char *) throw();
# 594 "/usr/include/stdlib.h" 3
extern "C" int mkstemp(char *);
# 604 "/usr/include/stdlib.h" 3
extern "C" int mkstemp64(char *);
# 614 "/usr/include/stdlib.h" 3
extern "C" char *mkdtemp(char *) throw();
# 625 "/usr/include/stdlib.h" 3
extern "C" int mkostemp(char *, int);
# 635 "/usr/include/stdlib.h" 3
extern "C" int mkostemp64(char *, int);
# 645 "/usr/include/stdlib.h" 3
extern "C" int system(const char *);
# 652 "/usr/include/stdlib.h" 3
extern "C" char *canonicalize_file_name(const char *) throw();
# 662 "/usr/include/stdlib.h" 3
extern "C" { extern inline char *realpath(const char *__restrict__, char *__restrict__) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 670 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_fn_t)(const void *, const void *); }


extern "C" { typedef __compar_fn_t comparison_fn_t; }



extern "C" { typedef int (*__compar_d_fn_t)(const void *, const void *, void *); }
# 683 "/usr/include/stdlib.h" 3
extern "C" void *bsearch(const void *, const void *, size_t, size_t, __compar_fn_t);
# 689 "/usr/include/stdlib.h" 3
extern "C" void qsort(void *, size_t, size_t, __compar_fn_t);


extern "C" void qsort_r(void *, size_t, size_t, __compar_d_fn_t, void *);
# 699 "/usr/include/stdlib.h" 3
extern "C" __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__));
extern "C" __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__));



extern "C" __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__));
# 713 "/usr/include/stdlib.h" 3
extern "C" div_t div(int, int) throw() __attribute__((__const__));

extern "C" ldiv_t ldiv(long, long) throw() __attribute__((__const__));
# 721 "/usr/include/stdlib.h" 3
extern "C" lldiv_t lldiv(long long, long long) throw() __attribute__((__const__));
# 735 "/usr/include/stdlib.h" 3
extern "C" char *ecvt(double, int, int *__restrict__, int *__restrict__) throw();
# 741 "/usr/include/stdlib.h" 3
extern "C" char *fcvt(double, int, int *__restrict__, int *__restrict__) throw();
# 747 "/usr/include/stdlib.h" 3
extern "C" char *gcvt(double, int, char *) throw();
# 753 "/usr/include/stdlib.h" 3
extern "C" char *qecvt(long double, int, int *__restrict__, int *__restrict__) throw();


extern "C" char *qfcvt(long double, int, int *__restrict__, int *__restrict__) throw();


extern "C" char *qgcvt(long double, int, char *) throw();
# 765 "/usr/include/stdlib.h" 3
extern "C" int ecvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();


extern "C" int fcvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();



extern "C" int qecvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();



extern "C" int qfcvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();
# 787 "/usr/include/stdlib.h" 3
extern "C" int mblen(const char *, size_t) throw();


extern "C" int mbtowc(wchar_t *__restrict__, const char *__restrict__, size_t) throw();



extern "C" { extern inline int wctomb(char *, wchar_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" { extern inline size_t mbstowcs(wchar_t *__restrict__, const char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline size_t wcstombs(char *__restrict__, const wchar_t *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 812 "/usr/include/stdlib.h" 3
extern "C" int rpmatch(const char *) throw();
# 823 "/usr/include/stdlib.h" 3
extern "C" int getsubopt(char **__restrict__, char *const *__restrict__, char **__restrict__) throw();
# 832 "/usr/include/stdlib.h" 3
extern "C" void setkey(const char *) throw();
# 840 "/usr/include/stdlib.h" 3
extern "C" int posix_openpt(int);
# 848 "/usr/include/stdlib.h" 3
extern "C" int grantpt(int) throw();



extern "C" int unlockpt(int) throw();




extern "C" char *ptsname(int) throw();
# 864 "/usr/include/stdlib.h" 3
extern "C" { extern inline int ptsname_r(int, char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" int getpt();
# 875 "/usr/include/stdlib.h" 3
extern "C" int getloadavg(double [], int) throw();
# 24 "/usr/include/bits/stdlib.h" 3
extern "C" char *__realpath_chk(const char *__restrict__, char *__restrict__, size_t) throw();


extern "C" char *__realpath_alias(const char *__restrict__, char *__restrict__) throw() __asm__("realpath");


extern "C" char *__realpath_chk_warn(const char *__restrict__, char *__restrict__, size_t) throw() __asm__("__realpath_chk");
# 38 "/usr/include/bits/stdlib.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw()
{
if (__builtin_object_size(__resolved, 2 > 1) != ((size_t)(-1)))
{

if (__builtin_object_size(__resolved, 2 > 1) < (4096)) {
return __realpath_chk_warn(__name, __resolved, __builtin_object_size(__resolved, 2 > 1)); }

return __realpath_chk(__name, __resolved, __builtin_object_size(__resolved, 2 > 1));
}

return __realpath_alias(__name, __resolved);
} }


extern "C" int __ptsname_r_chk(int, char *, size_t, size_t) throw();

extern "C" int __ptsname_r_alias(int, char *, size_t) throw() __asm__("ptsname_r");


extern "C" int __ptsname_r_chk_warn(int, char *, size_t, size_t) throw() __asm__("__ptsname_r_chk");
# 65 "/usr/include/bits/stdlib.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int ptsname_r(int __fd, char *__buf, size_t __buflen) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__buflen))) {
return __ptsname_r_chk(__fd, __buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
if (__buflen > __builtin_object_size(__buf, 2 > 1)) {
return __ptsname_r_chk_warn(__fd, __buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
}
return __ptsname_r_alias(__fd, __buf, __buflen);
} }


extern "C" int __wctomb_chk(char *, wchar_t, size_t) throw();

extern "C" int __wctomb_alias(char *, wchar_t) throw() __asm__("wctomb");



extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int wctomb(char *__s, wchar_t __wchar) throw()
{
# 93 "/usr/include/bits/stdlib.h" 3
if ((__builtin_object_size(__s, 2 > 1) != ((size_t)(-1))) && ((16) > __builtin_object_size(__s, 2 > 1))) {
return __wctomb_chk(__s, __wchar, __builtin_object_size(__s, 2 > 1)); }
return __wctomb_alias(__s, __wchar);
} }


extern "C" size_t __mbstowcs_chk(wchar_t *__restrict__, const char *__restrict__, size_t, size_t) throw();


extern "C" size_t __mbstowcs_alias(wchar_t *__restrict__, const char *__restrict__, size_t) throw() __asm__("mbstowcs");



extern "C" size_t __mbstowcs_chk_warn(wchar_t *__restrict__, const char *__restrict__, size_t, size_t) throw() __asm__("__mbstowcs_chk");
# 114 "/usr/include/bits/stdlib.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) size_t mbstowcs(wchar_t *__restrict__ __dst, const char *__restrict__ __src, size_t __len) throw()

{
if (__builtin_object_size(__dst, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__len))) {
return __mbstowcs_chk(__dst, __src, __len, __builtin_object_size(__dst, 2 > 1) / sizeof(wchar_t)); }


if (__len > (__builtin_object_size(__dst, 2 > 1) / sizeof(wchar_t))) {
return __mbstowcs_chk_warn(__dst, __src, __len, __builtin_object_size(__dst, 2 > 1) / sizeof(wchar_t)); }

}
return __mbstowcs_alias(__dst, __src, __len);
} }


extern "C" size_t __wcstombs_chk(char *__restrict__, const wchar_t *__restrict__, size_t, size_t) throw();


extern "C" size_t __wcstombs_alias(char *__restrict__, const wchar_t *__restrict__, size_t) throw() __asm__("wcstombs");



extern "C" size_t __wcstombs_chk_warn(char *__restrict__, const wchar_t *__restrict__, size_t, size_t) throw() __asm__("__wcstombs_chk");
# 145 "/usr/include/bits/stdlib.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) size_t wcstombs(char *__restrict__ __dst, const wchar_t *__restrict__ __src, size_t __len) throw()

{
if (__builtin_object_size(__dst, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__len))) {
return __wcstombs_chk(__dst, __src, __len, __builtin_object_size(__dst, 2 > 1)); }
if (__len > __builtin_object_size(__dst, 2 > 1)) {
return __wcstombs_chk_warn(__dst, __src, __len, __builtin_object_size(__dst, 2 > 1)); }
}
return __wcstombs_alias(__dst, __src, __len);
} }
# 74 "/usr/include/c++/4.3/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {

template<class _Iterator, class _Container> class __normal_iterator;


}

namespace std __attribute__((visibility("default"))) {

struct __true_type { };
struct __false_type { };

template<bool __T0>
struct __truth_type {
typedef __false_type __type; };


template<> struct __truth_type< true> {
typedef __true_type __type; };



template<class _Sp, class _Tp>
struct __traitor {

enum __cuda___value { __value = (((bool)_Sp::__value) || ((bool)_Tp::__value))};
typedef typename __truth_type< (((bool)_Sp::__value) || ((bool)_Tp::__value))> ::__type __type;
};



template<class _Sp, class _Tp>
struct __traitand {

enum __cuda___value { __value = (((bool)_Sp::__value) && ((bool)_Tp::__value))};
typedef typename __truth_type< (((bool)_Sp::__value) && ((bool)_Tp::__value))> ::__type __type;
};


template<class , class >
struct __are_same {

enum __cuda___value { __value};
typedef __false_type __type;
};

template<class _Tp>
struct __are_same< _Tp, _Tp> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<class _Tp>
struct __is_void {

enum __cuda___value { __value};
typedef __false_type __type;
};


template<> struct __is_void< void> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_integer {

enum __cuda___value { __value};
typedef __false_type __type;
};
# 156 "/usr/include/c++/4.3/bits/cpp_type_traits.h" 3
template<> struct __is_integer< bool> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< signed char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< unsigned char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};



template<> struct __is_integer< wchar_t> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};



template<> struct __is_integer< short> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< unsigned short> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< int> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< unsigned> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< long> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< unsigned long> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< long long> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_integer< unsigned long long> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_floating {

enum __cuda___value { __value};
typedef __false_type __type;
};



template<> struct __is_floating< float> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_floating< double> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_floating< long double> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_pointer {

enum __cuda___value { __value};
typedef __false_type __type;
};

template<class _Tp>
struct __is_pointer< _Tp *> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_normal_iterator {

enum __cuda___value { __value};
typedef __false_type __type;
};

template<class _Iterator, class _Container>
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> > {


enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> > {

};




template<class _Tp>
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> > {

};




template<class _Tp>
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> > {

};




template<class _Tp>
struct __is_char {

enum __cuda___value { __value};
typedef __false_type __type;
};


template<> struct __is_char< char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};



template<> struct __is_char< wchar_t> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<class _Tp>
struct __is_byte {

enum __cuda___value { __value};
typedef __false_type __type;
};


template<> struct __is_byte< char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_byte< signed char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};


template<> struct __is_byte< unsigned char> {

enum __cuda___value { __value = 1};
typedef __true_type __type;
};




template<class _Tp>
struct __is_move_iterator {

enum __cuda___value { __value};
typedef __false_type __type;
};
# 415 "/usr/include/c++/4.3/bits/cpp_type_traits.h" 3
}
# 43 "/usr/include/c++/4.3/ext/type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {


template<bool __T1, class >
struct __enable_if {
};

template<class _Tp>
struct __enable_if< true, _Tp> {
typedef _Tp __type; };



template<bool _Cond, class _Iftrue, class _Iffalse>
struct __conditional_type {
typedef _Iftrue __type; };

template<class _Iftrue, class _Iffalse>
struct __conditional_type< false, _Iftrue, _Iffalse> {
typedef _Iffalse __type; };



template<class _Tp>
struct __add_unsigned {


private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;


public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
};


template<> struct __add_unsigned< char> {
typedef unsigned char __type; };


template<> struct __add_unsigned< signed char> {
typedef unsigned char __type; };


template<> struct __add_unsigned< short> {
typedef unsigned short __type; };


template<> struct __add_unsigned< int> {
typedef unsigned __type; };


template<> struct __add_unsigned< long> {
typedef unsigned long __type; };


template<> struct __add_unsigned< long long> {
typedef unsigned long long __type; };



template<> struct __add_unsigned< bool> ;


template<> struct __add_unsigned< wchar_t> ;



template<class _Tp>
struct __remove_unsigned {


private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;


public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
};


template<> struct __remove_unsigned< char> {
typedef signed char __type; };


template<> struct __remove_unsigned< unsigned char> {
typedef signed char __type; };


template<> struct __remove_unsigned< unsigned short> {
typedef short __type; };


template<> struct __remove_unsigned< unsigned> {
typedef int __type; };


template<> struct __remove_unsigned< unsigned long> {
typedef long __type; };


template<> struct __remove_unsigned< unsigned long long> {
typedef long long __type; };



template<> struct __remove_unsigned< bool> ;


template<> struct __remove_unsigned< wchar_t> ;



template < typename _Type >
    inline bool
    __is_null_pointer ( _Type * __ptr )
    { return __ptr == 0; }

template < typename _Type >
    inline bool
    __is_null_pointer ( _Type )
    { return false; }



template<class _Tp, bool __T2 = std::__is_integer< _Tp> ::__value>
struct __promote {
typedef double __type; };

template<class _Tp>
struct __promote< _Tp, false> {
typedef _Tp __type; };

template<class _Tp, class _Up>
struct __promote_2 {


private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;


public: typedef __typeof__((__type1() + __type2())) __type;
};

template<class _Tp, class _Up, class _Vp>
struct __promote_3 {


private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;


public: typedef __typeof__(((__type1() + __type2()) + __type3())) __type;
};

template<class _Tp, class _Up, class _Vp, class _Wp>
struct __promote_4 {


private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
typedef typename __promote< _Wp, std::__is_integer< _Wp> ::__value> ::__type __type4;


public: typedef __typeof__((((__type1() + __type2()) + __type3()) + __type4())) __type;
};

}
# 82 "/usr/include/c++/4.3/cmath" 3
namespace std __attribute__((visibility("default"))) {



template < typename _Tp >
    _Tp __cmath_power ( _Tp, unsigned int );

template < typename _Tp >
    inline _Tp
    __pow_helper ( _Tp __x, int __n )
    {
      return __n < 0
        ? _Tp ( 1 ) / __cmath_power ( __x, - __n )
        : __cmath_power ( __x, __n );
    }


inline double abs(double __x)
{ return __builtin_fabs(__x); }


inline float abs(float __x)
{ return __builtin_fabsf(__x); }


inline long double abs(long double __x)
{ return __builtin_fabsl(__x); }

using ::acos;


inline float acos(float __x)
{ return __builtin_acosf(__x); }


inline long double acos(long double __x)
{ return __builtin_acosl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    acos ( _Tp __x )
    { return __builtin_acos ( __x ); }

using ::asin;


inline float asin(float __x)
{ return __builtin_asinf(__x); }


inline long double asin(long double __x)
{ return __builtin_asinl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    asin ( _Tp __x )
    { return __builtin_asin ( __x ); }

using ::atan;


inline float atan(float __x)
{ return __builtin_atanf(__x); }


inline long double atan(long double __x)
{ return __builtin_atanl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    atan ( _Tp __x )
    { return __builtin_atan ( __x ); }

using ::atan2;


inline float atan2(float __y, float __x)
{ return __builtin_atan2f(__y, __x); }


inline long double atan2(long double __y, long double __x)
{ return __builtin_atan2l(__y, __x); }

template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __traitand < __is_arithmetic < _Tp >,
            __is_arithmetic < _Up > > :: __value,
        _Tp > :: __type, _Up > :: __type
    atan2 ( _Tp __y, _Up __x )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return atan2 ( __type ( __y ), __type ( __x ) );
    }

using ::ceil;


inline float ceil(float __x)
{ return __builtin_ceilf(__x); }


inline long double ceil(long double __x)
{ return __builtin_ceill(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    ceil ( _Tp __x )
    { return __builtin_ceil ( __x ); }

using ::cos;


inline float cos(float __x)
{ return __builtin_cosf(__x); }


inline long double cos(long double __x)
{ return __builtin_cosl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cos ( _Tp __x )
    { return __builtin_cos ( __x ); }

using ::cosh;


inline float cosh(float __x)
{ return __builtin_coshf(__x); }


inline long double cosh(long double __x)
{ return __builtin_coshl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cosh ( _Tp __x )
    { return __builtin_cosh ( __x ); }

using ::exp;


inline float exp(float __x)
{ return __builtin_expf(__x); }


inline long double exp(long double __x)
{ return __builtin_expl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    exp ( _Tp __x )
    { return __builtin_exp ( __x ); }

using ::fabs;


inline float fabs(float __x)
{ return __builtin_fabsf(__x); }


inline long double fabs(long double __x)
{ return __builtin_fabsl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    fabs ( _Tp __x )
    { return __builtin_fabs ( __x ); }

using ::floor;


inline float floor(float __x)
{ return __builtin_floorf(__x); }


inline long double floor(long double __x)
{ return __builtin_floorl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    floor ( _Tp __x )
    { return __builtin_floor ( __x ); }

using ::fmod;


inline float fmod(float __x, float __y)
{ return __builtin_fmodf(__x, __y); }


inline long double fmod(long double __x, long double __y)
{ return __builtin_fmodl(__x, __y); }

using ::frexp;


inline float frexp(float __x, int *__exp)
{ return __builtin_frexpf(__x, __exp); }


inline long double frexp(long double __x, int *__exp)
{ return __builtin_frexpl(__x, __exp); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    frexp ( _Tp __x, int * __exp )
    { return __builtin_frexp ( __x, __exp ); }

using ::ldexp;


inline float ldexp(float __x, int __exp)
{ return __builtin_ldexpf(__x, __exp); }


inline long double ldexp(long double __x, int __exp)
{ return __builtin_ldexpl(__x, __exp); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
  ldexp ( _Tp __x, int __exp )
  { return __builtin_ldexp ( __x, __exp ); }

using ::log;


inline float log(float __x)
{ return __builtin_logf(__x); }


inline long double log(long double __x)
{ return __builtin_logl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log ( _Tp __x )
    { return __builtin_log ( __x ); }

using ::log10;


inline float log10(float __x)
{ return __builtin_log10f(__x); }


inline long double log10(long double __x)
{ return __builtin_log10l(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log10 ( _Tp __x )
    { return __builtin_log10 ( __x ); }

using ::modf;


inline float modf(float __x, float *__iptr)
{ return __builtin_modff(__x, __iptr); }


inline long double modf(long double __x, long double *__iptr)
{ return __builtin_modfl(__x, __iptr); }

using ::pow;


inline float pow(float __x, float __y)
{ return __builtin_powf(__x, __y); }


inline long double pow(long double __x, long double __y)
{ return __builtin_powl(__x, __y); }



inline double pow(double __x, int __i)
{ return __builtin_powi(__x, __i); }


inline float pow(float __x, int __n)
{ return __builtin_powif(__x, __n); }


inline long double pow(long double __x, int __n)
{ return __builtin_powil(__x, __n); }

template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __traitand < __is_arithmetic < _Tp >,
            __is_arithmetic < _Up > > :: __value,
        _Tp > :: __type, _Up > :: __type
    pow ( _Tp __x, _Up __y )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return pow ( __type ( __x ), __type ( __y ) );
    }

using ::sin;


inline float sin(float __x)
{ return __builtin_sinf(__x); }


inline long double sin(long double __x)
{ return __builtin_sinl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sin ( _Tp __x )
    { return __builtin_sin ( __x ); }

using ::sinh;


inline float sinh(float __x)
{ return __builtin_sinhf(__x); }


inline long double sinh(long double __x)
{ return __builtin_sinhl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sinh ( _Tp __x )
    { return __builtin_sinh ( __x ); }

using ::sqrt;


inline float sqrt(float __x)
{ return __builtin_sqrtf(__x); }


inline long double sqrt(long double __x)
{ return __builtin_sqrtl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sqrt ( _Tp __x )
    { return __builtin_sqrt ( __x ); }

using ::tan;


inline float tan(float __x)
{ return __builtin_tanf(__x); }


inline long double tan(long double __x)
{ return __builtin_tanl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tan ( _Tp __x )
    { return __builtin_tan ( __x ); }

using ::tanh;


inline float tanh(float __x)
{ return __builtin_tanhf(__x); }


inline long double tanh(long double __x)
{ return __builtin_tanhl(__x); }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tanh ( _Tp __x )
    { return __builtin_tanh ( __x ); }

}
# 483 "/usr/include/c++/4.3/cmath" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {

template < typename _Tp >
    inline int
    __capture_fpclassify ( _Tp __f ) { return ( sizeof ( __f ) == sizeof ( float ) ? __fpclassifyf ( __f ) : sizeof ( __f ) == sizeof ( double ) ? __fpclassify ( __f ) : __fpclassifyl ( __f ) ); }

}
# 505 "/usr/include/c++/4.3/cmath" 3
namespace std __attribute__((visibility("default"))) {

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    fpclassify ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return :: __gnu_cxx :: __capture_fpclassify ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isfinite ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isfinite ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isinf ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isinf ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnan ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnan ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnormal ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnormal ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    signbit ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_signbit ( __type ( __f ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreater ( __type ( __f1 ), __type ( __f2 ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreaterequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreaterequal ( __type ( __f1 ), __type ( __f2 ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isless ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isless ( __type ( __f1 ), __type ( __f2 ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessequal ( __type ( __f1 ), __type ( __f2 ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessgreater ( __type ( __f1 ), __type ( __f2 ) );
    }

template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isunordered ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isunordered ( __type ( __f1 ), __type ( __f2 ) );
    }

}
# 40 "/usr/include/c++/4.3/bits/cmath.tcc" 3
namespace std __attribute__((visibility("default"))) {

template < typename _Tp >
    inline _Tp
    __cmath_power ( _Tp __x, unsigned int __n )
    {
      _Tp __y = __n % 2 ? __x : _Tp ( 1 );

      while ( __n >>= 1 )
        {
          __x = __x * __x;
          if ( __n % 2 )
            __y = __y * __x;
        }

      return __y;
    }

}
# 54 "/usr/include/c++/4.3/cstddef" 3
namespace std __attribute__((visibility("default"))) {

using ::ptrdiff_t;
using ::size_t;

}
# 105 "/usr/include/c++/4.3/cstdlib" 3
namespace std __attribute__((visibility("default"))) {

using ::div_t;
using ::ldiv_t;

using ::abort;
using ::abs;
using ::atexit;
using ::atof;
using ::atoi;
using ::atol;
using ::bsearch;
using ::calloc;
using ::div;
using ::exit;
using ::free;
using ::getenv;
using ::labs;
using ::ldiv;
using ::malloc;

using ::mblen;
using ::mbstowcs;
using ::mbtowc;

using ::qsort;
using ::rand;
using ::realloc;
using ::srand;
using ::strtod;
using ::strtol;
using ::strtoul;
using ::system;

using ::wcstombs;
using ::wctomb;



inline long abs(long __i) { return labs(__i); }


inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); }

}
# 162 "/usr/include/c++/4.3/cstdlib" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {


using ::lldiv_t;
# 171 "/usr/include/c++/4.3/cstdlib" 3
using ::_Exit;



inline long long abs(long long __x) { return (__x >= (0)) ? __x : (-__x); }


using ::llabs;


inline lldiv_t div(long long __n, long long __d)
{ auto lldiv_t __q; (__q.quot) = __n / __d; (__q.rem) = __n % __d; return __q; }

using ::lldiv;
# 195 "/usr/include/c++/4.3/cstdlib" 3
using ::atoll;
using ::strtoll;
using ::strtoull;

using ::strtof;
using ::strtold;

}

namespace std __attribute__((visibility("default"))) {


using __gnu_cxx::lldiv_t;

using __gnu_cxx::_Exit;
using __gnu_cxx::abs;

using __gnu_cxx::llabs;
using __gnu_cxx::div;
using __gnu_cxx::lldiv;

using __gnu_cxx::atoll;
using __gnu_cxx::strtof;
using __gnu_cxx::strtoll;
using __gnu_cxx::strtoull;
using __gnu_cxx::strtold;

}
# 434 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" { extern inline __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__gnu_inline__)) __attribute__((__const__)); }

extern "C" __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__));

extern "C" __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__));
# 448 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__));
# 473 "/usr/local/cuda/bin/../include/math_functions.h"
namespace __gnu_cxx {

extern inline long long abs(long long) __attribute__((visibility("default")));
}

namespace std {

template<class T> extern inline T __pow_helper(T, int);
template<class T> extern inline T __cmath_power(T, unsigned);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;
using std::pow;
using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;
# 560 "/usr/local/cuda/bin/../include/math_functions.h"
namespace std {


extern inline long abs(long) __attribute__((visibility("default")));
extern inline float abs(float) __attribute__((visibility("default")));
extern inline double abs(double) __attribute__((visibility("default")));
extern inline float fabs(float) __attribute__((visibility("default")));
extern inline float ceil(float) __attribute__((visibility("default")));
extern inline float floor(float) __attribute__((visibility("default")));
extern inline float sqrt(float) __attribute__((visibility("default")));
extern inline float pow(float, float) __attribute__((visibility("default")));
extern inline float pow(float, int) __attribute__((visibility("default")));
extern inline double pow(double, int) __attribute__((visibility("default")));
extern inline float log(float) __attribute__((visibility("default")));
extern inline float log10(float) __attribute__((visibility("default")));
extern inline float fmod(float, float) __attribute__((visibility("default")));
extern inline float modf(float, float *) __attribute__((visibility("default")));
extern inline float exp(float) __attribute__((visibility("default")));
extern inline float frexp(float, int *) __attribute__((visibility("default")));
extern inline float ldexp(float, int) __attribute__((visibility("default")));
extern inline float asin(float) __attribute__((visibility("default")));
extern inline float sin(float) __attribute__((visibility("default")));
extern inline float sinh(float) __attribute__((visibility("default")));
extern inline float acos(float) __attribute__((visibility("default")));
extern inline float cos(float) __attribute__((visibility("default")));
extern inline float cosh(float) __attribute__((visibility("default")));
extern inline float atan(float) __attribute__((visibility("default")));
extern inline float atan2(float, float) __attribute__((visibility("default")));
extern inline float tan(float) __attribute__((visibility("default")));
extern inline float tanh(float) __attribute__((visibility("default")));


}


static inline float logb(float a)
{
return logbf(a);
}

static inline int ilogb(float a)
{
return ilogbf(a);
}

static inline float scalbn(float a, int b)
{
return scalbnf(a, b);
}

static inline float scalbln(float a, long b)
{
return scalblnf(a, b);
}

static inline float exp2(float a)
{
return exp2f(a);
}

static inline float exp10(float a)
{
return exp10f(a);
}

static inline float expm1(float a)
{
return expm1f(a);
}

static inline float log2(float a)
{
return log2f(a);
}

static inline float log1p(float a)
{
return log1pf(a);
}

static inline float rsqrt(float a)
{
return rsqrtf(a);
}

static inline float acosh(float a)
{
return acoshf(a);
}

static inline float asinh(float a)
{
return asinhf(a);
}

static inline float atanh(float a)
{
return atanhf(a);
}

static inline float hypot(float a, float b)
{
return hypotf(a, b);
}

static inline float cbrt(float a)
{
return cbrtf(a);
}

static inline void sincos(float a, float *sptr, float *cptr)
{
sincosf(a, sptr, cptr);
}

static inline float erf(float a)
{
return erff(a);
}

static inline float erfinv(float a)
{
return erfinvf(a);
}

static inline float erfc(float a)
{
return erfcf(a);
}

static inline float erfcinv(float a)
{
return erfcinvf(a);
}

static inline float lgamma(float a)
{
return lgammaf(a);
}

static inline float tgamma(float a)
{
return tgammaf(a);
}

static inline float copysign(float a, float b)
{
return copysignf(a, b);
}

static inline double copysign(double a, float b)
{
return copysign(a, (double)b);
}

static inline float copysign(float a, double b)
{
return copysignf(a, (float)b);
}

static inline float nextafter(float a, float b)
{
return nextafterf(a, b);
}

static inline float remainder(float a, float b)
{
return remainderf(a, b);
}

static inline float remquo(float a, float b, int *quo)
{
return remquof(a, b, quo);
}

static inline float round(float a)
{
return roundf(a);
}

static inline long lround(float a)
{
return lroundf(a);
}

static inline long long llround(float a)
{
return llroundf(a);
}

static inline float trunc(float a)
{
return truncf(a);
}

static inline float rint(float a)
{
return rintf(a);
}

static inline long lrint(float a)
{
return lrintf(a);
}

static inline long long llrint(float a)
{
return llrintf(a);
}

static inline float nearbyint(float a)
{
return nearbyintf(a);
}

static inline float fdim(float a, float b)
{
return fdimf(a, b);
}

static inline float fma(float a, float b, float c)
{
return fmaf(a, b, c);
}

static inline unsigned min(unsigned a, unsigned b)
{
return umin(a, b);
}

static inline unsigned min(int a, unsigned b)
{
return umin((unsigned)a, b);
}

static inline unsigned min(unsigned a, int b)
{
return umin(a, (unsigned)b);
}

static inline float min(float a, float b)
{
return fminf(a, b);
}

static inline double min(double a, double b)
{
return fmin(a, b);
}

static inline double min(float a, double b)
{
return fmin((double)a, b);
}

static inline double min(double a, float b)
{
return fmin(a, (double)b);
}

static inline unsigned max(unsigned a, unsigned b)
{
return umax(a, b);
}

static inline unsigned max(int a, unsigned b)
{
return umax((unsigned)a, b);
}

static inline unsigned max(unsigned a, int b)
{
return umax(a, (unsigned)b);
}

static inline float max(float a, float b)
{
return fmaxf(a, b);
}

static inline double max(double a, double b)
{
return fmax(a, b);
}

static inline double max(float a, double b)
{
return fmax((double)a, b);
}

static inline double max(double a, float b)
{
return fmax(a, (double)b);
}
# 59 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
template<class T, int dim = 1, cudaTextureReadMode mode = cudaReadModeElementType>
struct texture : public textureReference {

texture(int norm = 0, cudaTextureFilterMode
fMode = cudaFilterModePoint, cudaTextureAddressMode
aMode = cudaAddressModeClamp)
{
(this->normalized) = norm;
(this->filterMode) = fMode;
((this->addressMode)[0]) = aMode;
((this->addressMode)[1]) = aMode;
((this->addressMode)[2]) = aMode;
(this->channelDesc) = cudaCreateChannelDesc< T> ();
}

texture(int norm, cudaTextureFilterMode
fMode, cudaTextureAddressMode
aMode, cudaChannelFormatDesc
desc)
{
(this->normalized) = norm;
(this->filterMode) = fMode;
((this->addressMode)[0]) = aMode;
((this->addressMode)[1]) = aMode;
((this->addressMode)[2]) = aMode;
(this->channelDesc) = desc;
}
};
# 53 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
# 55 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
# 57 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
# 59 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
# 61 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
# 98 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
cudaSetupArgument(T
arg, size_t
offset)

{
return cudaSetupArgument((const void *)(&arg), sizeof(T), offset);
}
# 120 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbol(char *
symbol, const void *
src, size_t
count, size_t
offset = (0), cudaMemcpyKind
kind = cudaMemcpyHostToDevice)

{
return cudaMemcpyToSymbol((const char *)symbol, src, count, offset, kind);
}

template<class T> inline cudaError_t
cudaMemcpyToSymbol(const T &
symbol, const void *
src, size_t
count, size_t
offset = (0), cudaMemcpyKind
kind = cudaMemcpyHostToDevice)

{
return cudaMemcpyToSymbol((const char *)(&symbol), src, count, offset, kind);
}

static inline cudaError_t cudaMemcpyToSymbolAsync(char *
symbol, const void *
src, size_t
count, size_t
offset, cudaMemcpyKind
kind, cudaStream_t
stream)

{
return cudaMemcpyToSymbolAsync((const char *)symbol, src, count, offset, kind, stream);
}

template<class T> inline cudaError_t
cudaMemcpyToSymbolAsync(const T &
symbol, const void *
src, size_t
count, size_t
offset, cudaMemcpyKind
kind, cudaStream_t
stream)

{
return cudaMemcpyToSymbolAsync((const char *)(&symbol), src, count, offset, kind, stream);
}
# 174 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbol(void *
dst, char *
symbol, size_t
count, size_t
offset = (0), cudaMemcpyKind
kind = cudaMemcpyDeviceToHost)

{
return cudaMemcpyFromSymbol(dst, (const char *)symbol, count, offset, kind);
}

template<class T> inline cudaError_t
cudaMemcpyFromSymbol(void *
dst, const T &
symbol, size_t
count, size_t
offset = (0), cudaMemcpyKind
kind = cudaMemcpyDeviceToHost)

{
return cudaMemcpyFromSymbol(dst, (const char *)(&symbol), count, offset, kind);
}

static inline cudaError_t cudaMemcpyFromSymbolAsync(void *
dst, char *
symbol, size_t
count, size_t
offset, cudaMemcpyKind
kind, cudaStream_t
stream)

{
return cudaMemcpyFromSymbolAsync(dst, (const char *)symbol, count, offset, kind, stream);
}

template<class T> inline cudaError_t
cudaMemcpyFromSymbolAsync(void *
dst, const T &
symbol, size_t
count, size_t
offset, cudaMemcpyKind
kind, cudaStream_t
stream)

{
return cudaMemcpyFromSymbolAsync(dst, (const char *)(&symbol), count, offset, kind, stream);
}

static inline cudaError_t cudaGetSymbolAddress(void **
devPtr, char *
symbol)

{
return cudaGetSymbolAddress(devPtr, (const char *)symbol);
}
# 252 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
cudaGetSymbolAddress(void **
devPtr, const T &
symbol)

{
return cudaGetSymbolAddress(devPtr, (const char *)(&symbol));
}
# 267 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolSize(size_t *
size, char *
symbol)

{
return cudaGetSymbolSize(size, (const char *)symbol);
}
# 296 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
cudaGetSymbolSize(size_t *
size, const T &
symbol)

{
return cudaGetSymbolSize(size, (const char *)(&symbol));
}
# 352 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaBindTexture(size_t *
offset, const texture< T, dim, readMode> &
tex, const void *
devPtr, const cudaChannelFormatDesc &
desc, size_t
size = (((2147483647) * 2U) + 1U))

{
return cudaBindTexture(offset, &tex, devPtr, (&desc), size);
}
# 396 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaBindTexture(size_t *
offset, const texture< T, dim, readMode> &
tex, const void *
devPtr, size_t
size = (((2147483647) * 2U) + 1U))

{
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size);
}
# 451 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaBindTexture2D(size_t *
offset, const texture< T, dim, readMode> &
tex, const void *
devPtr, const cudaChannelFormatDesc &
desc, size_t
width, size_t
height, size_t
pitch)

{
return cudaBindTexture2D(offset, &tex, devPtr, (&desc), width, height, pitch);
}
# 493 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaBindTextureToArray(const texture< T, dim, readMode> &
tex, const cudaArray *
array, const cudaChannelFormatDesc &
desc)

{
return cudaBindTextureToArray(&tex, array, (&desc));
}
# 530 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaBindTextureToArray(const texture< T, dim, readMode> &
tex, const cudaArray *
array)

{
auto cudaChannelFormatDesc desc;
auto cudaError_t err = cudaGetChannelDesc(&desc, array);

return (err == (cudaSuccess)) ? (cudaBindTextureToArray(tex, array, desc)) : err;
}
# 568 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaUnbindTexture(const texture< T, dim, readMode> &
tex)

{
return cudaUnbindTexture(&tex);
}
# 607 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
cudaGetTextureAlignmentOffset(size_t *
offset, const texture< T, dim, readMode> &
tex)

{
return cudaGetTextureAlignmentOffset(offset, &tex);
}
# 649 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
cudaLaunch(T *
entry)

{
return cudaLaunch((const char *)entry);
}
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;



extern "C" { typedef _IO_FILE FILE; }
# 65 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE __FILE; }
# 90 "/usr/include/wchar.h" 3
extern "C" { typedef
# 79 "/usr/include/wchar.h" 3
struct __mbstate_t {
int __count;

union {

unsigned __wch;



char __wchb[4];
} __value;
} __mbstate_t; }
# 26 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 23 "/usr/include/_G_config.h" 3
struct _G_fpos_t {
__off_t __pos;
__mbstate_t __state;
} _G_fpos_t; }




extern "C" { typedef
# 28 "/usr/include/_G_config.h" 3
struct _G_fpos64_t {
__off64_t __pos;
__mbstate_t __state;
} _G_fpos64_t; }
# 53 "/usr/include/_G_config.h" 3
extern "C" { typedef short _G_int16_t; }
extern "C" { typedef int _G_int32_t; }
extern "C" { typedef unsigned short _G_uint16_t; }
extern "C" { typedef unsigned _G_uint32_t; }
# 43 "/usr/lib/gcc/x86_64-linux-gnu/4.3.3/include/stdarg.h" 3
extern "C" { typedef __builtin_va_list __gnuc_va_list; }
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
extern "C" { typedef void _IO_lock_t; }
# 186 "/usr/include/libio.h" 3
extern "C" { struct _IO_marker {
_IO_marker *_next;
_IO_FILE *_sbuf;



int _pos;
# 203 "/usr/include/libio.h" 3
}; }


enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv
};
# 271 "/usr/include/libio.h" 3
extern "C" { struct _IO_FILE {
int _flags;




char *_IO_read_ptr;
char *_IO_read_end;
char *_IO_read_base;
char *_IO_write_base;
char *_IO_write_ptr;
char *_IO_write_end;
char *_IO_buf_base;
char *_IO_buf_end;

char *_IO_save_base;
char *_IO_backup_base;
char *_IO_save_end;

_IO_marker *_markers;

_IO_FILE *_chain;

int _fileno;



int _flags2;

__off_t _old_offset;



unsigned short _cur_column;
signed char _vtable_offset;
char _shortbuf[1];



_IO_lock_t *_lock;
# 319 "/usr/include/libio.h" 3
__off64_t _offset;
# 328 "/usr/include/libio.h" 3
void *__pad1;
void *__pad2;
void *__pad3;
void *__pad4;
size_t __pad5;

int _mode;

char _unused2[((((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t))];

}; }
# 344 "/usr/include/libio.h" 3
struct _IO_FILE_plus;

extern "C" { extern _IO_FILE_plus _IO_2_1_stdin_; }
extern "C" { extern _IO_FILE_plus _IO_2_1_stdout_; }
extern "C" { extern _IO_FILE_plus _IO_2_1_stderr_; }
# 364 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_read_fn(void *, char *, size_t); }
# 372 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_write_fn(void *, const char *, size_t); }
# 381 "/usr/include/libio.h" 3
extern "C" { typedef int __io_seek_fn(void *, __off64_t *, int); }


extern "C" { typedef int __io_close_fn(void *); }




extern "C" { typedef __io_read_fn cookie_read_function_t; }
extern "C" { typedef __io_write_fn cookie_write_function_t; }
extern "C" { typedef __io_seek_fn cookie_seek_function_t; }
extern "C" { typedef __io_close_fn cookie_close_function_t; }
# 401 "/usr/include/libio.h" 3
extern "C" { typedef
# 396 "/usr/include/libio.h" 3
struct _IO_cookie_io_functions_t {
__io_read_fn *read;
__io_write_fn *write;
__io_seek_fn *seek;
__io_close_fn *close;
} _IO_cookie_io_functions_t; }
extern "C" { typedef _IO_cookie_io_functions_t cookie_io_functions_t; }

struct _IO_cookie_file;


extern "C" void _IO_cookie_init(_IO_cookie_file *, int, void *, _IO_cookie_io_functions_t);
# 416 "/usr/include/libio.h" 3
extern "C" int __underflow(_IO_FILE *);
extern "C" int __uflow(_IO_FILE *);
extern "C" int __overflow(_IO_FILE *, int);
# 458 "/usr/include/libio.h" 3
extern "C" int _IO_getc(_IO_FILE *);
extern "C" int _IO_putc(int, _IO_FILE *);
extern "C" int _IO_feof(_IO_FILE *) throw();
extern "C" int _IO_ferror(_IO_FILE *) throw();

extern "C" int _IO_peekc_locked(_IO_FILE *);
# 469 "/usr/include/libio.h" 3
extern "C" void _IO_flockfile(_IO_FILE *) throw();
extern "C" void _IO_funlockfile(_IO_FILE *) throw();
extern "C" int _IO_ftrylockfile(_IO_FILE *) throw();
# 488 "/usr/include/libio.h" 3
extern "C" int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__);

extern "C" int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list);

extern "C" __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t);
extern "C" size_t _IO_sgetn(_IO_FILE *, void *, size_t);

extern "C" __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int);
extern "C" __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int);

extern "C" void _IO_free_backup_area(_IO_FILE *) throw();
# 80 "/usr/include/stdio.h" 3
extern "C" { typedef __gnuc_va_list va_list; }
# 91 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos_t fpos_t; }
# 97 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos64_t fpos64_t; }
# 145 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdin; }
extern "C" { extern _IO_FILE *stdout; }
extern "C" { extern _IO_FILE *stderr; }
# 155 "/usr/include/stdio.h" 3
extern "C" int remove(const char *) throw();

extern "C" int rename(const char *, const char *) throw();




extern "C" int renameat(int, const char *, int, const char *) throw();
# 172 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile();
# 182 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile64();



extern "C" char *tmpnam(char *) throw();
# 192 "/usr/include/stdio.h" 3
extern "C" char *tmpnam_r(char *) throw();
# 204 "/usr/include/stdio.h" 3
extern "C" char *tempnam(const char *, const char *) throw() __attribute__((__malloc__));
# 214 "/usr/include/stdio.h" 3
extern "C" int fclose(FILE *);




extern "C" int fflush(FILE *);
# 229 "/usr/include/stdio.h" 3
extern "C" int fflush_unlocked(FILE *);
# 239 "/usr/include/stdio.h" 3
extern "C" int fcloseall();
# 249 "/usr/include/stdio.h" 3
extern "C" FILE *fopen(const char *__restrict__, const char *__restrict__);
# 255 "/usr/include/stdio.h" 3
extern "C" FILE *freopen(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 274 "/usr/include/stdio.h" 3
extern "C" FILE *fopen64(const char *__restrict__, const char *__restrict__);

extern "C" FILE *freopen64(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 283 "/usr/include/stdio.h" 3
extern "C" FILE *fdopen(int, const char *) throw();
# 289 "/usr/include/stdio.h" 3
extern "C" FILE *fopencookie(void *__restrict__, const char *__restrict__, _IO_cookie_io_functions_t) throw();




extern "C" FILE *fmemopen(void *, size_t, const char *) throw();
# 300 "/usr/include/stdio.h" 3
extern "C" FILE *open_memstream(char **, size_t *) throw();
# 307 "/usr/include/stdio.h" 3
extern "C" void setbuf(FILE *__restrict__, char *__restrict__) throw();



extern "C" int setvbuf(FILE *__restrict__, char *__restrict__, int, size_t) throw();
# 318 "/usr/include/stdio.h" 3
extern "C" void setbuffer(FILE *__restrict__, char *__restrict__, size_t) throw();



extern "C" void setlinebuf(FILE *) throw();
# 331 "/usr/include/stdio.h" 3
extern "C" { extern inline int fprintf(FILE *__restrict__, const char *__restrict__, ...) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 337 "/usr/include/stdio.h" 3
extern "C" { extern inline int printf(const char *__restrict__, ...) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }

extern "C" { extern inline int sprintf(char *__restrict__, const char *__restrict__, ...) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 346 "/usr/include/stdio.h" 3
extern "C" { extern inline int vfprintf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 352 "/usr/include/stdio.h" 3
extern "C" { extern inline int vprintf(const char *__restrict__, __gnuc_va_list) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }

extern "C" { extern inline int vsprintf(char *__restrict__, const char *__restrict__, __gnuc_va_list) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 361 "/usr/include/stdio.h" 3
extern "C" { extern inline int snprintf(char *__restrict__, size_t, const char *__restrict__, ...) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" { extern inline int vsnprintf(char *__restrict__, size_t, const char *__restrict__, __gnuc_va_list) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 374 "/usr/include/stdio.h" 3
extern "C" { extern inline int vasprintf(char **__restrict__, const char *__restrict__, __gnuc_va_list) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline int __asprintf(char **__restrict__, const char *__restrict__, ...) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline int asprintf(char **__restrict__, const char *__restrict__, ...) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 390 "/usr/include/stdio.h" 3
extern "C" { extern inline int vdprintf(int, const char *__restrict__, __gnuc_va_list) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline int dprintf(int, const char *__restrict__, ...) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 403 "/usr/include/stdio.h" 3
extern "C" int fscanf(FILE *__restrict__, const char *__restrict__, ...);
# 409 "/usr/include/stdio.h" 3
extern "C" int scanf(const char *__restrict__, ...);

extern "C" int sscanf(const char *__restrict__, const char *__restrict__, ...) throw();
# 449 "/usr/include/stdio.h" 3
extern "C" int vfscanf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 457 "/usr/include/stdio.h" 3
extern "C" int vscanf(const char *__restrict__, __gnuc_va_list);



extern "C" int vsscanf(const char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 509 "/usr/include/stdio.h" 3
extern "C" int fgetc(FILE *);
extern "C" int getc(FILE *);
# 516 "/usr/include/stdio.h" 3
extern "C" { extern inline int getchar() __attribute__((__gnu_inline__)); }
# 528 "/usr/include/stdio.h" 3
extern "C" { extern inline int getc_unlocked(FILE *) __attribute__((__gnu_inline__)); }
extern "C" { extern inline int getchar_unlocked() __attribute__((__gnu_inline__)); }
# 539 "/usr/include/stdio.h" 3
extern "C" { extern inline int fgetc_unlocked(FILE *) __attribute__((__gnu_inline__)); }
# 551 "/usr/include/stdio.h" 3
extern "C" int fputc(int, FILE *);
extern "C" int putc(int, FILE *);
# 558 "/usr/include/stdio.h" 3
extern "C" { extern inline int putchar(int) __attribute__((__gnu_inline__)); }
# 572 "/usr/include/stdio.h" 3
extern "C" { extern inline int fputc_unlocked(int, FILE *) __attribute__((__gnu_inline__)); }
# 580 "/usr/include/stdio.h" 3
extern "C" { extern inline int putc_unlocked(int, FILE *) __attribute__((__gnu_inline__)); }
extern "C" { extern inline int putchar_unlocked(int) __attribute__((__gnu_inline__)); }
# 588 "/usr/include/stdio.h" 3
extern "C" int getw(FILE *);


extern "C" int putw(int, FILE *);
# 600 "/usr/include/stdio.h" 3
extern "C" { extern inline char *fgets(char *__restrict__, int, FILE *__restrict__) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 608 "/usr/include/stdio.h" 3
extern "C" { extern inline char *gets(char *) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 618 "/usr/include/stdio.h" 3
extern "C" { extern inline char *fgets_unlocked(char *__restrict__, int, FILE *__restrict__) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 634 "/usr/include/stdio.h" 3
extern "C" __ssize_t __getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);


extern "C" __ssize_t getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 647 "/usr/include/stdio.h" 3
extern "C" { extern inline __ssize_t getline(char **__restrict__, size_t *__restrict__, FILE *__restrict__) __attribute__((__gnu_inline__)); }
# 658 "/usr/include/stdio.h" 3
extern "C" int fputs(const char *__restrict__, FILE *__restrict__);
# 664 "/usr/include/stdio.h" 3
extern "C" int puts(const char *);
# 671 "/usr/include/stdio.h" 3
extern "C" int ungetc(int, FILE *);
# 678 "/usr/include/stdio.h" 3
extern "C" { extern inline size_t fread(void *__restrict__, size_t, size_t, FILE *__restrict__) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 684 "/usr/include/stdio.h" 3
extern "C" size_t fwrite(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 695 "/usr/include/stdio.h" 3
extern "C" int fputs_unlocked(const char *__restrict__, FILE *__restrict__);
# 706 "/usr/include/stdio.h" 3
extern "C" { extern inline size_t fread_unlocked(void *__restrict__, size_t, size_t, FILE *__restrict__) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }

extern "C" size_t fwrite_unlocked(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 718 "/usr/include/stdio.h" 3
extern "C" int fseek(FILE *, long, int);




extern "C" long ftell(FILE *);




extern "C" void rewind(FILE *);
# 742 "/usr/include/stdio.h" 3
extern "C" int fseeko(FILE *, __off_t, int);




extern "C" __off_t ftello(FILE *);
# 767 "/usr/include/stdio.h" 3
extern "C" int fgetpos(FILE *__restrict__, fpos_t *__restrict__);




extern "C" int fsetpos(FILE *, const fpos_t *);
# 787 "/usr/include/stdio.h" 3
extern "C" int fseeko64(FILE *, __off64_t, int);
extern "C" __off64_t ftello64(FILE *);
extern "C" int fgetpos64(FILE *__restrict__, fpos64_t *__restrict__);
extern "C" int fsetpos64(FILE *, const fpos64_t *);




extern "C" void clearerr(FILE *) throw();

extern "C" int feof(FILE *) throw();

extern "C" int ferror(FILE *) throw();




extern "C" void clearerr_unlocked(FILE *) throw();
extern "C" { extern inline int feof_unlocked(FILE *) throw() __attribute__((__gnu_inline__)); }
extern "C" { extern inline int ferror_unlocked(FILE *) throw() __attribute__((__gnu_inline__)); }
# 815 "/usr/include/stdio.h" 3
extern "C" void perror(const char *);
# 27 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern int sys_nerr; }
extern "C" { extern const char *const sys_errlist[]; }


extern "C" { extern int _sys_nerr; }
extern "C" { extern const char *const _sys_errlist[]; }
# 827 "/usr/include/stdio.h" 3
extern "C" int fileno(FILE *) throw();




extern "C" int fileno_unlocked(FILE *) throw();
# 842 "/usr/include/stdio.h" 3
extern "C" FILE *popen(const char *, const char *);
# 848 "/usr/include/stdio.h" 3
extern "C" int pclose(FILE *);
# 854 "/usr/include/stdio.h" 3
extern "C" char *ctermid(char *) throw();
# 860 "/usr/include/stdio.h" 3
extern "C" char *cuserid(char *);




struct obstack;


extern "C" { extern inline int obstack_printf(obstack *__restrict__, const char *__restrict__, ...) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }


extern "C" { extern inline int obstack_vprintf(obstack *__restrict__, const char *__restrict__, __gnuc_va_list) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 882 "/usr/include/stdio.h" 3
extern "C" void flockfile(FILE *) throw();



extern "C" int ftrylockfile(FILE *) throw();


extern "C" void funlockfile(FILE *) throw();
# 45 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int getchar()
{
return _IO_getc(stdin);
} }
# 54 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int fgetc_unlocked(FILE *__fp)
{
return (__builtin_expect((__fp->_IO_read_ptr) >= (__fp->_IO_read_end), 0)) ? __uflow(__fp) : (*((unsigned char *)((__fp->_IO_read_ptr)++)));
} }
# 64 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int getc_unlocked(FILE *__fp)
{
return (__builtin_expect((__fp->_IO_read_ptr) >= (__fp->_IO_read_end), 0)) ? __uflow(__fp) : (*((unsigned char *)((__fp->_IO_read_ptr)++)));
} }



extern "C" { inline __attribute__((__gnu_inline__)) int getchar_unlocked()
{
return (__builtin_expect((stdin->_IO_read_ptr) >= (stdin->_IO_read_end), 0)) ? __uflow(stdin) : (*((unsigned char *)((stdin->_IO_read_ptr)++)));
} }
# 80 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int putchar(int __c)
{
return _IO_putc(__c, stdout);
} }
# 89 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int fputc_unlocked(int __c, FILE *__stream)
{
return (__builtin_expect((__stream->_IO_write_ptr) >= (__stream->_IO_write_end), 0)) ? __overflow(__stream, (unsigned char)__c) : ((unsigned char)((*((__stream->_IO_write_ptr)++)) = __c));
} }
# 99 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int putc_unlocked(int __c, FILE *__stream)
{
return (__builtin_expect((__stream->_IO_write_ptr) >= (__stream->_IO_write_end), 0)) ? __overflow(__stream, (unsigned char)__c) : ((unsigned char)((*((__stream->_IO_write_ptr)++)) = __c));
} }



extern "C" { inline __attribute__((__gnu_inline__)) int putchar_unlocked(int __c)
{
return (__builtin_expect((stdout->_IO_write_ptr) >= (stdout->_IO_write_end), 0)) ? __overflow(stdout, (unsigned char)__c) : ((unsigned char)((*((stdout->_IO_write_ptr)++)) = __c));
} }
# 116 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) __ssize_t getline(char **__lineptr, size_t *__n, FILE *__stream)
{
return __getdelim(__lineptr, __n, '\n', __stream);
} }
# 126 "/usr/include/bits/stdio.h" 3
extern "C" { inline __attribute__((__gnu_inline__)) int feof_unlocked(FILE *__stream) throw()
{
return ((__stream->_flags) & 16) != 0;
} }



extern "C" { inline __attribute__((__gnu_inline__)) int ferror_unlocked(FILE *__stream) throw()
{
return ((__stream->_flags) & 32) != 0;
} }
# 24 "/usr/include/bits/stdio2.h" 3
extern "C" int __builtin_va_arg_pack();

extern "C" int __sprintf_chk(char *__restrict__, int, size_t, const char *__restrict__, ...) throw();

extern "C" int __vsprintf_chk(char *__restrict__, int, size_t, const char *__restrict__, __gnuc_va_list) throw();
# 34 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int sprintf(char *__restrict__ __s, const char *__restrict__ __fmt, ...) throw()
{
return __builtin___sprintf_chk(__s, 2 - 1, __builtin_object_size(__s, 2 > 1), __fmt, __builtin_va_arg_pack());

} }
# 46 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vsprintf(char *__restrict__ __s, const char *__restrict__ __fmt, __gnuc_va_list __ap) throw()

{
return __builtin___vsprintf_chk(__s, 2 - 1, __builtin_object_size(__s, 2 > 1), __fmt, __ap);

} }



extern "C" int __snprintf_chk(char *__restrict__, size_t, int, size_t, const char *__restrict__, ...) throw();


extern "C" int __vsnprintf_chk(char *__restrict__, size_t, int, size_t, const char *__restrict__, __gnuc_va_list) throw();
# 64 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int snprintf(char *__restrict__ __s, size_t __n, const char *__restrict__ __fmt, ...) throw()

{
return __builtin___snprintf_chk(__s, __n, 2 - 1, __builtin_object_size(__s, 2 > 1), __fmt, __builtin_va_arg_pack());

} }
# 77 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vsnprintf(char *__restrict__ __s, size_t __n, const char *__restrict__ __fmt, __gnuc_va_list __ap) throw()

{
return __builtin___vsnprintf_chk(__s, __n, 2 - 1, __builtin_object_size(__s, 2 > 1), __fmt, __ap);

} }
# 88 "/usr/include/bits/stdio2.h" 3
extern "C" int __fprintf_chk(FILE *__restrict__, int, const char *__restrict__, ...);

extern "C" int __printf_chk(int, const char *__restrict__, ...);
extern "C" int __vfprintf_chk(FILE *__restrict__, int, const char *__restrict__, __gnuc_va_list);

extern "C" int __vprintf_chk(int, const char *__restrict__, __gnuc_va_list);




extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int fprintf(FILE *__restrict__ __stream, const char *__restrict__ __fmt, ...)
{
return __fprintf_chk(__stream, 2 - 1, __fmt, __builtin_va_arg_pack());

} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int printf(const char *__restrict__ __fmt, ...)
{
return __printf_chk(2 - 1, __fmt, __builtin_va_arg_pack());
} }
# 117 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vprintf(const char *__restrict__ __fmt, __gnuc_va_list __ap)
{

return __vfprintf_chk(stdout, 2 - 1, __fmt, __ap);



} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vfprintf(FILE *__restrict__ __stream, const char *__restrict__
__fmt, __gnuc_va_list __ap)
{
return __vfprintf_chk(__stream, 2 - 1, __fmt, __ap);
} }



extern "C" int __asprintf_chk(char **__restrict__, int, const char *__restrict__, ...) throw();


extern "C" int __vasprintf_chk(char **__restrict__, int, const char *__restrict__, __gnuc_va_list) throw();


extern "C" int __dprintf_chk(int, int, const char *__restrict__, ...);

extern "C" int __vdprintf_chk(int, int, const char *__restrict__, __gnuc_va_list);


extern "C" int __obstack_printf_chk(obstack *__restrict__, int, const char *__restrict__, ...) throw();



extern "C" int __obstack_vprintf_chk(obstack *__restrict__, int, const char *__restrict__, __gnuc_va_list) throw();
# 158 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()
{
return __asprintf_chk(__ptr, 2 - 1, __fmt, __builtin_va_arg_pack());

} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int __asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()

{
return __asprintf_chk(__ptr, 2 - 1, __fmt, __builtin_va_arg_pack());

} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int dprintf(int __fd, const char *__restrict__ __fmt, ...)
{
return __dprintf_chk(__fd, 2 - 1, __fmt, __builtin_va_arg_pack());

} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int obstack_printf(obstack *__restrict__ __obstack, const char *__restrict__ __fmt, ...) throw()

{
return __obstack_printf_chk(__obstack, 2 - 1, __fmt, __builtin_va_arg_pack());

} }
# 198 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vasprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, __gnuc_va_list __ap) throw()

{
return __vasprintf_chk(__ptr, 2 - 1, __fmt, __ap);
} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int vdprintf(int __fd, const char *__restrict__ __fmt, __gnuc_va_list __ap)
{
return __vdprintf_chk(__fd, 2 - 1, __fmt, __ap);
} }


extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int obstack_vprintf(obstack *__restrict__ __obstack, const char *__restrict__ __fmt, __gnuc_va_list __ap) throw()

{
return __obstack_vprintf_chk(__obstack, 2 - 1, __fmt, __ap);

} }
# 222 "/usr/include/bits/stdio2.h" 3
extern "C" char *__gets_chk(char *, size_t);
extern "C" char *__gets_warn(char *) __asm__("gets");




extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *gets(char *__str)
{
if (__builtin_object_size(__str, 2 > 1) != ((size_t)(-1))) {
return __gets_chk(__str, __builtin_object_size(__str, 2 > 1)); }
return __gets_warn(__str);
} }

extern "C" char *__fgets_chk(char *__restrict__, size_t, int, FILE *__restrict__);

extern "C" char *__fgets_alias(char *__restrict__, int, FILE *__restrict__) __asm__("fgets");


extern "C" char *__fgets_chk_warn(char *__restrict__, size_t, int, FILE *__restrict__) __asm__("__fgets_chk");
# 247 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *fgets(char *__restrict__ __s, int __n, FILE *__restrict__ __stream)
{
if (__builtin_object_size(__s, 2 > 1) != ((size_t)(-1)))
{
if ((!(__builtin_constant_p(__n))) || (__n <= 0)) {
return __fgets_chk(__s, __builtin_object_size(__s, 2 > 1), __n, __stream); }

if (((unsigned long)__n) > __builtin_object_size(__s, 2 > 1)) {
return __fgets_chk_warn(__s, __builtin_object_size(__s, 2 > 1), __n, __stream); }
}
return __fgets_alias(__s, __n, __stream);
} }

extern "C" size_t __fread_chk(void *__restrict__, size_t, size_t, size_t, FILE *__restrict__);


extern "C" size_t __fread_alias(void *__restrict__, size_t, size_t, FILE *__restrict__) __asm__("fread");



extern "C" size_t __fread_chk_warn(void *__restrict__, size_t, size_t, size_t, FILE *__restrict__) __asm__("__fread_chk");
# 276 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) size_t fread(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__
__stream)
{
if (__builtin_object_size(__ptr, 0) != ((size_t)(-1)))
{
if (((!(__builtin_constant_p(__size))) || (!(__builtin_constant_p(__n)))) || ((__size | __n) >= (((size_t)1) << (((8) * sizeof(size_t)) / (2))))) {


return __fread_chk(__ptr, __builtin_object_size(__ptr, 0), __size, __n, __stream); }

if ((__size * __n) > __builtin_object_size(__ptr, 0)) {
return __fread_chk_warn(__ptr, __builtin_object_size(__ptr, 0), __size, __n, __stream); }
}
return __fread_alias(__ptr, __size, __n, __stream);
} }


extern "C" char *__fgets_unlocked_chk(char *__restrict__, size_t, int, FILE *__restrict__);

extern "C" char *__fgets_unlocked_alias(char *__restrict__, int, FILE *__restrict__) __asm__("fgets_unlocked");


extern "C" char *__fgets_unlocked_chk_warn(char *__restrict__, size_t, int, FILE *__restrict__) __asm__("__fgets_unlocked_chk");
# 305 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *fgets_unlocked(char *__restrict__ __s, int __n, FILE *__restrict__ __stream)
{
if (__builtin_object_size(__s, 2 > 1) != ((size_t)(-1)))
{
if ((!(__builtin_constant_p(__n))) || (__n <= 0)) {
return __fgets_unlocked_chk(__s, __builtin_object_size(__s, 2 > 1), __n, __stream); }

if (((unsigned long)__n) > __builtin_object_size(__s, 2 > 1)) {
return __fgets_unlocked_chk_warn(__s, __builtin_object_size(__s, 2 > 1), __n, __stream); }
}
return __fgets_unlocked_alias(__s, __n, __stream);
} }




extern "C" size_t __fread_unlocked_chk(void *__restrict__, size_t, size_t, size_t, FILE *__restrict__);


extern "C" size_t __fread_unlocked_alias(void *__restrict__, size_t, size_t, FILE *__restrict__) __asm__("fread_unlocked");



extern "C" size_t __fread_unlocked_chk_warn(void *__restrict__, size_t, size_t, size_t, FILE *__restrict__) __asm__("__fread_unlocked_chk");
# 337 "/usr/include/bits/stdio2.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) size_t fread_unlocked(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__
__stream)
{
if (__builtin_object_size(__ptr, 0) != ((size_t)(-1)))
{
if (((!(__builtin_constant_p(__size))) || (!(__builtin_constant_p(__n)))) || ((__size | __n) >= (((size_t)1) << (((8) * sizeof(size_t)) / (2))))) {


return __fread_unlocked_chk(__ptr, __builtin_object_size(__ptr, 0), __size, __n, __stream); }


if ((__size * __n) > __builtin_object_size(__ptr, 0)) {
return __fread_unlocked_chk_warn(__ptr, __builtin_object_size(__ptr, 0), __size, __n, __stream); }

}


if ((((__builtin_constant_p(__size)) && (__builtin_constant_p(__n))) && ((__size | __n) < (((size_t)1) << (((8) * sizeof(size_t)) / (2))))) && ((__size * __n) <= (8)))



{
auto size_t __cnt = (__size * __n);
auto char *__cptr = ((char *)__ptr);
if (__cnt == (0)) {
return 0; }

for (; __cnt > (0); --__cnt)
{
auto int __c = ((__builtin_expect((__stream->_IO_read_ptr) >= (__stream->_IO_read_end), 0)) ? __uflow(__stream) : (*((unsigned char *)((__stream->_IO_read_ptr)++))));
if (__c == (-1)) {
break; }
(*(__cptr++)) = __c;
}
return (__cptr - (char *)__ptr) / __size;
}

return __fread_unlocked_alias(__ptr, __size, __n, __stream);
} }
# 238 "/usr/include/unistd.h" 3
extern "C" { typedef __intptr_t intptr_t; }
# 245 "/usr/include/unistd.h" 3
extern "C" { typedef __socklen_t socklen_t; }
# 258 "/usr/include/unistd.h" 3
extern "C" int access(const char *, int) throw();




extern "C" int euidaccess(const char *, int) throw();



extern "C" int eaccess(const char *, int) throw();
# 275 "/usr/include/unistd.h" 3
extern "C" int faccessat(int, const char *, int, int) throw();
# 301 "/usr/include/unistd.h" 3
extern "C" __off_t lseek(int, __off_t, int) throw();
# 312 "/usr/include/unistd.h" 3
extern "C" __off64_t lseek64(int, __off64_t, int) throw();
# 320 "/usr/include/unistd.h" 3
extern "C" int close(int);
# 327 "/usr/include/unistd.h" 3
extern "C" { extern inline ssize_t read(int, void *, size_t) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 333 "/usr/include/unistd.h" 3
extern "C" ssize_t write(int, const void *, size_t);
# 343 "/usr/include/unistd.h" 3
extern "C" { extern inline ssize_t pread(int, void *, size_t, __off_t) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 351 "/usr/include/unistd.h" 3
extern "C" ssize_t pwrite(int, const void *, size_t, __off_t);
# 371 "/usr/include/unistd.h" 3
extern "C" { extern inline ssize_t pread64(int, void *, size_t, __off64_t) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" ssize_t pwrite64(int, const void *, size_t, __off64_t);
# 384 "/usr/include/unistd.h" 3
extern "C" int pipe(int [2]) throw();




extern "C" int pipe2(int [2], int) throw();
# 399 "/usr/include/unistd.h" 3
extern "C" unsigned alarm(unsigned) throw();
# 411 "/usr/include/unistd.h" 3
extern "C" unsigned sleep(unsigned);
# 418 "/usr/include/unistd.h" 3
extern "C" __useconds_t ualarm(__useconds_t, __useconds_t) throw();
# 426 "/usr/include/unistd.h" 3
extern "C" int usleep(__useconds_t);
# 435 "/usr/include/unistd.h" 3
extern "C" int pause();



extern "C" int chown(const char *, __uid_t, __gid_t) throw();




extern "C" int fchown(int, __uid_t, __gid_t) throw();




extern "C" int lchown(const char *, __uid_t, __gid_t) throw();
# 457 "/usr/include/unistd.h" 3
extern "C" int fchownat(int, const char *, __uid_t, __gid_t, int) throw();
# 463 "/usr/include/unistd.h" 3
extern "C" int chdir(const char *) throw();



extern "C" int fchdir(int) throw();
# 477 "/usr/include/unistd.h" 3
extern "C" { extern inline char *getcwd(char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 483 "/usr/include/unistd.h" 3
extern "C" char *get_current_dir_name() throw();
# 490 "/usr/include/unistd.h" 3
extern "C" { extern inline char *getwd(char *) throw() __attribute__((__deprecated__)) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 496 "/usr/include/unistd.h" 3
extern "C" int dup(int) throw();


extern "C" int dup2(int, int) throw();




extern "C" int dup3(int, int, int) throw();



extern "C" { extern char **__environ; }

extern "C" { extern char **environ; }
# 516 "/usr/include/unistd.h" 3
extern "C" int execve(const char *, char *const [], char *const []) throw();
# 522 "/usr/include/unistd.h" 3
extern "C" int fexecve(int, char *const [], char *const []) throw();
# 528 "/usr/include/unistd.h" 3
extern "C" int execv(const char *, char *const []) throw();




extern "C" int execle(const char *, const char *, ...) throw();




extern "C" int execl(const char *, const char *, ...) throw();




extern "C" int execvp(const char *, char *const []) throw();
# 549 "/usr/include/unistd.h" 3
extern "C" int execlp(const char *, const char *, ...) throw();
# 555 "/usr/include/unistd.h" 3
extern "C" int nice(int) throw();




extern "C" void _exit(int) __attribute__((__noreturn__));
# 27 "/usr/include/bits/confname.h" 3
enum __cuda__PC_LINK_MAX {
_PC_LINK_MAX,

_PC_MAX_CANON,

_PC_MAX_INPUT,

_PC_NAME_MAX,

_PC_PATH_MAX,

_PC_PIPE_BUF,

_PC_CHOWN_RESTRICTED,

_PC_NO_TRUNC,

_PC_VDISABLE,

_PC_SYNC_IO,

_PC_ASYNC_IO,

_PC_PRIO_IO,

_PC_SOCK_MAXBUF,

_PC_FILESIZEBITS,

_PC_REC_INCR_XFER_SIZE,

_PC_REC_MAX_XFER_SIZE,

_PC_REC_MIN_XFER_SIZE,

_PC_REC_XFER_ALIGN,

_PC_ALLOC_SIZE_MIN,

_PC_SYMLINK_MAX,

_PC_2_SYMLINKS

};



enum __cuda__SC_ARG_MAX {
_SC_ARG_MAX,

_SC_CHILD_MAX,

_SC_CLK_TCK,

_SC_NGROUPS_MAX,

_SC_OPEN_MAX,

_SC_STREAM_MAX,

_SC_TZNAME_MAX,

_SC_JOB_CONTROL,

_SC_SAVED_IDS,

_SC_REALTIME_SIGNALS,

_SC_PRIORITY_SCHEDULING,

_SC_TIMERS,

_SC_ASYNCHRONOUS_IO,

_SC_PRIORITIZED_IO,

_SC_SYNCHRONIZED_IO,

_SC_FSYNC,

_SC_MAPPED_FILES,

_SC_MEMLOCK,

_SC_MEMLOCK_RANGE,

_SC_MEMORY_PROTECTION,

_SC_MESSAGE_PASSING,

_SC_SEMAPHORES,

_SC_SHARED_MEMORY_OBJECTS,

_SC_AIO_LISTIO_MAX,

_SC_AIO_MAX,

_SC_AIO_PRIO_DELTA_MAX,

_SC_DELAYTIMER_MAX,

_SC_MQ_OPEN_MAX,

_SC_MQ_PRIO_MAX,

_SC_VERSION,

_SC_PAGESIZE,


_SC_RTSIG_MAX,

_SC_SEM_NSEMS_MAX,

_SC_SEM_VALUE_MAX,

_SC_SIGQUEUE_MAX,

_SC_TIMER_MAX,




_SC_BC_BASE_MAX,

_SC_BC_DIM_MAX,

_SC_BC_SCALE_MAX,

_SC_BC_STRING_MAX,

_SC_COLL_WEIGHTS_MAX,

_SC_EQUIV_CLASS_MAX,

_SC_EXPR_NEST_MAX,

_SC_LINE_MAX,

_SC_RE_DUP_MAX,

_SC_CHARCLASS_NAME_MAX,


_SC_2_VERSION,

_SC_2_C_BIND,

_SC_2_C_DEV,

_SC_2_FORT_DEV,

_SC_2_FORT_RUN,

_SC_2_SW_DEV,

_SC_2_LOCALEDEF,


_SC_PII,

_SC_PII_XTI,

_SC_PII_SOCKET,

_SC_PII_INTERNET,

_SC_PII_OSI,

_SC_POLL,

_SC_SELECT,

_SC_UIO_MAXIOV,

_SC_IOV_MAX = 60,

_SC_PII_INTERNET_STREAM,

_SC_PII_INTERNET_DGRAM,

_SC_PII_OSI_COTS,

_SC_PII_OSI_CLTS,

_SC_PII_OSI_M,

_SC_T_IOV_MAX,



_SC_THREADS,

_SC_THREAD_SAFE_FUNCTIONS,

_SC_GETGR_R_SIZE_MAX,

_SC_GETPW_R_SIZE_MAX,

_SC_LOGIN_NAME_MAX,

_SC_TTY_NAME_MAX,

_SC_THREAD_DESTRUCTOR_ITERATIONS,

_SC_THREAD_KEYS_MAX,

_SC_THREAD_STACK_MIN,

_SC_THREAD_THREADS_MAX,

_SC_THREAD_ATTR_STACKADDR,

_SC_THREAD_ATTR_STACKSIZE,

_SC_THREAD_PRIORITY_SCHEDULING,

_SC_THREAD_PRIO_INHERIT,

_SC_THREAD_PRIO_PROTECT,

_SC_THREAD_PROCESS_SHARED,


_SC_NPROCESSORS_CONF,

_SC_NPROCESSORS_ONLN,

_SC_PHYS_PAGES,

_SC_AVPHYS_PAGES,

_SC_ATEXIT_MAX,

_SC_PASS_MAX,


_SC_XOPEN_VERSION,

_SC_XOPEN_XCU_VERSION,

_SC_XOPEN_UNIX,

_SC_XOPEN_CRYPT,

_SC_XOPEN_ENH_I18N,

_SC_XOPEN_SHM,


_SC_2_CHAR_TERM,

_SC_2_C_VERSION,

_SC_2_UPE,


_SC_XOPEN_XPG2,

_SC_XOPEN_XPG3,

_SC_XOPEN_XPG4,


_SC_CHAR_BIT,

_SC_CHAR_MAX,

_SC_CHAR_MIN,

_SC_INT_MAX,

_SC_INT_MIN,

_SC_LONG_BIT,

_SC_WORD_BIT,

_SC_MB_LEN_MAX,

_SC_NZERO,

_SC_SSIZE_MAX,

_SC_SCHAR_MAX,

_SC_SCHAR_MIN,

_SC_SHRT_MAX,

_SC_SHRT_MIN,

_SC_UCHAR_MAX,

_SC_UINT_MAX,

_SC_ULONG_MAX,

_SC_USHRT_MAX,


_SC_NL_ARGMAX,

_SC_NL_LANGMAX,

_SC_NL_MSGMAX,

_SC_NL_NMAX,

_SC_NL_SETMAX,

_SC_NL_TEXTMAX,


_SC_XBS5_ILP32_OFF32,

_SC_XBS5_ILP32_OFFBIG,

_SC_XBS5_LP64_OFF64,

_SC_XBS5_LPBIG_OFFBIG,


_SC_XOPEN_LEGACY,

_SC_XOPEN_REALTIME,

_SC_XOPEN_REALTIME_THREADS,


_SC_ADVISORY_INFO,

_SC_BARRIERS,

_SC_BASE,

_SC_C_LANG_SUPPORT,

_SC_C_LANG_SUPPORT_R,

_SC_CLOCK_SELECTION,

_SC_CPUTIME,

_SC_THREAD_CPUTIME,

_SC_DEVICE_IO,

_SC_DEVICE_SPECIFIC,

_SC_DEVICE_SPECIFIC_R,

_SC_FD_MGMT,

_SC_FIFO,

_SC_PIPE,

_SC_FILE_ATTRIBUTES,

_SC_FILE_LOCKING,

_SC_FILE_SYSTEM,

_SC_MONOTONIC_CLOCK,

_SC_MULTI_PROCESS,

_SC_SINGLE_PROCESS,

_SC_NETWORKING,

_SC_READER_WRITER_LOCKS,

_SC_SPIN_LOCKS,

_SC_REGEXP,

_SC_REGEX_VERSION,

_SC_SHELL,

_SC_SIGNALS,

_SC_SPAWN,

_SC_SPORADIC_SERVER,

_SC_THREAD_SPORADIC_SERVER,

_SC_SYSTEM_DATABASE,

_SC_SYSTEM_DATABASE_R,

_SC_TIMEOUTS,

_SC_TYPED_MEMORY_OBJECTS,

_SC_USER_GROUPS,

_SC_USER_GROUPS_R,

_SC_2_PBS,

_SC_2_PBS_ACCOUNTING,

_SC_2_PBS_LOCATE,

_SC_2_PBS_MESSAGE,

_SC_2_PBS_TRACK,

_SC_SYMLOOP_MAX,

_SC_STREAMS,

_SC_2_PBS_CHECKPOINT,


_SC_V6_ILP32_OFF32,

_SC_V6_ILP32_OFFBIG,

_SC_V6_LP64_OFF64,

_SC_V6_LPBIG_OFFBIG,


_SC_HOST_NAME_MAX,

_SC_TRACE,

_SC_TRACE_EVENT_FILTER,

_SC_TRACE_INHERIT,

_SC_TRACE_LOG,


_SC_LEVEL1_ICACHE_SIZE,

_SC_LEVEL1_ICACHE_ASSOC,

_SC_LEVEL1_ICACHE_LINESIZE,

_SC_LEVEL1_DCACHE_SIZE,

_SC_LEVEL1_DCACHE_ASSOC,

_SC_LEVEL1_DCACHE_LINESIZE,

_SC_LEVEL2_CACHE_SIZE,

_SC_LEVEL2_CACHE_ASSOC,

_SC_LEVEL2_CACHE_LINESIZE,

_SC_LEVEL3_CACHE_SIZE,

_SC_LEVEL3_CACHE_ASSOC,

_SC_LEVEL3_CACHE_LINESIZE,

_SC_LEVEL4_CACHE_SIZE,

_SC_LEVEL4_CACHE_ASSOC,

_SC_LEVEL4_CACHE_LINESIZE,



_SC_IPV6 = 235,

_SC_RAW_SOCKETS

};



enum __cuda__CS_PATH {
_CS_PATH,


_CS_V6_WIDTH_RESTRICTED_ENVS,



_CS_GNU_LIBC_VERSION,

_CS_GNU_LIBPTHREAD_VERSION,


_CS_LFS_CFLAGS = 1000,

_CS_LFS_LDFLAGS,

_CS_LFS_LIBS,

_CS_LFS_LINTFLAGS,

_CS_LFS64_CFLAGS,

_CS_LFS64_LDFLAGS,

_CS_LFS64_LIBS,

_CS_LFS64_LINTFLAGS,


_CS_XBS5_ILP32_OFF32_CFLAGS = 1100,

_CS_XBS5_ILP32_OFF32_LDFLAGS,

_CS_XBS5_ILP32_OFF32_LIBS,

_CS_XBS5_ILP32_OFF32_LINTFLAGS,

_CS_XBS5_ILP32_OFFBIG_CFLAGS,

_CS_XBS5_ILP32_OFFBIG_LDFLAGS,

_CS_XBS5_ILP32_OFFBIG_LIBS,

_CS_XBS5_ILP32_OFFBIG_LINTFLAGS,

_CS_XBS5_LP64_OFF64_CFLAGS,

_CS_XBS5_LP64_OFF64_LDFLAGS,

_CS_XBS5_LP64_OFF64_LIBS,

_CS_XBS5_LP64_OFF64_LINTFLAGS,

_CS_XBS5_LPBIG_OFFBIG_CFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LDFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LIBS,

_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,


_CS_POSIX_V6_ILP32_OFF32_CFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LDFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LIBS,

_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LIBS,

_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,

_CS_POSIX_V6_LP64_OFF64_CFLAGS,

_CS_POSIX_V6_LP64_OFF64_LDFLAGS,

_CS_POSIX_V6_LP64_OFF64_LIBS,

_CS_POSIX_V6_LP64_OFF64_LINTFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LIBS,

_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS

};
# 569 "/usr/include/unistd.h" 3
extern "C" long pathconf(const char *, int) throw();



extern "C" long fpathconf(int, int) throw();


extern "C" long sysconf(int) throw();



extern "C" { extern inline size_t confstr(int, char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" __pid_t getpid() throw();


extern "C" __pid_t getppid() throw();




extern "C" __pid_t getpgrp() throw();
# 603 "/usr/include/unistd.h" 3
extern "C" __pid_t __getpgid(__pid_t) throw();

extern "C" __pid_t getpgid(__pid_t) throw();
# 612 "/usr/include/unistd.h" 3
extern "C" int setpgid(__pid_t, __pid_t) throw();
# 629 "/usr/include/unistd.h" 3
extern "C" int setpgrp() throw();
# 646 "/usr/include/unistd.h" 3
extern "C" __pid_t setsid() throw();



extern "C" __pid_t getsid(__pid_t) throw();



extern "C" __uid_t getuid() throw();


extern "C" __uid_t geteuid() throw();


extern "C" __gid_t getgid() throw();


extern "C" __gid_t getegid() throw();




extern "C" { extern inline int getgroups(int, __gid_t []) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }



extern "C" int group_member(__gid_t) throw();
# 679 "/usr/include/unistd.h" 3
extern "C" int setuid(__uid_t) throw();




extern "C" int setreuid(__uid_t, __uid_t) throw();




extern "C" int seteuid(__uid_t) throw();
# 696 "/usr/include/unistd.h" 3
extern "C" int setgid(__gid_t) throw();




extern "C" int setregid(__gid_t, __gid_t) throw();




extern "C" int setegid(__gid_t) throw();
# 712 "/usr/include/unistd.h" 3
extern "C" int getresuid(__uid_t *, __uid_t *, __uid_t *) throw();




extern "C" int getresgid(__gid_t *, __gid_t *, __gid_t *) throw();




extern "C" int setresuid(__uid_t, __uid_t, __uid_t) throw();




extern "C" int setresgid(__gid_t, __gid_t, __gid_t) throw();
# 735 "/usr/include/unistd.h" 3
extern "C" __pid_t fork() throw();
# 742 "/usr/include/unistd.h" 3
extern "C" __pid_t vfork() throw();
# 748 "/usr/include/unistd.h" 3
extern "C" char *ttyname(int) throw();



extern "C" { extern inline int ttyname_r(int, char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" int isatty(int) throw();
# 763 "/usr/include/unistd.h" 3
extern "C" int ttyslot() throw();




extern "C" int link(const char *, const char *) throw();
# 774 "/usr/include/unistd.h" 3
extern "C" int linkat(int, const char *, int, const char *, int) throw();
# 781 "/usr/include/unistd.h" 3
extern "C" int symlink(const char *, const char *) throw();
# 787 "/usr/include/unistd.h" 3
extern "C" { extern inline ssize_t readlink(const char *__restrict__, char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 794 "/usr/include/unistd.h" 3
extern "C" int symlinkat(const char *, int, const char *) throw();



extern "C" { extern inline ssize_t readlinkat(int, const char *__restrict__, char *__restrict__, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 804 "/usr/include/unistd.h" 3
extern "C" int unlink(const char *) throw();



extern "C" int unlinkat(int, const char *, int) throw();




extern "C" int rmdir(const char *) throw();



extern "C" __pid_t tcgetpgrp(int) throw();


extern "C" int tcsetpgrp(int, __pid_t) throw();
# 827 "/usr/include/unistd.h" 3
extern "C" char *getlogin();
# 835 "/usr/include/unistd.h" 3
extern "C" { extern inline int getlogin_r(char *, size_t) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }




extern "C" int setlogin(const char *) throw();
# 59 "/usr/include/getopt.h" 3
extern "C" { extern char *optarg; }
# 73 "/usr/include/getopt.h" 3
extern "C" { extern int optind; }




extern "C" { extern int opterr; }



extern "C" { extern int optopt; }
# 152 "/usr/include/getopt.h" 3
extern "C" int getopt(int, char *const *, const char *) throw();
# 857 "/usr/include/unistd.h" 3
extern "C" { extern inline int gethostname(char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }
# 864 "/usr/include/unistd.h" 3
extern "C" int sethostname(const char *, size_t) throw();




extern "C" int sethostid(long) throw();
# 875 "/usr/include/unistd.h" 3
extern "C" { extern inline int getdomainname(char *, size_t) throw() __attribute__((__always_inline__)) __attribute__((__gnu_inline__)); }

extern "C" int setdomainname(const char *, size_t) throw();
# 884 "/usr/include/unistd.h" 3
extern "C" int vhangup() throw();


extern "C" int revoke(const char *) throw();
# 895 "/usr/include/unistd.h" 3
extern "C" int profil(unsigned short *, size_t, size_t, unsigned) throw();
# 903 "/usr/include/unistd.h" 3
extern "C" int acct(const char *) throw();



extern "C" char *getusershell() throw();
extern "C" void endusershell() throw();
extern "C" void setusershell() throw();
# 915 "/usr/include/unistd.h" 3
extern "C" int daemon(int, int) throw();
# 922 "/usr/include/unistd.h" 3
extern "C" int chroot(const char *) throw();



extern "C" char *getpass(const char *);
# 935 "/usr/include/unistd.h" 3
extern "C" int fsync(int);
# 942 "/usr/include/unistd.h" 3
extern "C" long gethostid();


extern "C" void sync() throw();




extern "C" int getpagesize() throw() __attribute__((__const__));




extern "C" int getdtablesize() throw();




extern "C" int truncate(const char *, __off_t) throw();
# 972 "/usr/include/unistd.h" 3
extern "C" int truncate64(const char *, __off64_t) throw();
# 982 "/usr/include/unistd.h" 3
extern "C" int ftruncate(int, __off_t) throw();
# 992 "/usr/include/unistd.h" 3
extern "C" int ftruncate64(int, __off64_t) throw();
# 1002 "/usr/include/unistd.h" 3
extern "C" int brk(void *) throw();
# 1008 "/usr/include/unistd.h" 3
extern "C" void *sbrk(intptr_t) throw();
# 1023 "/usr/include/unistd.h" 3
extern "C" long syscall(long, ...) throw();
# 1046 "/usr/include/unistd.h" 3
extern "C" int lockf(int, int, __off_t);
# 1056 "/usr/include/unistd.h" 3
extern "C" int lockf64(int, int, __off64_t);
# 1077 "/usr/include/unistd.h" 3
extern "C" int fdatasync(int);
# 1085 "/usr/include/unistd.h" 3
extern "C" char *crypt(const char *, const char *) throw();




extern "C" void encrypt(char *, int) throw();
# 1097 "/usr/include/unistd.h" 3
extern "C" void swab(const void *__restrict__, void *__restrict__, ssize_t) throw();
# 1106 "/usr/include/unistd.h" 3
extern "C" char *ctermid(char *) throw();
# 24 "/usr/include/bits/unistd.h" 3
extern "C" ssize_t __read_chk(int, void *, size_t, size_t);

extern "C" ssize_t __read_alias(int, void *, size_t) __asm__("read");

extern "C" ssize_t __read_chk_warn(int, void *, size_t, size_t) __asm__("__read_chk");
# 35 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) ssize_t read(int __fd, void *__buf, size_t __nbytes)
{
if (__builtin_object_size(__buf, 0) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__nbytes))) {
return __read_chk(__fd, __buf, __nbytes, __builtin_object_size(__buf, 0)); }

if (__nbytes > __builtin_object_size(__buf, 0)) {
return __read_chk_warn(__fd, __buf, __nbytes, __builtin_object_size(__buf, 0)); }
}
return __read_alias(__fd, __buf, __nbytes);
} }


extern "C" ssize_t __pread_chk(int, void *, size_t, __off_t, size_t);

extern "C" ssize_t __pread64_chk(int, void *, size_t, __off64_t, size_t);

extern "C" ssize_t __pread_alias(int, void *, size_t, __off_t) __asm__("pread");


extern "C" ssize_t __pread64_alias(int, void *, size_t, __off64_t) __asm__("pread64");


extern "C" ssize_t __pread_chk_warn(int, void *, size_t, __off_t, size_t) __asm__("__pread_chk");




extern "C" ssize_t __pread64_chk_warn(int, void *, size_t, __off64_t, size_t) __asm__("__pread64_chk");
# 73 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) ssize_t pread(int __fd, void *__buf, size_t __nbytes, __off_t __offset)
{
if (__builtin_object_size(__buf, 0) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__nbytes))) {
return __pread_chk(__fd, __buf, __nbytes, __offset, __builtin_object_size(__buf, 0)); }

if (__nbytes > __builtin_object_size(__buf, 0)) {
return __pread_chk_warn(__fd, __buf, __nbytes, __offset, __builtin_object_size(__buf, 0)); }

}
return __pread_alias(__fd, __buf, __nbytes, __offset);
} }
# 106 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) ssize_t pread64(int __fd, void *__buf, size_t __nbytes, __off64_t __offset)
{
if (__builtin_object_size(__buf, 0) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__nbytes))) {
return __pread64_chk(__fd, __buf, __nbytes, __offset, __builtin_object_size(__buf, 0)); }

if (__nbytes > __builtin_object_size(__buf, 0)) {
return __pread64_chk_warn(__fd, __buf, __nbytes, __offset, __builtin_object_size(__buf, 0)); }

}

return __pread64_alias(__fd, __buf, __nbytes, __offset);
} }




extern "C" ssize_t __readlink_chk(const char *__restrict__, char *__restrict__, size_t, size_t) throw();



extern "C" ssize_t __readlink_alias(const char *__restrict__, char *__restrict__, size_t) throw() __asm__("readlink");



extern "C" ssize_t __readlink_chk_warn(const char *__restrict__, char *__restrict__, size_t, size_t) throw() __asm__("__readlink_chk");
# 140 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) ssize_t readlink(const char *__restrict__ __path, char *__restrict__ __buf, size_t __len) throw()

{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__len))) {
return __readlink_chk(__path, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }

if (__len > __builtin_object_size(__buf, 2 > 1)) {
return __readlink_chk_warn(__path, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }
}
return __readlink_alias(__path, __buf, __len);
} }



extern "C" ssize_t __readlinkat_chk(int, const char *__restrict__, char *__restrict__, size_t, size_t) throw();



extern "C" ssize_t __readlinkat_alias(int, const char *__restrict__, char *__restrict__, size_t) throw() __asm__("readlinkat");




extern "C" ssize_t __readlinkat_chk_warn(int, const char *__restrict__, char *__restrict__, size_t, size_t) throw() __asm__("__readlinkat_chk");
# 174 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) ssize_t readlinkat(int __fd, const char *__restrict__ __path, char *__restrict__ __buf, size_t __len) throw()

{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__len))) {
return __readlinkat_chk(__fd, __path, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }

if (__len > __builtin_object_size(__buf, 2 > 1)) {
return __readlinkat_chk_warn(__fd, __path, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }

}
return __readlinkat_alias(__fd, __path, __buf, __len);
} }


extern "C" char *__getcwd_chk(char *, size_t, size_t) throw();

extern "C" char *__getcwd_alias(char *, size_t) throw() __asm__("getcwd");

extern "C" char *__getcwd_chk_warn(char *, size_t, size_t) throw() __asm__("__getcwd_chk");
# 201 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *getcwd(char *__buf, size_t __size) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__size))) {
return __getcwd_chk(__buf, __size, __builtin_object_size(__buf, 2 > 1)); }

if (__size > __builtin_object_size(__buf, 2 > 1)) {
return __getcwd_chk_warn(__buf, __size, __builtin_object_size(__buf, 2 > 1)); }
}
return __getcwd_alias(__buf, __size);
} }


extern "C" char *__getwd_chk(char *, size_t) throw();

extern "C" char *__getwd_warn(char *) throw() __asm__("getwd");




extern "C" { inline __attribute__((__deprecated__)) __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) char *getwd(char *__buf) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1))) {
return __getwd_chk(__buf, __builtin_object_size(__buf, 2 > 1)); }
return __getwd_warn(__buf);
} }


extern "C" size_t __confstr_chk(int, char *, size_t, size_t) throw();

extern "C" size_t __confstr_alias(int, char *, size_t) throw() __asm__("confstr");

extern "C" size_t __confstr_chk_warn(int, char *, size_t, size_t) throw() __asm__("__confstr_chk");
# 241 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) size_t confstr(int __name, char *__buf, size_t __len) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__len))) {
return __confstr_chk(__name, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }

if (__builtin_object_size(__buf, 2 > 1) < __len) {
return __confstr_chk_warn(__name, __buf, __len, __builtin_object_size(__buf, 2 > 1)); }
}
return __confstr_alias(__name, __buf, __len);
} }


extern "C" int __getgroups_chk(int, __gid_t [], size_t) throw();

extern "C" int __getgroups_alias(int, __gid_t []) throw() __asm__("getgroups");

extern "C" int __getgroups_chk_warn(int, __gid_t [], size_t) throw() __asm__("__getgroups_chk");
# 266 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int getgroups(int __size, __gid_t __list[]) throw()
{
if (__builtin_object_size(__list, 2 > 1) != ((size_t)(-1)))
{
if ((!(__builtin_constant_p(__size))) || (__size < 0)) {
return __getgroups_chk(__size, __list, __builtin_object_size(__list, 2 > 1)); }

if ((__size * sizeof(__gid_t)) > __builtin_object_size(__list, 2 > 1)) {
return __getgroups_chk_warn(__size, __list, __builtin_object_size(__list, 2 > 1)); }
}
return __getgroups_alias(__size, __list);
} }


extern "C" int __ttyname_r_chk(int, char *, size_t, size_t) throw();

extern "C" int __ttyname_r_alias(int, char *, size_t) throw() __asm__("ttyname_r");


extern "C" int __ttyname_r_chk_warn(int, char *, size_t, size_t) throw() __asm__("__ttyname_r_chk");
# 292 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int ttyname_r(int __fd, char *__buf, size_t __buflen) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__buflen))) {
return __ttyname_r_chk(__fd, __buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }

if (__buflen > __builtin_object_size(__buf, 2 > 1)) {
return __ttyname_r_chk_warn(__fd, __buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
}
return __ttyname_r_alias(__fd, __buf, __buflen);
} }



extern "C" int __getlogin_r_chk(char *, size_t, size_t);

extern "C" int __getlogin_r_alias(char *, size_t) __asm__("getlogin_r");

extern "C" int __getlogin_r_chk_warn(char *, size_t, size_t) __asm__("__getlogin_r_chk");
# 318 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int getlogin_r(char *__buf, size_t __buflen)
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__buflen))) {
return __getlogin_r_chk(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }

if (__buflen > __builtin_object_size(__buf, 2 > 1)) {
return __getlogin_r_chk_warn(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
}
return __getlogin_r_alias(__buf, __buflen);
} }




extern "C" int __gethostname_chk(char *, size_t, size_t) throw();

extern "C" int __gethostname_alias(char *, size_t) throw() __asm__("gethostname");

extern "C" int __gethostname_chk_warn(char *, size_t, size_t) throw() __asm__("__gethostname_chk");
# 345 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int gethostname(char *__buf, size_t __buflen) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__buflen))) {
return __gethostname_chk(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }

if (__buflen > __builtin_object_size(__buf, 2 > 1)) {
return __gethostname_chk_warn(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
}
return __gethostname_alias(__buf, __buflen);
} }




extern "C" int __getdomainname_chk(char *, size_t, size_t) throw();

extern "C" int __getdomainname_alias(char *, size_t) throw() __asm__("getdomainname");


extern "C" int __getdomainname_chk_warn(char *, size_t, size_t) throw() __asm__("__getdomainname_chk");
# 374 "/usr/include/bits/unistd.h" 3
extern "C" { inline __attribute__((__always_inline__)) __attribute__((__gnu_inline__)) int getdomainname(char *__buf, size_t __buflen) throw()
{
if (__builtin_object_size(__buf, 2 > 1) != ((size_t)(-1)))
{
if (!(__builtin_constant_p(__buflen))) {
return __getdomainname_chk(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }

if (__buflen > __builtin_object_size(__buf, 2 > 1)) {
return __getdomainname_chk_warn(__buf, __buflen, __builtin_object_size(__buf, 2 > 1)); }
}
return __getdomainname_alias(__buf, __buflen);
} }
# 12 "../../parboil/parboil.h"
extern "C" { struct pb_Parameters {
char *outFile;


char **inpFiles;



int synchronizeGpu;
# 27 "../../parboil/parboil.h"
}; }
# 39 "../../parboil/parboil.h"
extern "C" pb_Parameters *pb_ReadParameters(int *, char **);




extern "C" void pb_FreeParameters(pb_Parameters *);




extern "C" int pb_Parameters_CountInputs(pb_Parameters *);



extern "C" { typedef unsigned long long pb_Timestamp; }




enum pb_TimerState {
pb_Timer_STOPPED,
pb_Timer_RUNNING
};

extern "C" { struct pb_Timer {
pb_TimerState state;
pb_Timestamp elapsed;
pb_Timestamp init;


}; }
# 76 "../../parboil/parboil.h"
extern "C" void pb_ResetTimer(pb_Timer *);
# 84 "../../parboil/parboil.h"
extern "C" void pb_StartTimer(pb_Timer *);
# 91 "../../parboil/parboil.h"
extern "C" void pb_StopTimer(pb_Timer *);



extern "C" double pb_GetElapsedTime(pb_Timer *);


enum pb_TimerID {
pb_TimerID_NONE,
pb_TimerID_IO,
pb_TimerID_GPU,
pb_TimerID_COPY,

pb_TimerID_COMPUTE,


pb_TimerID_LAST
};


extern "C" { struct pb_TimerSet {
pb_TimerID current;
pb_Timer timers[pb_TimerID_LAST];
}; }



extern "C" void pb_InitializeTimerSet(pb_TimerSet *);
# 124 "../../parboil/parboil.h"
extern "C" void pb_SwitchToTimer(pb_TimerSet *, pb_TimerID);



extern "C" void pb_PrintTimerSet(pb_TimerSet *);
# 71 "/usr/include/assert.h" 3
extern "C" void __assert_fail(const char *, const char *, unsigned, const char *) throw() __attribute__((__noreturn__));




extern "C" void __assert_perror_fail(int, const char *, unsigned, const char *) throw() __attribute__((__noreturn__));
# 84 "/usr/include/assert.h" 3
extern "C" void __assert(const char *, const char *, int) throw() __attribute__((__noreturn__));
# 18 "randomc.h"
typedef signed int int32;
typedef unsigned uint32;
# 36 "randomc.h"
# 13 "petri.h"
extern void PetrinetKernel__entry(int *, int, int, int);
# 201 "petri_kernel.cu"
void PetrinetKernel__entry(int *g_s, float *g_v, int *g_m, int n, int s, int seed);
# 51 "main.cu"
static int N; static int s; static int t; static int N2; static int NSQUARE2;
uint32 host_mt[624];


extern void *AllocateDeviceMemory(int);
extern void CopyFromDeviceMemory(void *, void *, int);
extern void CopyFromHostMemory(void *, void *, int);
extern void FreeDeviceMemory(void *);
extern void PetrinetOnDevice(pb_TimerSet *, pb_Parameters *);

extern void compute_statistics();

float results[4];
float *h_vars;
int *h_maxs;

int main(int argc, char **argv)
{
auto pb_TimerSet timers;
auto pb_Parameters *params;

pb_InitializeTimerSet(&timers);
params = pb_ReadParameters(&argc, argv);

if (argc < 4)
{
printf("Usage: petri n s t\nn: the place-transition grid is 2nX2n\ns: the maximum steps in a trajectory\nt: number of trajectories\n");



return -1;
}

pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
N = atoi(argv[1]);
if (N < 1) {
return -1; }
s = atoi(argv[2]);
if (s < 1) {
return -1; }

t = atoi(argv[3]);
if (t < 1) {
return -1; }

N2 = N + N;
NSQUARE2 = N * N2;

h_vars = (float *)malloc(t * sizeof(float));
h_maxs = (int *)malloc(t * sizeof(int));


PetrinetOnDevice(&timers, params);
pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

compute_statistics();

free(h_vars);
free(h_maxs);

pb_SwitchToTimer(&timers, pb_TimerID_IO);

printf("petri N=%d s=%d t=%d\n", N, s, t);
printf("mean_vars: %f    var_vars: %f\n", (results)[0], (results)[1]);
printf("mean_maxs: %f    var_maxs: %f\n", (results)[2], (results)[3]);

if (params->outFile)
{
auto FILE *out;
out = fopen(params->outFile, "w");
fprintf(out, "%f %f %f %f\n", (results)[0], (results)[1], (results)[2], (results)[3]);

fclose(out);
}


pb_SwitchToTimer(&timers, pb_TimerID_NONE);
pb_PrintTimerSet(&timers);
pb_FreeParameters(params);

return 0;
}

void compute_statistics()
{
auto float sum = (0);
auto float sum_vars = (0);
auto float sum_max = (0);
auto float sum_max_vars = (0);
auto int i;
for (i = 0; i < t; i++)
{
sum += h_vars[i];
sum_vars += h_vars[i] * h_vars[i];
sum_max += h_maxs[i];
sum_max_vars += h_maxs[i] * h_maxs[i];
}
((results)[0]) = sum / t;
((results)[1]) = sum_vars / t - (results)[0] * (results)[0];
((results)[2]) = sum_max / t;
((results)[3]) = sum_max_vars / t - (results)[2] * (results)[2];
}

void PetrinetOnDevice(pb_TimerSet *timers, pb_Parameters *
params)
{

auto int i;
auto int unit_size = ((NSQUARE2 * (sizeof(int) + sizeof(char)) + sizeof(float)) + sizeof(int));

auto int block_num = (750000000 / unit_size);
auto int *p_hmaxs;
auto float *p_hvars;
auto int *g_places;
auto float *g_vars;
auto int *g_maxs;

pb_SwitchToTimer(timers, pb_TimerID_COPY);
g_places = (int *)AllocateDeviceMemory(((unit_size - sizeof(float)) - sizeof(int)) * block_num);

{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
g_vars = (float *)AllocateDeviceMemory(block_num * sizeof(float));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
g_maxs = (int *)AllocateDeviceMemory(block_num * sizeof(int));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }


auto dim3 grid(block_num);
auto dim3 threads(256);

p_hmaxs = h_maxs;
p_hvars = h_vars;


for (i = 0; i < (t - block_num); i += block_num)
{
if (params->synchronizeGpu) { cudaThreadSynchronize(); }
pb_SwitchToTimer(timers, pb_TimerID_GPU);
cudaConfigureCall(grid, threads) ? ((void)0) : (PetrinetKernel__entry)(g_places, g_vars, g_maxs, N, s, 5489 * (i + 1));

{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }

if (params->synchronizeGpu) { cudaThreadSynchronize(); }
pb_SwitchToTimer(timers, pb_TimerID_COPY);
CopyFromDeviceMemory(p_hmaxs, g_maxs, block_num * sizeof(int));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
CopyFromDeviceMemory(p_hvars, g_vars, block_num * sizeof(float));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
if (params->synchronizeGpu) { cudaThreadSynchronize(); }

pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
p_hmaxs += block_num;
p_hvars += block_num;
}

auto dim3 grid1(t - i);
pb_SwitchToTimer(timers, pb_TimerID_GPU);
cudaConfigureCall(grid1, threads) ? ((void)0) : (PetrinetKernel__entry)(g_places, g_vars, g_maxs, N, s, time(__null));

{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }


if (params->synchronizeGpu) { cudaThreadSynchronize(); }
pb_SwitchToTimer(timers, pb_TimerID_COPY);
CopyFromDeviceMemory(p_hmaxs, g_maxs, (t - i) * sizeof(int));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
CopyFromDeviceMemory(p_hvars, g_vars, (t - i) * sizeof(float));
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }


FreeDeviceMemory(g_places);
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
FreeDeviceMemory(g_vars);
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
FreeDeviceMemory(g_maxs);
{ auto cudaError_t err = cudaGetLastError(); if (err) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); } }
}


void *AllocateDeviceMemory(int size)
{
auto int *mem;
cudaMalloc((void **)(&mem), size);
return mem;
}


void CopyFromDeviceMemory(void *h_p, void *d_p, int size)
{
cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
}


void CopyFromHostMemory(void *d_p, void *h_p, int size)
{
cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
}


void FreeDeviceMemory(void *mem)
{
if (mem != (__null)) {
cudaFree(mem); }
}

# 1 "/tmp/tmpxft_00006e97_00000000-1_main.cudafe1.stub.c" 1

extern "C" {

# 1 "/tmp/tmpxft_00006e97_00000000-3_main.fatbin.c" 1
# 1 "/usr/local/cuda/bin/../include/__cudaFatFormat.h" 1
# 83 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
extern "C" {
# 97 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;
# 113 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* ptx;
} __cudaFatPtxEntry;
# 125 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* debug;
} __cudaFatDebugEntry;


typedef enum {
      __cudaFatDontSearchFlag = (1 << 0),
      __cudaFatDontCacheFlag = (1 << 1),
      __cudaFatSassDebugFlag = (1 << 2)
} __cudaFatCudaBinaryFlag;
# 144 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* name;
} __cudaFatSymbol;
# 158 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
} __cudaFatCudaBinary;
# 194 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
    typedef enum {
        __cudaFatAvoidPTX,
        __cudaFatPreferBestCode
    } __cudaFatCompilationPolicy;
# 217 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
void fatGetCubinForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *cubin, char* *dbgInfoFile );
# 228 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
void fatFreeCubin( char* cubin, char* dbgInfoFile );


}
# 2 "/tmp/tmpxft_00006e97_00000000-3_main.fatbin.c" 2




extern "C" {


static const unsigned long long __deviceText_$sm_13$[] = {
0x6365746968637261ull,0x6d737b2065727574ull,0x6962610a7d33315full,0x206e6f6973726576ull,
0x6f6d0a7d317b2020ull,0x202020656d616e64ull,0x696275637b202020ull,0x2065646f630a7d6eull,
0x20656d616e090a7bull,0x655034315a5f203dull,0x654b74656e697274ull,0x665069506c656e72ull,
0x6c090a6969695f53ull,0x0a30203d206d656dull,0x203d206d656d7309ull,0x6572090a38343532ull,
0x0a3832203d202067ull,0x203d202072616209ull,0x74736e6f63090a31ull,0x65730909090a7b20ull,
0x203d20656d616e67ull,0x09090a74736e6f63ull,0x206d756e67657309ull,0x0909090a31203d20ull,
0x202074657366666full,0x620909090a30203dull,0x3d20202073657479ull,0x656d09090a343620ull,
0x300909090a7b206dull,0x3732303030303078ull,0x3030303078302030ull,0x3078302032653030ull,
0x2030303130303030ull,0x3030303030307830ull,0x300909090a206636ull,0x3730303030303078ull,
0x3030303078302030ull,0x3878302065363230ull,0x2030303030303030ull,0x6666666666377830ull,
0x300909090a206666ull,0x6430623830393978ull,0x3030303078302066ull,0x3978302031303030ull,
0x2030383635633264ull,0x3030366366657830ull,0x300909090a203030ull,0x6666666666666678ull,
0x6666666678302063ull,0x3078302066666666ull,0x2066663030303030ull,0x3030303866627830ull,
0x0a7d09090a203030ull,0x636e6962090a7d09ull,0x09090a7b2065646full,0x3431303030317830ull,
0x3034347830203930ull,0x7830203038376330ull,0x6430383035303034ull,0x3030303030783020ull,
0x3009090a20303837ull,0x3061303430303678ull,0x3030303078302064ull,0x3378302030383763ull,
0x2064303630303130ull,0x3730303134637830ull,0x783009090a203038ull,0x3930383034303036ull,
0x6330303030783020ull,0x3033783020303837ull,0x3020643134303130ull,0x3837303031346378ull,
0x33783009090a2030ull,0x2031326530613030ull,0x3730303163657830ull,0x3730337830203038ull,
0x7830206466313163ull,0x3863376330306336ull,0x3031783009090a20ull,0x3020333030633130ull,
0x3832303030303078ull,0x3230303378302030ull,0x6578302039326530ull,0x2030383730303163ull,
0x303032783009090aull,0x7830203930653030ull,0x3038373832303430ull,0x3435303034783020ull,
0x3030783020643063ull,0x0a20303837303032ull,0x3031303378300909ull,0x6378302064303630ull,
0x2030383730303134ull,0x6334343030367830ull,0x3230307830203930ull,0x090a203038376330ull,
0x3032303033783009ull,0x3463783020643034ull,0x3020303837303031ull,0x3038663030303178ull,
0x3330363078302039ull,0x09090a2030383763ull,0x3030303030617830ull,0x3034307830206432ull,
0x7830203038373030ull,0x3130386330303032ull,0x6330323430783020ull,0x3009090a20303837ull,
0x3034303830303378ull,0x3031346378302064ull,0x3278302030383730ull,0x2064303631303030ull,
0x3763303034307830ull,0x783009090a203038ull,0x6430363032303033ull,0x3030313463783020ull,
0x3032783020303837ull,0x3020393034383130ull,0x3030303030303078ull,0x31783009090a2033ull,
0x2031313038313030ull,0x3031303130307830ull,0x3030327830203331ull,0x7830206430303030ull,
0x3038376330303430ull,0x3033783009090a20ull,0x3020646631313230ull,0x6337343130633678ull,
0x6530306478302038ull,0x6178302031313630ull,0x2030383730306330ull,0x303031783009090aull,
0x7830203330303231ull,0x3038323030303030ull,0x6531303031783020ull,0x3030783020333030ull,
0x0a20303837303030ull,0x3230303378300909ull,0x6578302039326530ull,0x2030383730303163ull,
0x3030303030617830ull,0x3034307830206432ull,0x090a203038373030ull,0x3138303033783009ull,
0x3463783020393030ull,0x3020303837303031ull,0x3034303030303378ull,0x3230343078302031ull,
0x09090a2030383738ull,0x3731303030337830ull,0x3034367830206466ull,0x7830203863373831ull,
0x3330306432303061ull,0x3030303030783020ull,0x3009090a20303030ull,0x3030643230303178ull,
0x3030303078302033ull,0x3278302030383230ull,0x2031306530303030ull,0x3738323034307830ull,
0x783009090a203038ull,0x6430633431303034ull,0x3030323030783020ull,0x3033783020303837ull,
0x3020643036303031ull,0x3837303031346378ull,0x36783009090a2030ull,0x2031306334303030ull,
0x3763303230307830ull,0x3030327830203038ull,0x7830203930343030ull,0x3038376332303430ull,
0x3033783009090a20ull,0x3020313030303230ull,0x3837303031346378ull,0x3230303378302030ull,
0x6378302039303430ull,0x2030383730303134ull,0x303132783009090aull,0x7830203030386530ull,
0x3030303832303032ull,0x3831303031783020ull,0x3030783020393030ull,0x0a20333130313031ull,
0x6530306478300909ull,0x6178302039303030ull,0x2030383730306330ull,0x3631633730337830ull,
0x3034367830203130ull,0x090a203263373830ull,0x3030303061783009ull,0x6330783020646631ull,
0x3020386537343130ull,0x3430303030303078ull,0x3030303278302035ull,0x09090a2030383732ull,
0x3037343030617830ull,0x3030307830203330ull,0x7830203030303030ull,0x3330303734303031ull,
0x3030303030783020ull,0x3009090a20303031ull,0x3063343030306178ull,0x3032343078302039ull,
0x3178302030383730ull,0x2031303038383330ull,0x3030303030307830ull,0x783009090a203330ull,
0x3930386430303032ull,0x3830323430783020ull,0x3030783020303837ull,0x3020353030303030ull,
0x3837303030306378ull,0x30783009090a2030ull,0x2031306131303030ull,0x3738303234657830ull,
0x3030317830203038ull,0x7830206430303831ull,0x3330303030303030ull,0x3430783009090a20ull,
0x3020313030303030ull,0x3837303030303478ull,0x6333303278302030ull,0x3078302031303038ull,
0x2066666666666666ull,0x303030783009090aull,0x7830203930303030ull,0x3038373030303063ull,
0x6365313833783020ull,0x3465783020313030ull,0x0a20303837303033ull,0x3532303178300909ull,
0x3078302039303038ull,0x2037393837306336ull,0x3063303038647830ull,0x3234307830203130ull,
0x090a203038373830ull,0x3035303034783009ull,0x3030783020313130ull,0x3020303837303030ull,
0x3132303430303678ull,0x3130303078302031ull,0x09090a2030383730ull,0x3830303130337830ull,
0x3134637830203131ull,0x7830203038373030ull,0x3130303034303036ull,0x3031303030783020ull,
0x3009090a20303837ull,0x3036303030303278ull,0x3030343078302031ull,0x3278302030383730ull,
0x2064303638313030ull,0x3030303030307830ull,0x783009090a203330ull,0x3130323030303630ull,
0x3030323465783020ull,0x3033783020303837ull,0x3020646637303038ull,0x6337343134633678ull,
0x31783009090a2038ull,0x2033303038333030ull,0x3230303030307830ull,0x3030667830203038ull,
0x7830203130303030ull,0x3230303030303065ull,0x3638783009090a20ull,0x3020333065666631ull,
0x3030303030303078ull,0x6337303378302030ull,0x3678302064663764ull,0x2038633763303263ull,
0x323031783009090aull,0x7830203330303938ull,0x3038323030303030ull,0x6630303131783020ull,
0x3032783020303034ull,0x0a20383065386130ull,0x3030303278300909ull,0x3078302031313464ull,
0x2030383730303234ull,0x3031313030347830ull,0x3031347830203831ull,0x090a206330633235ull,
0x3130303036783009ull,0x3030783020353232ull,0x3020303837383130ull,0x3136303031303378ull,
0x3031346378302039ull,0x09090a2030383730ull,0x3639643030327830ull,0x3030307830206430ull,
0x7830203330303030ull,0x3532323130313033ull,0x3030313463783020ull,0x3009090a20303837ull,
0x3063343430303678ull,0x3132303078302039ull,0x3278302030383738ull,0x2031333639363030ull,
0x3030303030307830ull,0x783009090a206631ull,0x3530363032303030ull,0x3030303063783020ull,
0x3036783020303837ull,0x3020393130313030ull,0x3837343230303078ull,0x33783009090a2030ull,
0x2031303430323030ull,0x3730303134637830ull,0x3830337830203038ull,0x7830206466393135ull,
0x3864376330346336ull,0x3033783009090a20ull,0x3020646637313138ull,0x6337633034343678ull,
0x3030303178302038ull,0x3078302035353866ull,0x2030383763333034ull,0x303032783009090aull,
0x7830203133386330ull,0x3038373030323430ull,0x3132383033783020ull,0x3436783020646637ull,
0x0a20386537343034ull,0x3030303078300909ull,0x3278302039343030ull,0x2030383732303030ull,
0x3731333830337830ull,0x3434367830206466ull,0x090a203865376330ull,0x3030303030783009ull,
0x3032783020643430ull,0x3020303837323030ull,0x6637313438303378ull,0x3034343678302064ull,
0x09090a2038653734ull,0x3030303030307830ull,0x3030327830203135ull,0x7830203038373230ull,
0x3330306436303061ull,0x3030303030783020ull,0x3009090a20303030ull,0x3030643630303178ull,
0x3030303078302033ull,0x6478302030303130ull,0x2031303063363834ull,0x3730303634307830ull,
0x783009090a203038ull,0x3930326337383464ull,0x3030363430783020ull,0x3064783020303837ull,
0x3020313030303230ull,0x3837343030343078ull,0x64783009090a2030ull,0x2039303836633034ull,
0x3730303030327830ull,0x3030337830203038ull,0x7830203930303031ull,0x3038373030313465ull,
0x3864783009090a20ull,0x3020393030633230ull,0x3837383032343078ull,0x3938306478302030ull,
0x3078302031303030ull,0x2030383730303434ull,0x383064783009090aull,0x7830203533343038ull,
0x3038373830343430ull,0x3030303061783020ull,0x6332783020646631ull,0x0a20386537343130ull,
0x3030303178300909ull,0x3078302035333430ull,0x2030303365333034ull,0x3030303030667830ull,
0x3030657830203130ull,0x090a203230303030ull,0x6666313638783009ull,0x3030783020333065ull,
0x3020303030303030ull,0x3030303030343078ull,0x3332346578302031ull,0x09090a2030383634ull,
0x6566663136387830ull,0x3030307830203330ull,0x7830203030303030ull,0x3330306537303061ull,
0x3030303030783020ull,0x3009090a20303030ull,0x3030653730303178ull,0x3030303078302033ull,
0x3078302030303130ull,0x2039303631323030ull,0x3730303030637830ull,0x783009090a203038ull,
0x3930303837303864ull,0x3030303032783020ull,0x3864783020303837ull,0x3020313030633638ull,
0x3837303036343078ull,0x64783009090a2030ull,0x2039303263373838ull,0x3730303634307830ull,
0x3030647830203038ull,0x7830203130303032ull,0x3038373430303430ull,0x3033783009090a20ull,
0x3020393030303130ull,0x3837303031346578ull,0x3230346478302030ull,0x3078302039303063ull,
0x2030383738303234ull,0x383064783009090aull,0x7830203130303039ull,0x3038373030343430ull,
0x3038383064783020ull,0x3430783020353334ull,0x0a20303837383034ull,0x3030306178300909ull,
0x3278302064663130ull,0x2038653734313063ull,0x3430303030317830ull,0x3034307830203533ull,
0x090a203030336533ull,0x3030303066783009ull,0x3065783020313030ull,0x3020323030303030ull,
0x3065666631363878ull,0x3030303078302033ull,0x09090a2030303030ull,0x3631323030307830ull,
0x3030637830203930ull,0x7830203038363030ull,0x3130306531303830ull,0x3433323465783020ull,
0x3009090a20303836ull,0x3065666631363878ull,0x3030303078302033ull,0x6178302030303030ull,
0x2033303031393030ull,0x3030303030307830ull,0x783009090a203030ull,0x3330303139303031ull,
0x3130303030783020ull,0x3030783020303031ull,0x3020393036313230ull,0x3837303030306378ull,
0x64783009090a2030ull,0x2039303839653038ull,0x3730303030327830ull,0x3838647830203038ull,
0x7830203130306336ull,0x3038373030363430ull,0x3864783009090a20ull,0x3020393032633738ull,
0x3837303036343078ull,0x3230306478302030ull,0x3078302031303030ull,0x2030383734303034ull,
0x303464783009090aull,0x7830203930383137ull,0x3038373030303032ull,0x3031303033783020ull,
0x3465783020393030ull,0x0a20303837303031ull,0x3230386478300909ull,0x3078302039303063ull,
0x2030383738303234ull,0x3030393830647830ull,0x3434307830203130ull,0x090a203038373030ull,
0x3038383064783009ull,0x3430783020353334ull,0x3020303837383034ull,0x6631303030306178ull,
0x3130633278302064ull,0x09090a2038653734ull,0x3430303030317830ull,0x3034307830203533ull,
0x7830203030336533ull,0x3130303030303066ull,0x3030303065783020ull,0x3009090a20323030ull,
0x3065666631363878ull,0x3030303078302033ull,0x3078302030303030ull,0x2039303631323030ull,
0x3631303030637830ull,0x783009090a203038ull,0x3130366133303830ull,0x3533323465783020ull,
0x3638783020303836ull,0x3020333065666631ull,0x3030303030303078ull,0x61783009090a2030ull,
0x2033303035613030ull,0x3030303030307830ull,0x3030307830203030ull,0x7830203130323230ull,
0x3065373030303061ull,0x3031783009090a20ull,0x3020333030356130ull,0x3031323030303078ull,
0x3331306478302030ull,0x3278302039303065ull,0x2030383730303030ull,0x383064783009090aull,
0x7830203130616437ull,0x3038373030363430ull,0x6336383864783020ull,0x3430783020393030ull,
0x0a20303837303036ull,0x3230306478300909ull,0x3078302035333030ull,0x2030383734303034ull,
0x3863633030647830ull,0x3030327830203930ull,0x090a203038373030ull,0x3131303033783009ull,
0x3465783020313061ull,0x3020303837303031ull,0x3030633030386478ull,0x3032343078302064ull,
0x09090a2030383738ull,0x6131393830647830ull,0x3434307830203930ull,0x7830203038373030ull,
0x3130363038383064ull,0x3830343430783020ull,0x3009090a20303837ull,0x6635303030306178ull,
0x3130633278302064ull,0x3178302038653734ull,0x2031303630303030ull,0x3365333034307830ull,
0x783009090a203030ull,0x3130386634303030ull,0x3030323465783020ull,0x3066783020303837ull,
0x3020313030303030ull,0x3030303030306578ull,0x38783009090a2032ull,0x2033306566663136ull,
0x3030303030307830ull,0x3035317830203030ull,0x7830203030306530ull,0x3830633832303031ull,
0x3032783009090a20ull,0x3020333030363133ull,0x3837303030303078ull,0x6230303378302030ull,
0x6578302039306131ull,0x2030383730303134ull,0x303064783009090aull,0x7830203930343064ull,
0x3038373830303430ull,0x3037303033783020ull,0x3463783020643034ull,0x0a20303837303031ull,
0x6138306478300909ull,0x3078302064303630ull,0x2030383730303434ull,0x3430333030647830ull,
0x3034307830203930ull,0x090a203038373830ull,0x3066303033783009ull,0x3463783020643034ull,
0x3020303837303031ull,0x3036306238306478ull,0x3034343078302064ull,0x09090a2030383730ull,
0x3430333030647830ull,0x3034307830203930ull,0x7830203038373830ull,0x6430343032313033ull,
0x3030313465783020ull,0x3009090a20303837ull,0x3334303330306478ull,0x3030343078302035ull,
0x6178302030383738ull,0x2033303036653030ull,0x3030303030307830ull,0x783009090a203030ull,
0x3935303030303031ull,0x6333303430783020ull,0x3030783020303837ull,0x3020313034323030ull,
0x6537303030306178ull,0x31783009090a2030ull,0x2033303035653030ull,0x3132303030307830ull,
0x3031317830203030ull,0x7830203830346630ull,0x3030636132303031ull,0x3032783009090a20ull,
0x3020333030353333ull,0x3837303030303078ull,0x3030303178302030ull,0x3178302038303038ull,
0x2034323466303031ull,0x303031783009090aull,0x7830203130633230ull,0x3038376333303630ull,
0x3635333032783020ull,0x3030783020333030ull,0x0a20303837303030ull,0x6633303278300909ull,
0x3078302035323838ull,0x2066666666666666ull,0x3438663330327830ull,0x6666307830203530ull,
0x090a206666666666ull,0x3063373033783009ull,0x6336783020646635ull,0x3020386537383030ull,
0x3032313030303178ull,0x3330343078302035ull,0x09090a2030383265ull,0x3038303030317830ull,
0x3030317830203431ull,0x7830203030323832ull,0x6466353039383064ull,0x3030343430783020ull,
0x3009090a20386537ull,0x3334643030313278ull,0x3336343078302039ull,0x3278302030383636ull,
0x2064306130303031ull,0x3636323434307830ull,0x783009090a203038ull,0x3933613065303033ull,
0x3631306336783020ull,0x3064783020303836ull,0x3020393336306530ull,0x3836323030343078ull,
0x31783009090a2030ull,0x2039336130303030ull,0x3165333034307830ull,0x3030317830203030ull,
0x7830206430343130ull,0x3038376330303434ull,0x3034783009090a20ull,0x3020633330303730ull,
0x3465303430303478ull,0x3630303678302030ull,0x3078302064333230ull,0x2030383763333030ull,
0x303036783009090aull,0x7830203134633035ull,0x3038373034303030ull,0x3130313033783020ull,
0x3463783020643365ull,0x0a20303837303031ull,0x3031303378300909ull,0x6378302031343032ull,
0x2030383730303134ull,0x3030363030367830ull,0x3030307830203130ull,0x090a203038376333ull,
0x3034303036783009ull,0x3030783020643063ull,0x3020303837303430ull,0x3063393030303278ull,
0x3530303278302030ull,0x09090a2063303638ull,0x3231323030337830ull,0x3063367830203531ull,
0x7830203038373431ull,0x3532343831303032ull,0x3030303030783020ull,0x3009090a20333030ull,
0x3030303030303278ull,0x3330343078302031ull,0x6478302030383730ull,0x2031303030653030ull,
0x3730303230387830ull,0x783009090a203038ull,0x3930383130303032ull,0x6330303430783020ull,
0x3064783020303837ull,0x3020393034306530ull,0x3837303032303878ull,0x64783009090a2030ull,
0x2035313231353030ull,0x3730303034307830ull,0x3030617830203038ull,0x7830203134383030ull,
0x3038373831306330ull,0x3061783009090a20ull,0x3020643330303030ull,0x3837383130633078ull,
0x6337303378302030ull,0x3678302064663132ull,0x2038653730313063ull,0x373033783009090aull,
0x7830206466663163ull,0x3865323231306336ull,0x3365303031783020ull,0x3030783020333030ull,
0x0a20303031323030ull,0x6338306478300909ull,0x3078302031306330ull,0x2030383730303434ull,
0x3630323030337830ull,0x3134637830203930ull,0x090a203038373030ull,0x3930303032783009ull,
0x3032783020303038ull,0x3020303030383230ull,0x3061613730303278ull,0x3030303078302039ull,
0x09090a2033303030ull,0x3030653030647830ull,0x6330617830203930ull,0x7830203038373030ull,
0x3130303831303031ull,0x3030303030783020ull,0x3009090a20333030ull,0x3030366530303178ull,
0x3030303078302033ull,0x3178302030383730ull,0x2031303866303030ull,0x3763333036307830ull,
0x783009090a203038ull,0x3330303665303031ull,0x3030303030783020ull,0x3031783020303837ull,
0x3020313038663030ull,0x3837633330343078ull,0x66783009090a2030ull,0x2031303030303030ull,
0x3030303030657830ull,0x3136387830203230ull,0x7830203330656666ull,0x3030303030303030ull,
0x3061783009090a20ull,0x3020333030333131ull,0x3030303030303078ull,0x6337303378302030ull,
0x3678302031303030ull,0x2030653734313063ull,0x303061783009090aull,0x7830206466313030ull,
0x3866373431306330ull,0x3331313031783020ull,0x3030783020333030ull,0x0a20303031323030ull,
0x3030313178300909ull,0x3178302038303466ull,0x2030306361323030ull,0x3035333330327830ull,
0x3030307830203330ull,0x090a203038373030ull,0x6137303032783009ull,0x3030783020393061ull,
0x3020333030303030ull,0x3034303938306478ull,0x3034343078302064ull,0x09090a2030383730ull,
0x3030393830647830ull,0x3434307830203130ull,0x7830203038373030ull,0x3330303231313061ull,
0x3030303030783020ull,0x3009090a20303030ull,0x6637303030303378ull,0x3030633678302064ull,
0x3178302038653738ull,0x2033303066303130ull,0x3232303030307830ull,0x783009090a203038ull,
0x3130343130303031ull,0x6330303434783020ull,0x3034783020303837ull,0x3020643034303130ull,
0x3837303030303078ull,0x36783009090a2030ull,0x2064303630303030ull,0x3763303030307830ull,
0x3130337830203038ull,0x7830206430363030ull,0x3038373030313463ull,0x3036783009090a20ull,
0x3020643034303030ull,0x3837633030303078ull,0x6338306478302030ull,0x3078302031306330ull,
0x2030383730303434ull,0x303032783009090aull,0x7830203432363865ull,0x6330383930303032ull,
0x3132303033783020ull,0x3463783020313032ull,0x0a20303837303031ull,0x3030303278300909ull,
0x3078302031303630ull,0x2030383730303034ull,0x3030653030647830ull,0x6330387830203130ull,
0x090a203038373030ull,0x3032303033783009ull,0x6336783020646631ull,0x3020386537383030ull,
0x3030313131303178ull,0x3030303078302033ull,0x09090a2030383232ull,0x3431303030317830ull,
0x3034347830203130ull,0x7830203038376330ull,0x3532343131303034ull,0x3030303030783020ull,
0x3009090a20303837ull,0x3236313030303678ull,0x3230303078302035ull,0x3378302030383734ull,
0x2035323231303130ull,0x3730303134637830ull,0x783009090a203038ull,0x3532343130303036ull,
0x3432303030783020ull,0x3032783020303837ull,0x3020303063393930ull,0x3234663930313278ull,
0x33783009090a2034ull,0x2031303030323030ull,0x3730303134637830ull,0x3030337830203038ull,
0x7830203532323132ull,0x3038373030313463ull,0x3032783009090a20ull,0x3020313030386333ull,
0x6666666666663078ull,0x6337303378302066ull,0x3678302064666431ull,0x2038653738303063ull,
0x303132783009090aull,0x7830203130323130ull,0x3038323233343430ull,0x3030303032783020ull,
0x3430783020313030ull,0x0a20303837633030ull,0x6530306478300909ull,0x3878302031303030ull,
0x2030383730306330ull,0x3130323030337830ull,0x3063367830206466ull,0x090a203865373830ull,
0x3131313031783009ull,0x3030783020333030ull,0x3020303832323030ull,0x3030383130303178ull,
0x3030303078302031ull,0x09090a2033303030ull,0x3032313130317830ull,0x3030307830203330ull,
0x7830203038373030ull,0x3130386630303031ull,0x6333303430783020ull,0x3009090a20303837ull,
0x6631306337303378ull,0x3130633678302064ull,0x6678302061663734ull,0x2031303030303030ull,
0x3030303030657830ull,0x783009090a203230ull,0x3330306132313061ull,0x3030303030783020ull,
0x3031783020303030ull,0x3020333030613231ull,0x3031333030303078ull,0x31783009090a2030ull,
0x2038303466303031ull,0x6361323030317830ull,0x3330327830203030ull,0x7830203330303533ull,
0x3038373030303030ull,0x3031783009090a20ull,0x3020633030383030ull,0x3234663030313178ull,
0x3030303178302034ull,0x3078302031306332ull,0x2030383763333036ull,0x333032783009090aull,
0x7830203330303635ull,0x3038373030303030ull,0x3130303031783020ull,0x3434783020393034ull,
0x0a20303837633030ull,0x6633303278300909ull,0x3078302035326539ull,0x2066666666666666ull,
0x3430353030347830ull,0x3030347830206335ull,0x090a203036383037ull,0x3034303036783009ull,
0x3030783020643536ull,0x3020303837633530ull,0x3661303630303678ull,0x3630303078302031ull,
0x09090a2030383730ull,0x6532303130337830ull,0x3134637830206435ull,0x7830203038373030ull,
0x3136303330313033ull,0x3030313463783020ull,0x3009090a20303837ull,0x3534303430303678ull,
0x3530303078302064ull,0x3678302030383763ull,0x2039303830363030ull,0x3730363030307830ull,
0x783009090a203038ull,0x6330656165303032ull,0x3830303032783020ull,0x3032783020633534ull,
0x3020313030616633ull,0x6666666666663078ull,0x32783009090a2066ull,0x2063303638633030ull,
0x3432303030317830ull,0x3030647830203031ull,0x7830203930363065ull,0x3038373030323061ull,
0x3032783009090a20ull,0x3020393038313030ull,0x3837633530343078ull,0x6530306478302030ull,
0x6178302031303430ull,0x2030383730303230ull,0x303066783009090aull,0x7830203130303030ull,
0x3230303030303065ull,0x6666313638783020ull,0x3030783020333065ull,0x0a20303030303030ull,
0x3431306178300909ull,0x3078302033303039ull,0x2030303030303030ull,0x3039343130317830ull,
0x3030307830203330ull,0x090a203030313330ull,0x6132303031783009ull,0x3131783020303063ull,
0x3020383034663030ull,0x3030353333303278ull,0x3030303078302033ull,0x09090a2030383730ull,
0x3038303030317830ull,0x3031317830206330ull,0x7830203432346630ull,0x3130633230303031ull,
0x6333303630783020ull,0x3009090a20303837ull,0x3030363533303278ull,0x3030303078302033ull,
0x3178302030383730ull,0x2039303431303030ull,0x3763303034347830ull,0x783009090a203038ull,
0x3532613036303034ull,0x3030303030783020ull,0x3036783020303837ull,0x3020353238303730ull,
0x3837343230303078ull,0x33783009090a2030ull,0x2035323231303130ull,0x3730303134637830ull,
0x3030367830203038ull,0x7830203532383036ull,0x3038373432303030ull,0x3032783009090a20ull,
0x3020643030303030ull,0x3837343230343078ull,0x6633313278302030ull,0x3078302039353466ull,
0x2066666666666666ull,0x303032783009090aull,0x7830206430363831ull,0x3330303030303030ull,
0x3230303033783020ull,0x6336783020646664ull,0x0a20386537343130ull,0x3030303178300909ull,
0x3078302064303231ull,0x2030303565333034ull,0x3431353030347830ull,0x3030307830203130ull,
0x090a203038373030ull,0x3134303036783009ull,0x3030783020313036ull,0x3020303837303030ull,
0x3030303031303378ull,0x3031346378302031ull,0x09090a2030383730ull,0x3431343030367830ull,
0x3030307830203130ull,0x7830203038373030ull,0x3030303865303032ull,0x3863303032783020ull,
0x3009090a20383030ull,0x3034306530306478ull,0x3032303878302031ull,0x3278302030383730ull,
0x2031303030313030ull,0x3030303030307830ull,0x783009090a203330ull,0x3130343065303064ull,
0x3030323061783020ull,0x3032783020303837ull,0x3020393038313030ull,0x3837633030343078ull,
0x64783009090a2030ull,0x2031303430653030ull,0x3730303230387830ull,0x3030327830203038ull,
0x7830203130303031ull,0x3330303030303030ull,0x3064783009090a20ull,0x3020313034306530ull,
0x3837303032306178ull,0x3030306678302030ull,0x6578302031303030ull,0x2032303030303030ull,
0x313638783009090aull,0x7830203330656666ull,0x3030303030303030ull,0x3038303464783020ull,
0x3032783020393030ull,0x0a20303837303030ull,0x3030393178300909ull,0x3178302030303065ull,
0x2038306338323030ull,0x3036313330327830ull,0x3030307830203330ull,0x090a203038373030ull,
0x3138313061783009ull,0x3030783020333030ull,0x3020303030303030ull,0x3530303030303178ull,
0x3330343078302039ull,0x09090a2030383763ull,0x3432303030307830ull,0x3030617830203130ull,
0x7830203065373030ull,0x3330303038313031ull,0x3230303030783020ull,0x3009090a20303031ull,
0x3034663030313178ull,0x3230303178302038ull,0x3278302030306361ull,0x2033303035333330ull,
0x3730303030307830ull,0x783009090a203038ull,0x3830303830303031ull,0x6630303131783020ull,
0x3031783020343234ull,0x3020313063323030ull,0x3837633330363078ull,0x32783009090a2030ull,
0x2033303036353330ull,0x3730303030307830ull,0x3330327830203038ull,0x7830203532383866ull,
0x6666666666666630ull,0x3032783009090a20ull,0x3020353034386633ull,0x6666666666663078ull,
0x6337303378302066ull,0x3678302064663530ull,0x2038653738303063ull,0x303031783009090aull,
0x7830203530323130ull,0x3038326533303430ull,0x3830303031783020ull,0x3031783020343130ull,
0x0a20303032383230ull,0x3938306478300909ull,0x3078302064663530ull,0x2038653730303434ull,
0x3464303031327830ull,0x3634307830203933ull,0x090a203038363633ull,0x3030303132783009ull,
0x3430783020643061ull,0x3020303836363234ull,0x3361306530303378ull,0x3130633678302039ull,
0x09090a2030383636ull,0x3630653030647830ull,0x3034307830203933ull,0x7830203038363230ull,
0x3933613030303031ull,0x6533303430783020ull,0x3009090a20303031ull,0x3034313030303178ull,
0x3030343478302064ull,0x3478302030383763ull,0x2063333030373030ull,0x6530343030347830ull,
0x783009090a203034ull,0x6433323036303036ull,0x6333303030783020ull,0x3036783020303837ull,
0x3020313463303530ull,0x3837303430303078ull,0x33783009090a2030ull,0x2064336531303130ull,
0x3730303134637830ull,0x3130337830203038ull,0x7830203134303230ull,0x3038373030313463ull,
0x3036783009090a20ull,0x3020313030303630ull,0x3837633330303078ull,0x3430303678302030ull,
0x3078302064306330ull,0x2030383730343030ull,0x303032783009090aull,0x7830203030633930ull,
0x6330363835303032ull,0x3039303033783020ull,0x6336783020353134ull,0x0a20303837343130ull,
0x3130303278300909ull,0x3078302035323438ull,0x2033303030303030ull,0x3030303030327830ull,
0x3034307830203130ull,0x090a203038373033ull,0x3065303064783009ull,0x3038783020313030ull,
0x3020303837303032ull,0x3038313030303278ull,0x3030343078302039ull,0x09090a2030383763ull,
0x3430653030647830ull,0x3230387830203930ull,0x7830203038373030ull,0x3531323135303064ull,
0x3030303430783020ull,0x3009090a20303837ull,0x3438303030306178ull,0x3130633078302031ull,
0x6178302030383738ull,0x2064333030303030ull,0x3738313063307830ull,0x783009090a203038ull,
0x6466313263373033ull,0x3031306336783020ull,0x3033783020386537ull,0x3020646666316337ull,
0x6532323130633678ull,0x31783009090a2038ull,0x2033303065373130ull,0x3132303030307830ull,
0x3830647830203030ull,0x7830203130633063ull,0x3038373030343430ull,0x3033783009090a20ull,
0x3020393036303230ull,0x3837303031346378ull,0x3030303278302030ull,0x3278302030303839ull,
0x2030303038323030ull,0x303032783009090aull,0x7830203930616162ull,0x3330303030303030ull,
0x3065303064783020ull,0x3061783020393030ull,0x0a20303837303063ull,0x3130303178300909ull,
0x3078302031303038ull,0x2033303030303030ull,0x3031383130317830ull,0x3030307830203330ull,
0x090a203038373030ull,0x6630303031783009ull,0x3630783020313038ull,0x3020303837633330ull,
0x3030313831303178ull,0x3030303078302033ull,0x09090a2030383730ull,0x3866303030317830ull,
0x3034307830203130ull,0x7830203038376333ull,0x3130303030303066ull,0x3030303065783020ull,
0x3009090a20323030ull,0x3065666631363878ull,0x3030303078302033ull,0x6178302030303030ull,
0x2033303065613130ull,0x3030303030307830ull,0x783009090a203030ull,0x3130303063373033ull,
0x3431306336783020ull,0x3061783020306537ull,0x3020646631303030ull,0x6637343130633078ull,
0x31783009090a2038ull,0x2033303065613130ull,0x3132303030307830ull,0x3031317830203030ull,
0x7830203830346630ull,0x3030636132303031ull,0x3032783009090a20ull,0x3020333030353333ull,
0x3837303030303078ull,0x6230303278302030ull,0x3078302039306161ull,0x2033303030303030ull,
0x383064783009090aull,0x7830206430343039ull,0x3038373030343430ull,0x3039383064783020ull,
0x3430783020313030ull,0x0a20303837303034ull,0x6131306178300909ull,0x3078302033303064ull,
0x2030303030303030ull,0x3730303030337830ull,0x3063367830206466ull,0x090a203865373830ull,
0x6161313031783009ull,0x3030783020333030ull,0x3020303832323030ull,0x3034313030303178ull,
0x3030343478302031ull,0x09090a2030383763ull,0x3430313030347830ull,0x3030307830206430ull,
0x7830203038373030ull,0x6430363030303036ull,0x6330303030783020ull,0x3009090a20303837ull,
0x3036303031303378ull,0x3031346378302064ull,0x3678302030383730ull,0x2064303430303030ull,
0x3763303030307830ull,0x783009090a203038ull,0x3130633063383064ull,0x3030343430783020ull,
0x3032783020303837ull,0x3020343236386530ull,0x3038393030303278ull,0x33783009090a2063ull,
0x2031303231323030ull,0x3730303134637830ull,0x3030327830203038ull,0x7830203130363030ull,
0x3038373030303430ull,0x3064783009090a20ull,0x3020313030306530ull,0x3837303063303878ull,
0x3230303378302030ull,0x3678302064663130ull,0x2038653738303063ull,0x313031783009090aull,
0x7830203330306361ull,0x3038323230303030ull,0x3130303031783020ull,0x3434783020313034ull,
0x0a20303837633030ull,0x3130303478300909ull,0x3078302035323431ull,0x2030383730303030ull,
0x3631303030367830ull,0x3030307830203532ull,0x090a203038373432ull,0x3130313033783009ull,
0x3463783020353232ull,0x3020303837303031ull,0x3234313030303678ull,0x3230303078302035ull,
0x09090a2030383734ull,0x6339393030327830ull,0x3031327830203030ull,0x7830203432346639ull,
0x3130303032303033ull,0x3030313463783020ull,0x3009090a20303837ull,0x3232313230303378ull,
0x3031346378302035ull,0x3278302030383730ull,0x2031303038633330ull,0x6666666666307830ull,
0x783009090a206666ull,0x6466643163373033ull,0x3830306336783020ull,0x3132783020386537ull,
0x3020313032313030ull,0x3832323334343078ull,0x32783009090a2030ull,0x2031303030303030ull,
0x3763303034307830ull,0x3030647830203038ull,0x7830203130303065ull,0x3038373030633038ull,
0x3033783009090a20ull,0x3020646631303230ull,0x6537383030633678ull,0x6131303178302038ull,
0x3078302033303063ull,0x2030383232303030ull,0x303031783009090aull,0x7830203130303831ull,
0x3330303030303030ull,0x6461313031783020ull,0x3030783020333030ull,0x0a20303837303030ull,
0x3030303178300909ull,0x3078302031303866ull,0x2030383763333034ull,0x3130633730337830ull,
0x3063367830206466ull,0x090a206166373431ull,0x3030303066783009ull,0x3065783020313030ull,
0x3020323030303030ull,0x3030356331306178ull,0x3030303078302033ull,0x09090a2030303030ull,
0x3035633130317830ull,0x3030307830203330ull,0x7830203030313330ull,0x3830346630303131ull,
0x6132303031783020ull,0x3009090a20303063ull,0x3030353333303278ull,0x3030303078302033ull,
0x3178302030383730ull,0x2063303038303030ull,0x3466303031317830ull,0x783009090a203432ull,
0x3130633230303031ull,0x6333303630783020ull,0x3032783020303837ull,0x3020333030363533ull,
0x3837303030303078ull,0x31783009090a2030ull,0x2039303431303030ull,0x3763303034347830ull,
0x3330327830203038ull,0x7830203532653966ull,0x6666666666666630ull,0x3034783009090a20ull,
0x3020633534303530ull,0x3638303730303478ull,0x3430303678302030ull,0x3078302064353630ull,
0x2030383763353030ull,0x303036783009090aull,0x7830203136613036ull,0x3038373036303030ull,
0x3230313033783020ull,0x3463783020643565ull,0x0a20303837303031ull,0x3031303378300909ull,
0x6378302031363033ull,0x2030383730303134ull,0x3430343030367830ull,0x3030307830206435ull,
0x090a203038376335ull,0x3036303036783009ull,0x3030783020393038ull,0x3020303837303630ull,
0x3065616530303278ull,0x3030303278302063ull,0x09090a2063353438ull,0x3061663330327830ull,
0x6666307830203130ull,0x7830206666666666ull,0x6330363863303032ull,0x3230303031783020ull,
0x3009090a20303134ull,0x3036306530306478ull,0x3032306178302039ull,0x3278302030383730ull,
0x2039303831303030ull,0x3763353034307830ull,0x783009090a203038ull,0x3130343065303064ull,
0x3030323061783020ull,0x3066783020303837ull,0x3020313030303030ull,0x3030303030306578ull,
0x38783009090a2032ull,0x2033306566663136ull,0x3030303030307830ull,0x3130617830203030ull,
0x7830203330303465ull,0x3030303030303030ull,0x3031783009090a20ull,0x3020333030346531ull,
0x3031333030303078ull,0x3030313178302030ull,0x3178302038303466ull,0x2030306361323030ull,
0x333032783009090aull,0x7830203330303533ull,0x3038373030303030ull,0x3830303031783020ull,
0x3131783020633030ull,0x0a20343234663030ull,0x3030303178300909ull,0x3078302031306332ull,
0x2030383763333036ull,0x3036353330327830ull,0x3030307830203330ull,0x090a203038373030ull,
0x3130303031783009ull,0x3434783020393034ull,0x3020303837633030ull,0x3261303630303478ull,
0x3030303078302035ull,0x09090a2030383730ull,0x3830373030367830ull,0x3030307830203532ull,
0x7830203038373432ull,0x3532323130313033ull,0x3030313463783020ull,0x3009090a20303837ull,
0x3238303630303678ull,0x3230303078302035ull,0x3278302030383734ull,0x2064303030303030ull,
0x3734323034307830ull,0x783009090a203038ull,0x6435346666333132ull,0x6666666630783020ull,
0x3032783020666666ull,0x3020643036383130ull,0x3030303030303078ull,0x33783009090a2033ull,
0x2064663130373130ull,0x3734313063367830ull,0x3030317830203865ull,0x7830206430323130ull,
0x3030356533303430ull,0x3034783009090a20ull,0x3020313034313530ull,0x3837303030303078ull,
0x3430303678302030ull,0x3078302031303631ull,0x2030383730303030ull,0x313033783009090aull,
0x7830203130303030ull,0x3038373030313463ull,0x3134303036783020ull,0x3030783020313034ull,
0x0a20303837303030ull,0x6530303278300909ull,0x3278302030303038ull,0x2038303038633030ull,
0x3430653030647830ull,0x3230387830203130ull,0x090a203038373030ull,0x3031303032783009ull,
0x3030783020313030ull,0x3020333030303030ull,0x3034306530306478ull,0x3032306178302031ull,
0x09090a2030383730ull,0x3831303030327830ull,0x3034307830203930ull,0x7830203038376330ull,
0x3130343065303064ull,0x3030323038783020ull,0x3009090a20303837ull,0x3030303130303278ull,
0x3030303078302031ull,0x6478302033303030ull,0x2031303430653030ull,0x3730303230617830ull,
0x783009090a203038ull,0x3130303030303066ull,0x3030303065783020ull,0x3638783020323030ull,
0x3020333065666631ull,0x3030303030303078ull,0x61783009090a2030ull,0x2033303064653130ull,
0x3030303030307830ull,0x3030307830203030ull,0x7830203130363230ull,0x3065373030303061ull,
0x3031783009090a20ull,0x3020333030646531ull,0x3031323030303078ull,0x3031346478302030ull,
0x3278302039303030ull,0x2030383730303030ull,0x303931783009090aull,0x7830203030306530ull,
0x3830633832303031ull,0x3631333032783020ull,0x3030783020333030ull,0x0a20303837303030ull,
0x3030303178300909ull,0x3078302039353030ull,0x2030383763333034ull,0x3030303030667830ull,
0x3030657830203130ull,0x090a203230303030ull,0x3032323061783009ull,0x3030783020333030ull,
0x3020303030303030ull,0x3038323030303078ull,0x3030306178302031ull,0x09090a2030653730ull,
0x3066313230317830ull,0x3030307830203330ull,0x7830203030313230ull,0x3830346630303131ull,
0x6132303031783020ull,0x3009090a20303063ull,0x3030353333303278ull,0x3030303078302033ull,
0x3178302030383730ull,0x2038303038303030ull,0x3466303031317830ull,0x783009090a203432ull,
0x3130633230303031ull,0x6333303630783020ull,0x3032783020303837ull,0x3020333030363533ull,
0x3837303030303078ull,0x32783009090a2030ull,0x2035323838663330ull,0x6666666666307830ull,
0x3330327830206666ull,0x7830203530343866ull,0x6666666666666630ull,0x3033783009090a20ull,
0x3020646635306337ull,0x6537383030633678ull,0x3030303178302038ull,0x3078302035303231ull,
0x2030383265333034ull,0x303031783009090aull,0x7830203431303830ull,0x3030323832303031ull,
0x3039383064783020ull,0x3430783020646635ull,0x0a20386537303034ull,0x3030313278300909ull,
0x3078302039333464ull,0x2030383636333634ull,0x6130303031327830ull,0x3434307830206430ull,
0x090a203038363632ull,0x3065303033783009ull,0x6336783020393361ull,0x3020303836363130ull,
0x3336306530306478ull,0x3030343078302039ull,0x09090a2030383632ull,0x6130303030317830ull,
0x3034307830203933ull,0x7830203030316533ull,0x6430343130303031ull,0x6330303434783020ull,
0x3009090a20303837ull,0x3330303730303478ull,0x3430303478302063ull,0x3678302030346530ull,
0x2064333230363030ull,0x3763333030307830ull,0x783009090a203038ull,0x3134633035303036ull,
0x3034303030783020ull,0x3033783020303837ull,0x3020643365313031ull,0x3837303031346378ull,
0x33783009090a2030ull,0x2031343032303130ull,0x3730303134637830ull,0x3030367830203038ull,
0x7830203130303036ull,0x3038376333303030ull,0x3036783009090a20ull,0x3020643063303430ull,
0x3837303430303078ull,0x3030303278302030ull,0x3278302030306339ull,0x2063303638353030ull,
0x303033783009090aull,0x7830203531343039ull,0x3038373431306336ull,0x3831303032783020ull,
0x3030783020353234ull,0x0a20333030303030ull,0x3030303278300909ull,0x3078302031303030ull,
0x2030383730333034ull,0x3030653030647830ull,0x3230387830203130ull,0x090a203038373030ull,
0x3130303032783009ull,0x3430783020393038ull,0x3020303837633030ull,0x3034306530306478ull,
0x3032303878302039ull,0x09090a2030383730ull,0x3231353030647830ull,0x3034307830203531ull,
0x7830203038373030ull,0x3134383030303061ull,0x3831306330783020ull,0x3009090a20303837ull,
0x3330303030306178ull,0x3130633078302064ull,0x3378302030383738ull,0x2064663132633730ull,
0x3730313063367830ull,0x783009090a203865ull,0x6466663163373033ull,0x3231306336783020ull,
0x3031783020386532ull,0x3020333030643132ull,0x3031323030303078ull,0x64783009090a2030ull,
0x2031306330633830ull,0x3730303434307830ull,0x3030337830203038ull,0x7830203930363032ull,
0x3038373030313463ull,0x3032783009090a20ull,0x3020303038393030ull,0x3030383230303278ull,
0x6430303278302030ull,0x3078302039306161ull,0x2033303030303030ull,0x303064783009090aull,
0x7830203930303065ull,0x3038373030633061ull,0x3831303031783020ull,0x3030783020313030ull,
0x0a20333030303030ull,0x3232303178300909ull,0x3078302033303030ull,0x2030383730303030ull,
0x3866303030317830ull,0x3036307830203130ull,0x090a203038376333ull,0x3032323031783009ull,
0x3030783020333030ull,0x3020303837303030ull,0x3038663030303178ull,0x3330343078302031ull,
0x09090a2030383763ull,0x3030303030667830ull,0x3030657830203130ull,0x7830203230303030ull,
0x3330656666313638ull,0x3030303030783020ull,0x3009090a20303030ull,0x3030643432306178ull,
0x3030303078302033ull,0x3378302030303030ull,0x2031303030633730ull,0x3734313063367830ull,
0x783009090a203065ull,0x6466313030303061ull,0x3431306330783020ull,0x3031783020386637ull,
0x3020333030643432ull,0x3031323030303078ull,0x31783009090a2030ull,0x2038303466303031ull,
0x6361323030317830ull,0x3330327830203030ull,0x7830203330303533ull,0x3038373030303030ull,
0x3032783009090a20ull,0x3020393061616430ull,0x3030303030303078ull,0x3938306478302033ull,
0x3078302064303430ull,0x2030383730303434ull,0x383064783009090aull,0x7830203130303039ull,
0x3038373030343430ull,0x6334323061783020ull,0x3030783020333030ull,0x0a20303030303030ull,
0x3030303378300909ull,0x3678302064663730ull,0x2038653738303063ull,0x3039343230317830ull,
0x3030307830203330ull,0x090a203038323230ull,0x3130303031783009ull,0x3434783020313034ull,
0x3020303837633030ull,0x3034303130303478ull,0x3030303078302064ull,0x09090a2030383730ull,
0x3630303030367830ull,0x3030307830206430ull,0x7830203038376330ull,0x6430363030313033ull,
0x3030313463783020ull,0x3009090a20303837ull,0x3034303030303678ull,0x3030303078302064ull,
0x6478302030383763ull,0x2031306330633830ull,0x3730303434307830ull,0x783009090a203038ull,
0x3432363865303032ull,0x3930303032783020ull,0x3033783020633038ull,0x3020313032313230ull,
0x3837303031346378ull,0x32783009090a2030ull,0x2031303630303030ull,0x3730303034307830ull,
0x3030647830203038ull,0x7830203130303065ull,0x3038373030633038ull,0x3033783009090a20ull,
0x3020646631303230ull,0x6537383030633678ull,0x3432303178302038ull,0x3078302033303062ull,
0x2030383232303030ull,0x303031783009090aull,0x7830203130343130ull,0x3038376330303434ull,
0x3131303034783020ull,0x3030783020353234ull,0x0a20303837303030ull,0x3030303678300909ull,
0x3078302035323631ull,0x2030383734323030ull,0x3231303130337830ull,0x3134637830203532ull,
0x090a203038373030ull,0x3130303036783009ull,0x3030783020353234ull,0x3020303837343230ull,
0x3063393930303278ull,0x3930313278302030ull,0x09090a2034323466ull,0x3030323030337830ull,
0x3134637830203130ull,0x7830203038373030ull,0x3532323132303033ull,0x3030313463783020ull,
0x3009090a20303837ull,0x3030386333303278ull,0x6666663078302031ull,0x3378302066666666ull,
0x2064666431633730ull,0x3738303063367830ull,0x783009090a203865ull,0x3130323130303132ull,
0x3233343430783020ull,0x3032783020303832ull,0x3020313030303030ull,0x3837633030343078ull,
0x64783009090a2030ull,0x2031303030653030ull,0x3730306330387830ull,0x3030337830203038ull,
0x7830206466313032ull,0x3865373830306336ull,0x3031783009090a20ull,0x3020333030623432ull,
0x3832323030303078ull,0x3130303178302030ull,0x3078302031303038ull,0x2033303030303030ull,
0x323031783009090aull,0x7830203330306334ull,0x3038373030303030ull,0x6630303031783020ull,
0x3430783020313038ull,0x0a20303837633330ull,0x6337303378300909ull,0x3678302064663130ull,
0x2061663734313063ull,0x3030303030667830ull,0x3030657830203130ull,0x090a203230303030ull,
0x3436323061783009ull,0x3030783020333030ull,0x3020303030303030ull,0x3030343632303178ull,
0x3030303078302033ull,0x09090a2030303133ull,0x3466303031317830ull,0x3030317830203830ull,
0x7830203030636132ull,0x3330303533333032ull,0x3030303030783020ull,0x3009090a20303837ull,
0x3030383030303178ull,0x3030313178302063ull,0x3178302034323466ull,0x2031306332303030ull,
0x3763333036307830ull,0x783009090a203038ull,0x3330303635333032ull,0x3030303030783020ull,
0x3031783020303837ull,0x3020393034313030ull,0x3837633030343478ull,0x32783009090a2030ull,
0x2035326539663330ull,0x6666666666307830ull,0x3030347830206666ull,0x7830206335343035ull,
0x3036383037303034ull,0x3036783009090a20ull,0x3020643536303430ull,0x3837633530303078ull,
0x3630303678302030ull,0x3078302031366130ull,0x2030383730363030ull,0x313033783009090aull,
0x7830206435653230ull,0x3038373030313463ull,0x3330313033783020ull,0x3463783020313630ull,
0x0a20303837303031ull,0x3430303678300909ull,0x3078302064353430ull,0x2030383763353030ull,
0x3830363030367830ull,0x3030307830203930ull,0x090a203038373036ull,0x6165303032783009ull,
0x3032783020633065ull,0x3020633534383030ull,0x3030616633303278ull,0x6666663078302031ull,
0x09090a2066666666ull,0x3638633030327830ull,0x3030317830206330ull,0x7830203031343230ull,
0x3930363065303064ull,0x3030323061783020ull,0x3009090a20303837ull,0x3038313030303278ull,
0x3530343078302039ull,0x6478302030383763ull,0x2031303430653030ull,0x3730303230617830ull,
0x783009090a203038ull,0x3130303030303066ull,0x3030303065783020ull,0x3638783020323030ull,
0x3020333065666631ull,0x3030303030303078ull,0x61783009090a2030ull,0x2033303033383230ull,
0x3030303030307830ull,0x3230317830203030ull,0x7830203330303338ull,0x3030313330303030ull,
0x3031783009090a20ull,0x3020303063613230ull,0x3034663030313178ull,0x3333303278302038ull,
0x3078302033303035ull,0x2030383730303030ull,0x303031783009090aull,0x7830206330303830ull,
0x3432346630303131ull,0x3230303031783020ull,0x3630783020313063ull,0x0a20303837633330ull,
0x3533303278300909ull,0x3078302033303036ull,0x2030383730303030ull,0x3431303030317830ull,
0x3034347830203930ull,0x090a203038376330ull,0x3036303034783009ull,0x3030783020353261ull,
0x3020303837303030ull,0x3238303730303678ull,0x3230303078302035ull,0x09090a2030383734ull,
0x3231303130337830ull,0x3134637830203532ull,0x7830203038373030ull,0x3532383036303036ull,
0x3432303030783020ull,0x3009090a20303837ull,0x3030383030303178ull,0x3930303278302063ull,
0x3278302030303638ull,0x2039353466663331ull,0x6666666666307830ull,0x783009090a206666ull,
0x3130303831303032ull,0x3030303030783020ull,0x3033783020333030ull,0x3020646637303631ull,
0x6537343130633678ull,0x31783009090a2038ull,0x2031303231303030ull,0x3565333036307830ull,
0x3030347830203030ull,0x7830206430343135ull,0x3038373030303030ull,0x3036783009090a20ull,
0x3020643036313430ull,0x3837633030303078ull,0x3031303378302030ull,0x6378302064303630ull,
0x2030383730303134ull,0x303036783009090aull,0x7830203930343134ull,0x3038376330303030ull,
0x3865303032783020ull,0x3032783020383034ull,0x0a20633034386330ull,0x6530306478300909ull,
0x3878302039303630ull,0x2030383730303230ull,0x3830313030327830ull,0x3030307830203131ull,
0x090a203330303030ull,0x3065303064783009ull,0x3061783020393036ull,0x3020303837303032ull,
0x3038313030303278ull,0x3030343078302039ull,0x09090a2030383730ull,0x3430653030647830ull,
0x3230387830203130ull,0x7830203038373030ull,0x3130303031303032ull,0x3030303030783020ull,
0x3009090a20333030ull,0x3034306530306478ull,0x3032306178302031ull,0x6678302030383730ull,
0x2031303030303030ull,0x3030303030657830ull,0x783009090a203230ull,0x3330656666313638ull,
0x3030303030783020ull,0x3032783020303030ull,0x3020353561613833ull,0x3130303030303078ull,
0x33783009090a2033ull,0x2064663764353130ull,0x3730313263367830ull,0x3030317830203865ull,
0x7830203330303136ull,0x3038323230303030ull,0x3031783009090a20ull,0x3020333030623832ull,
0x3837303030303078ull,0x6430303278302030ull,0x3078302031303639ull,0x2033303030303030ull,
0x303030783009090aull,0x7830203530303032ull,0x3038373030303063ull,0x3130303132783020ull,
0x3430783020646631ull,0x0a20386337343334ull,0x3030303178300909ull,0x3078302035303866ull,
0x2030383763333034ull,0x3866303030317830ull,0x3036307830203131ull,0x090a203038376333ull,
0x3163323031783009ull,0x3030783020333030ull,0x3020303839303030ull,0x3065303030303278ull,
0x3230343078302031ull,0x09090a2030383738ull,0x6334313030347830ull,0x3230307830203930ull,
0x7830203038373030ull,0x3930343030313033ull,0x3030313463783020ull,0x3009090a20303837ull,
0x3063343030303678ull,0x3032303078302031ull,0x3378302030383738ull,0x2031303030323030ull,
0x3730303134637830ull,0x783009090a203038ull,0x3531386630303031ull,0x6333303430783020ull,
0x3032783020303837ull,0x3020393138633030ull,0x3837303032343078ull,0x33783009090a2030ull,
0x2031306130383030ull,0x3730303134637830ull,0x3030327830203038ull,0x7830203130363130ull,
0x3038373030303430ull,0x3033783009090a20ull,0x3020313030303230ull,0x3837303031346378ull,
0x3030303278302030ull,0x3078302031306330ull,0x2030383730303034ull,0x303064783009090aull,
0x7830203930303065ull,0x3038373030633038ull,0x3065383064783020ull,0x3430783020313034ull,
0x0a20303837303034ull,0x3830303378300909ull,0x6578302064303430ull,0x2030383730303163ull,
0x3430303130337830ull,0x3163657830203533ull,0x090a203038373030ull,0x3031303034783009ull,
0x3030783020313330ull,0x3020303837303030ull,0x3230303130303378ull,0x3030633878302035ull,
0x09090a2030383730ull,0x3630653830647830ull,0x3434307830206430ull,0x7830203038373030ull,
0x3133323030303036ull,0x3033303030783020ull,0x3009090a20303837ull,0x3034303831303378ull,
0x3031636578302035ull,0x6478302030383730ull,0x2039306131653830ull,0x3730303434307830ull,
0x783009090a203038ull,0x3533633037303034ull,0x3030303030783020ull,0x3064783020303837ull,
0x3020353032306538ull,0x3837303034343078ull,0x33783009090a2030ull,0x2031333831303130ull,
0x3730303134637830ull,0x3030367830203038ull,0x7830203933653036ull,0x3038373433303030ull,
0x3034783009090a20ull,0x3020353338303530ull,0x3837303030303078ull,0x3030303678302030ull,
0x3078302031303030ull,0x2030383730333030ull,0x313033783009090aull,0x7830203933633130ull,
0x3038373030313463ull,0x3034303036783020ull,0x3030783020353361ull,0x0a20303837343330ull,
0x3330303478300909ull,0x3078302031333430ull,0x2030383730303030ull,0x3030303030617830ull,
0x3034347830203130ull,0x090a203038373431ull,0x3036303036783009ull,0x3030783020393363ull,
0x3020303837383330ull,0x3361313031303378ull,0x3031346378302035ull,0x09090a2030383730ull,
0x3630323030367830ull,0x3030307830203133ull,0x7830203038373033ull,0x6433303030303062ull,
0x3031303030783020ull,0x3009090a20303837ull,0x3363313030306178ull,0x3130343478302039ull,
0x3678302030383734ull,0x2031313830343030ull,0x3734333030307830ull,0x783009090a203038ull,
0x3130383130313033ull,0x3030313463783020ull,0x3033783020303837ull,0x3020313336303930ull,
0x3837303030633878ull,0x62783009090a2030ull,0x2035326331303030ull,0x3763333030307830ull,
0x3030617830203038ull,0x7830206430383030ull,0x3038373431303434ull,0x3036783009090a20ull,
0x3020313034303230ull,0x3837303030303078ull,0x3130303278302030ull,0x3078302035316138ull,
0x2033303030303030ull,0x303033783009090aull,0x7830203131343063ull,0x3038373030306338ull,
0x3030303062783020ull,0x3030783020393036ull,0x0a20303837343230ull,0x3030306178300909ull,
0x3478302031303030ull,0x2030383734313034ull,0x3131353030337830ull,0x3063367830206466ull,
0x090a203863373431ull,0x3034303033783009ull,0x6338783020353032ull,0x3020303837303030ull,
0x3130303030306278ull,0x3030303078302031ull,0x09090a2030383738ull,0x3036393230317830ull,
0x3030307830203330ull,0x7830203038323030ull,0x3130343165383064ull,0x3030343430783020ull,
0x3009090a20303837ull,0x3030386633303278ull,0x6666663078302031ull,0x3378302066666666ull,
0x2064663731303030ull,0x3730313034367830ull,0x783009090a203863ull,0x3330303366323061ull,
0x3030303030783020ull,0x3031783020303030ull,0x3020333030336632ull,0x3832303030303078ull,
0x32783009090a2030ull,0x2031306530303030ull,0x3738323034307830ull,0x3030347830203038ull,
0x7830203930633431ull,0x3038373030323030ull,0x3033783009090a20ull,0x3020393034303031ull,
0x3837303031346378ull,0x3830303378302030ull,0x6378302064303031ull,0x2030383730303134ull,
0x303036783009090aull,0x7830203130633430ull,0x3038373830323030ull,0x3130303032783020ull,
0x3430783020393036ull,0x0a20303837633030ull,0x3230303378300909ull,0x6378302031303030ull,
0x2030383730303134ull,0x3430323030337830ull,0x3134637830203930ull,0x090a203038373030ull,
0x6530303132783009ull,0x3032783020303038ull,0x3020303030383230ull,0x3030306530306478ull,
0x3063303878302039ull,0x09090a2030383730ull,0x3430653830647830ull,0x3434307830206430ull,
0x7830203038373030ull,0x3130343038303033ull,0x3030316365783020ull,0x3009090a20303837ull,
0x3234303031303378ull,0x3031636578302031ull,0x3478302030383730ull,0x2039316330373030ull,
0x3730303030307830ull,0x783009090a203038ull,0x3531363031303033ull,0x3030306338783020ull,
0x3064783020303837ull,0x3020313030306538ull,0x3837303034343078ull,0x36783009090a2030ull,
0x2039316530363030ull,0x3738313030307830ull,0x3130337830203038ull,0x7830203530343038ull,
0x3038373030316365ull,0x3064783009090a20ull,0x3020393030316538ull,0x3837303034343078ull,
0x3130303478302030ull,0x3078302031323030ull,0x2030383730303030ull,0x383064783009090aull,
0x7830203530323065ull,0x3038373030343430ull,0x3030313033783020ull,0x3463783020393163ull,
0x0a20303837303031ull,0x3030303678300909ull,0x3078302035323230ull,0x2030383730323030ull,
0x3830353030347830ull,0x3030307830203132ull,0x090a203038373030ull,0x3036303036783009ull,
0x3030783020643063ull,0x3020303837383130ull,0x3232313031303378ull,0x3031346378302035ull,
0x09090a2030383730ull,0x6130343030367830ull,0x3030307830203132ull,0x7830203038373032ull,
0x3931343033303034ull,0x3030303030783020ull,0x3009090a20303837ull,0x3036303030306178ull,
0x3130343478302064ull,0x3678302030383734ull,0x2035323030303030ull,0x3734323030307830ull,
0x783009090a203038ull,0x3132303130313033ull,0x3030313463783020ull,0x3036783020303837ull,
0x3020393136303230ull,0x3837383130303078ull,0x62783009090a2030ull,0x2039323630303030ull,
0x3730313030307830ull,0x3030617830203038ull,0x7830203532323130ull,0x3038373431303434ull,
0x3036783009090a20ull,0x3020313138303430ull,0x3837303230303078ull,0x3031303378302030ull,
0x6378302064306330ull,0x2030383730303134ull,0x303033783009090aull,0x7830203931303035ull,
0x3038373030306338ull,0x3130303062783020ull,0x3030783020353132ull,0x0a20303837383230ull,
0x3030306178300909ull,0x3478302031313830ull,0x2030383734313034ull,0x3430323030367830ull,
0x3030307830203130ull,0x090a203038376330ull,0x3036303033783009ull,0x6338783020643034ull,
0x3020303837303030ull,0x3038303030306278ull,0x3130303078302039ull,0x09090a2030383734ull,
0x3030303030617830ull,0x3034347830203130ull,0x7830203038373431ull,0x3530323033303033ull,
0x3030306338783020ull,0x3009090a20303837ull,0x3130303030306278ull,0x3030303078302031ull,
0x3078302030383738ull,0x2031303030303034ull,0x3730313234657830ull,0x783009090a203238ull,
0x3130303032303430ull,0x3430323465783020ull,0x3638783020303837ull,0x3020333065666631ull,
0x3030303030303078ull,0x31783009090a2030ull,0x2035303038303030ull,0x3030303030307830ull,
0x3030337830206230ull,0x7830206466333062ull,0x3863376330303436ull,0x3061783009090a20ull,
0x3020333030343033ull,0x3030303030303078ull,0x3033303178302030ull,0x3078302033303034ull,
0x2030383230303030ull,0x303032783009090aull,0x7830203130323030ull,0x3038376332303430ull,
0x3032303030783020ull,0x3063783020393030ull,0x0a20303837303030ull,0x3030353178300909ull,
0x6278302030303065ull,0x2030306137303039ull,0x3030303034307830ull,0x3234657830203130ull,
0x090a203038373030ull,0x3638303864783009ull,0x3032783020393038ull,0x3020303837303030ull,
0x3030303830346478ull,0x3030303278302064ull,0x09090a2030383730ull,0x3030303038317830ull,
0x3034347830203130ull,0x7830203038376330ull,0x6466316330306333ull,0x3831323436783020ull,
0x3009090a20386337ull,0x3030303030383178ull,0x3030343478302031ull,0x3078302030383763ull,
0x2031303030323034ull,0x3530303234657830ull,0x783009090a203030ull,0x3130303030303066ull,
0x3030303065783020ull,0x3638783020323030ull,0x3020333065666631ull,0x3030303030303078ull,
0x33783009090a2030ull,0x2035303230313030ull,0x3730303163657830ull,0x3730337830203038ull,
0x7830206466333063ull,0x3863373031306336ull,0x3031783009090a20ull,0x3020333030376632ull,
0x3832303030303078ull,0x3030303078302030ull,0x6178302031303232ull,0x2030653730303030ull,
0x303033783009090aull,0x7830203330303030ull,0x3030313230303030ull,0x3030303061783020ull,
0x3434783020353065ull,0x0a20303837343130ull,0x3030306178300909ull,0x3078302031306334ull,
0x2030383730303234ull,0x3230303030397830ull,0x3030307830203930ull,0x090a203038373030ull,
0x3032303033783009ull,0x3463783020353030ull,0x3020303837303031ull,0x3061643230316578ull,
0x3336303078302039ull,0x09090a2030383763ull,0x6363303030327830ull,0x3234307830203130ull,
0x7830203038373430ull,0x3530383638303064ull,0x3030303032783020ull,0x3009090a20303837ull,
0x3030306530306478ull,0x3063306178302039ull,0x3178302030383730ull,0x2030303065303035ull,
0x3066313031327830ull,0x783009090a203430ull,0x3130323065303064ull,0x3030633061783020ull,
0x3033783020303837ull,0x3020333030303030ull,0x3837303030303078ull,0x61783009090a2030ull,
0x2064303430303030ull,0x3734303034347830ull,0x3030397830203038ull,0x7830203532363030ull,
0x3038373030303030ull,0x3061783009090a20ull,0x3020643030303030ull,0x3837343630343478ull,
0x6533303278302030ull,0x3078302035323239ull,0x2066666666666666ull,0x303063783009090aull,
0x7830206430363039ull,0x3065376330303030ull,0x3030303061783020ull,0x3438783020643036ull,
0x0a20303837343630ull,0x3530303478300909ull,0x3078302039356330ull,0x2030383730303030ull,
0x6530343030367830ull,0x3030307830203935ull,0x090a203038373835ull,0x3230313033783009ull,
0x3463783020393563ull,0x3020303837303031ull,0x3563303430303678ull,0x3530303078302039ull,
0x09090a2030383738ull,0x3030303430327830ull,0x3034307830203935ull,0x7830203038373835ull,
0x3935633230303061ull,0x3436303434783020ull,0x3009090a20303837ull,0x3263323930306378ull,
0x3030303078302035ull,0x6178302030653763ull,0x2035323231303030ull,0x3734363034387830ull,
0x783009090a203038ull,0x6430363030303032ull,0x3432303430783020ull,0x3034783020303837ull,
0x3020353263303530ull,0x3837303030303078ull,0x36783009090a2030ull,0x2035326530343030ull,
0x3734323030307830ull,0x3130337830203038ull,0x7830203532323130ull,0x3038373030313463ull,
0x3036783009090a20ull,0x3020353263303430ull,0x3837343230303078ull,0x3030303378302030ull,
0x3078302035323231ull,0x2030383730303034ull,0x303033783009090aull,0x7830203532343039ull,
0x3038376330303436ull,0x3130303033783020ull,0x3430783020643032ull,0x0a20303837633030ull,
0x3530303478300909ull,0x3078302035326330ull,0x2030383730303030ull,0x6530343030367830ull,
0x3030307830203532ull,0x090a203038373432ull,0x3130313033783009ull,0x3463783020353232ull,
0x3020303837303031ull,0x3063303430303678ull,0x3230303078302064ull,0x09090a2030383734ull,
0x3630303030337830ull,0x3034307830203130ull,0x7830203038373030ull,0x6466353063373033ull,
0x3431303436783020ull,0x3009090a20386537ull,0x3030303230306478ull,0x3230343078302031ull,
0x3378302030303565ull,0x2033303030303030ull,0x3732303030307830ull,0x783009090a203038ull,
0x3130303030303066ull,0x3030303065783020ull,0x3061783020303030ull,0x3020643034303030ull,
0x3837343131343078ull,0x61783009090a2030ull,0x2035323630303030ull,0x3734303034347830ull,
0x3030617830203038ull,0x7830206435303030ull,0x3038373431313430ull,0x3039783009090a20ull,
0x3020313632313030ull,0x3837303030303078ull,0x3030306178302030ull,0x3478302035326532ull,
0x2030383734363034ull,0x333032783009090aull,0x7830203136306265ull,0x6666666666666630ull,
0x3138313063783020ull,0x3030783020353232ull,0x0a20306537633030ull,0x3030306178300909ull,
0x3878302035323231ull,0x2030383734363034ull,0x6330333130347830ull,0x3030307830203536ull,
0x090a203038373030ull,0x3032313036783009ull,0x3030783020353665ull,0x3020303837343630ull,
0x3632333031303378ull,0x3031346378302035ull,0x09090a2030383730ull,0x6330323130367830ull,
0x3030307830203536ull,0x7830203038373436ull,0x3536653230343032ull,0x3436303430783020ull,
0x3009090a20303837ull,0x3632333030306178ull,0x3630343478302035ull,0x6378302030383734ull,
0x2031363233383130ull,0x3763303030307830ull,0x783009090a203065ull,0x3136303330303061ull,
0x3436303438783020ull,0x3032783020303837ull,0x3020353232313030ull,0x3837303630343078ull,
0x34783009090a2030ull,0x2031363632363030ull,0x3730303030307830ull,0x3030367830203038ull,
0x7830203136343237ull,0x3038373036303030ull,0x3033783009090a20ull,0x3020313630333031ull,
0x3837303031346378ull,0x3630303678302030ull,0x3078302031363432ull,0x2030383730363030ull,
0x303033783009090aull,0x7830206435303330ull,0x3038376335303430ull,0x3037313033783020ull,
0x3436783020643036ull,0x0a20303837633030ull,0x3030306478300909ull,0x3078302031303430ull,
0x2030383738303034ull,0x3030663130337830ull,0x3134657830203130ull,0x090a203038373030ull,
0x3030303033783009ull,0x3430783020353236ull,0x3020303837343230ull,0x3030303030306178ull,
0x3130633278302064ull,0x09090a2030383734ull,0x3630393030647830ull,0x3034307830206430ull,
0x7830203038373830ull,0x6466353063373033ull,0x3431306336783020ull,0x3009090a20386537ull,
0x3030303030303278ull,0x3030343078302031ull,0x6478302030383763ull,0x2031303030323030ull,
0x3565323034307830ull,0x783009090a203030ull,0x3330303030303033ull,0x3230303030783020ull,
0x3066783020303837ull,0x3020313030303030ull,0x3030303030306578ull,0x61783009090a2030ull,
0x2064353231303030ull,0x3734313134307830ull,0x3030617830203038ull,0x7830203136653230ull,
0x3038373430303434ull,0x3061783009090a20ull,0x3020353630303030ull,0x3837343131343078ull,
0x3030303978302030ull,0x3078302039363033ull,0x2030383730303030ull,0x303061783009090aull,
0x7830203136323330ull,0x3038373436303434ull,0x6265333032783020ull,0x6630783020393634ull,
0x0a20666666666666ull,0x6131306378300909ull,0x3078302031363033ull,0x2030653763303030ull,
0x3033303030617830ull,0x3034387830203136ull,0x090a203038373436ull,0x3531333034783009ull,
0x3030783020643663ull,0x3020303837303030ull,0x3665353033303678ull,0x3630303078302064ull,
0x09090a2030383763ull,0x3633303130337830ull,0x3134637830206436ull,0x7830203038373030ull,
0x6436633530333036ull,0x6336303030783020ull,0x3009090a20303837ull,0x3632333034303278ull,
0x3630343078302064ull,0x6178302030383763ull,0x2064363633303030ull,0x3734363034347830ull,
0x783009090a203038ull,0x3936363361313063ull,0x6330303030783020ull,0x3061783020306537ull,
0x3020393634333030ull,0x3837343630343878ull,0x32783009090a2030ull,0x2031363033303030ull,
0x3738363034307830ull,0x3230347830203038ull,0x7830203936323665ull,0x3038373030303030ull,
0x3036783009090a20ull,0x3020393630366632ull,0x3837383630303078ull,0x3031303378302030ull,
0x6378302039363433ull,0x2030383730303134ull,0x323036783009090aull,0x7830203936303665ull,
0x3038373836303030ull,0x3330303033783020ull,0x3430783020393634ull,0x0a20303837343630ull,
0x6131303378300909ull,0x3678302039366532ull,0x2030383763303034ull,0x3433303030337830ull,
0x3034307830203136ull,0x090a203038373036ull,0x3665323034783009ull,0x3030783020393632ull,
0x3020303837303030ull,0x3630366632303678ull,0x3630303078302039ull,0x09090a2030383738ull,
0x3433303130337830ull,0x3134637830203936ull,0x7830203038373030ull,0x6435303665323036ull,
0x3836303030783020ull,0x3009090a20303837ull,0x3030306631303378ull,0x3031346578302031ull,
0x3378302030383730ull,0x2031366532303030ull,0x3734363034307830ull,0x783009090a203038ull,
0x6435303030303061ull,0x3431306332783020ull,0x3064783020303837ull,0x3020643565323831ull,
0x3837383030343078ull,0x33783009090a2030ull,0x2064663331633730ull,0x3734313063367830ull,
0x3030327830203865ull,0x7830203130303030ull,0x3038376335303430ull,0x3064783009090a20ull,
0x3020313030303930ull,0x3035653230343078ull,0x3030303378302030ull,0x3078302033303030ull,
0x2030383732303030ull,0x303066783009090aull,0x7830203130303030ull,0x3130303030303065ull,
0x000a7d0a7d090a20ull
};


}



extern "C" {


static const unsigned long long __deviceText_$compute_13$[] = {
0x6f69737265762e09ull,0x2e090a342e31206eull,0x7320746567726174ull,0x2f2f090a33315f6dull,
0x656c69706d6f6320ull,0x2f20687469772064ull,0x61636f6c2f727375ull,0x6f2f616475632f6cull,
0x696c2f34366e6570ull,0x2f090a65622f2f62ull,0x6e65706f766e202full,0x746c697562206363ull,
0x39303032206e6f20ull,0x0a0a39302d34302dull,0x752e206765722e09ull,0x313c617225203233ull,
0x65722e090a3b3e37ull,0x25203436752e2067ull,0x3b3e37313c616472ull,0x2e206765722e090aull,
0x3c61662520323366ull,0x722e090a3b3e3731ull,0x203436662e206765ull,0x3e37313c61646625ull,
0x206765722e090a3bull,0x767225203233752eull,0x722e090a3b3e353cull,0x203436752e206765ull,
0x3b3e353c76647225ull,0x2e206765722e090aull,0x3c76662520323366ull,0x65722e090a3b3e35ull,
0x25203436662e2067ull,0x0a3b3e353c766466ull,0x2d2d2d2f2f090a0aull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x6d6f43202f2f090aull,0x2f20676e696c6970ull,
0x78706d742f706d74ull,0x36303030305f7466ull,0x303030305f373965ull,0x6d5f372d30303030ull,
0x337070632e6e6961ull,0x706d742f2820692eull,0x432e23494263632full,0x090a2946665a516cull,
0x2d2d2d2d2d2d2f2full,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x090a0a2d2d2d2d2dull,
0x2d2d2d2d2d2d2f2full,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2f090a2d2d2d2d2dull,
0x6e6f6974704f202full,0x2d2d2f2f090a3a73ull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x5420202f2f090a2dull,0x74703a7465677261ull,0x733a415349202c78ull,
0x6e45202c33315f6dull,0x74696c3a6e616964ull,0x696f50202c656c74ull,0x7a6953207265746eull,
0x2f2f090a34363a65ull,0x4f2809334f2d2020ull,0x74617a696d697470ull,0x6576656c206e6f69ull,
0x20202f2f090a296cull,0x626544280930672dull,0x6c6576656c206775ull,0x2d20202f2f090a29ull,
0x6f7065522809326dull,0x7369766461207472ull,0x090a29736569726full,0x2d2d2d2d2d2d2f2full,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x090a0a2d2d2d2d2dull,0x093109656c69662eull,
0x6e616d6d6f633c22ull,0x223e656e696c2d64ull,0x09656c69662e090aull,0x2f706d742f220932ull,
0x305f746678706d74ull,0x5f37396536303030ull,0x3030303030303030ull,0x2e6e69616d5f362dull,
0x2e32656661647563ull,0x662e090a22757067ull,0x2f22093309656c69ull,0x2f62696c2f727375ull,
0x5f3638782f636367ull,0x78756e696c2d3436ull,0x332e342f756e672dull,0x756c636e692f332eull,
0x65646474732f6564ull,0x662e090a22682e66ull,0x2f22093409656c69ull,0x61636f6c2f727375ull,
0x622f616475632f6cull,0x6e692f2e2e2f6e69ull,0x72632f6564756c63ull,0x6563697665642f74ull,
0x656d69746e75725full,0x69662e090a22682eull,0x752f22093509656cull,0x6c61636f6c2f7273ull,
0x69622f616475632full,0x636e692f2e2e2f6eull,0x736f682f6564756cull,0x656e696665645f74ull,
0x662e090a22682e73ull,0x2f22093609656c69ull,0x61636f6c2f727375ull,0x622f616475632f6cull,
0x6e692f2e2e2f6e69ull,0x75622f6564756c63ull,0x79745f6e69746c69ull,0x090a22682e736570ull,
0x093709656c69662eull,0x6f6c2f7273752f22ull,0x616475632f6c6163ull,0x2f2e2e2f6e69622full,
0x2f6564756c636e69ull,0x745f656369766564ull,0x0a22682e73657079ull,0x3809656c69662e09ull,
0x6c2f7273752f2209ull,0x6475632f6c61636full,0x2e2e2f6e69622f61ull,0x6564756c636e692full,
0x5f7265766972642full,0x22682e7365707974ull,0x09656c69662e090aull,0x2f7273752f220939ull,
0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,0x7275747865742f65ull,
0x2e73657079745f65ull,0x6c69662e090a2268ull,0x752f220930310965ull,0x6c61636f6c2f7273ull,
0x69622f616475632full,0x636e692f2e2e2f6eull,0x6365762f6564756cull,0x657079745f726f74ull,
0x662e090a22682e73ull,0x2209313109656c69ull,0x636f6c2f7273752full,0x2f616475632f6c61ull,
0x692f2e2e2f6e6962ull,0x642f6564756c636eull,0x616c5f6563697665ull,0x7261705f68636e75ull,
0x2e73726574656d61ull,0x6c69662e090a2268ull,0x752f220932310965ull,0x6c61636f6c2f7273ull,
0x69622f616475632full,0x636e692f2e2e2f6eull,0x7472632f6564756cull,0x656761726f74732full,
0x682e7373616c635full,0x656c69662e090a22ull,0x73752f2209333109ull,0x64756c636e692f72ull,
0x742f737469622f65ull,0x0a22682e73657079ull,0x3109656c69662e09ull,0x2f7273752f220934ull,
0x2f6564756c636e69ull,0x0a22682e656d6974ull,0x3109656c69662e09ull,0x6f646e6172220935ull,
0x2e090a22682e636dull,0x09363109656c6966ull,0x65675f646e617222ull,0x2e090a2275632e6eull,
0x09373109656c6966ull,0x6b5f697274657022ull,0x75632e6c656e7265ull,0x656c69662e090a22ull,
0x73752f2209383109ull,0x2f6c61636f6c2f72ull,0x6e69622f61647563ull,0x6c636e692f2e2e2full,
0x6d6d6f632f656475ull,0x74636e75665f6e6full,0x0a22682e736e6f69ull,0x3109656c69662e09ull,
0x2f7273752f220939ull,0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,
0x75662f7472632f65ull,0x6f7263616d5f636eull,0x69662e090a22682eull,0x2f2209303209656cull,
0x61636f6c2f727375ull,0x622f616475632f6cull,0x6e692f2e2e2f6e69ull,0x616d2f6564756c63ull,
0x74636e75665f6874ull,0x0a22682e736e6f69ull,0x3209656c69662e09ull,0x2f7273752f220931ull,
0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,0x6563697665642f65ull,
0x6f6974636e75665full,0x2e090a22682e736eull,0x09323209656c6966ull,0x6f6c2f7273752f22ull,
0x616475632f6c6163ull,0x2f2e2e2f6e69622full,0x2f6564756c636e69ull,0x6e6f635f6874616dull,
0x682e73746e617473ull,0x656c69662e090a22ull,0x73752f2209333209ull,0x2f6c61636f6c2f72ull,
0x6e69622f61647563ull,0x6c636e692f2e2e2full,0x315f6d732f656475ull,0x63696d6f74615f31ull,
0x6f6974636e75665full,0x2e090a22682e736eull,0x09343209656c6966ull,0x6f6c2f7273752f22ull,
0x616475632f6c6163ull,0x2f2e2e2f6e69622full,0x2f6564756c636e69ull,0x74615f32315f6d73ull,
0x6e75665f63696d6full,0x682e736e6f697463ull,0x656c69662e090a22ull,0x73752f2209353209ull,
0x2f6c61636f6c2f72ull,0x6e69622f61647563ull,0x6c636e692f2e2e2full,0x315f6d732f656475ull,
0x656c62756f645f33ull,0x6f6974636e75665full,0x2e090a22682e736eull,0x09363209656c6966ull,
0x6f6c2f7273752f22ull,0x616475632f6c6163ull,0x2f2e2e2f6e69622full,0x2f6564756c636e69ull,
0x5f65727574786574ull,0x75665f6863746566ull,0x2e736e6f6974636eull,0x6c69662e090a2268ull,
0x752f220937320965ull,0x6c61636f6c2f7273ull,0x69622f616475632full,0x636e692f2e2e2f6eull,
0x74616d2f6564756cull,0x6974636e75665f68ull,0x5f6c62645f736e6full,0x0a22682e33787470ull,
0x65726168732e090aull,0x6e67696c612e2064ull,0x6d2038622e203420ull,0x3b5d363934325b74ull,
0x72746e652e090a0aull,0x655034315a5f2079ull,0x654b74656e697274ull,0x665069506c656e72ull,
0x0a28206969695f53ull,0x6d617261702e0909ull,0x5f5f203436752e20ull,0x6d72617061647563ull,
0x74655034315a5f5full,0x72654b74656e6972ull,0x53665069506c656eull,0x735f675f6969695full,
0x7261702e09090a2cull,0x203436752e206d61ull,0x6170616475635f5full,0x5034315a5f5f6d72ull,
0x4b74656e69727465ull,0x5069506c656e7265ull,0x675f6969695f5366ull,0x702e09090a2c765full,
0x36752e206d617261ull,0x616475635f5f2034ull,0x315a5f5f6d726170ull,0x656e697274655034ull,
0x506c656e72654b74ull,0x6969695f53665069ull,0x09090a2c6d5f675full,0x2e206d617261702eull,
0x75635f5f20323373ull,0x5f5f6d7261706164ull,0x697274655034315aull,0x656e72654b74656eull,
0x695f53665069506cull,0x09090a2c6e5f6969ull,0x2e206d617261702eull,0x75635f5f20323373ull,
0x5f5f6d7261706164ull,0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,
0x09090a2c735f6969ull,0x2e206d617261702eull,0x75635f5f20323373ull,0x5f5f6d7261706164ull,
0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,0x29646565735f6969ull,
0x65722e090a7b090aull,0x25203631752e2067ull,0x0a3b3e34313c6872ull,0x752e206765722e09ull,
0x36333c7225203233ull,0x65722e090a3b3e37ull,0x25203436752e2067ull,0x3b3e3236313c6472ull,
0x2e206765722e090aull,0x313c662520323366ull,0x65722e090a3b3e39ull,0x20646572702e2067ull,
0x0a3b3e30353c7025ull,0x316c6176202f2f09ull,0x2f2f090a34203d20ull,0x203d20326c617620ull,
0x6176202f2f090a38ull,0x0a3231203d20336cull,0x6b72616d202f2f09ull,0x090a3631203d2031ull,
0x326b72616d202f2full,0x2f090a3032203d20ull,0x0a30203d2079202full,0x373109636f6c2e09ull,
0x240a300931303209ull,0x315a5f5f3142424cull,0x656e697274655034ull,0x506c656e72654b74ull,
0x6969695f53665069ull,0x09636f6c2e090a3aull,0x3009383032093731ull,0x7261702e646c090aull,
0x09203233732e6d61ull,0x5f5f5b202c317225ull,0x6d72617061647563ull,0x74655034315a5f5full,
0x72654b74656e6972ull,0x53665069506c656eull,0x3b5d6e5f6969695full,0x363a6469202f2f09ull,
0x6475635f5f203132ull,0x5a5f5f6d72617061ull,0x6e69727465503431ull,0x6c656e72654b7465ull,
0x69695f5366506950ull,0x0a3078302b6e5f69ull,0x2e6f6c2e6c756d09ull,0x3272250920323373ull,
0x25202c317225202cull,0x09202020203b3172ull,0x6c756d090a202f2full,0x203233732e6f6c2eull,
0x7225202c33722509ull,0x2020203b32202c32ull,0x0a202f2f09202020ull,0x3233732e72687309ull,
0x25202c3472250920ull,0x203b3031202c3372ull,0x0920202020202020ull,0x766f6d090a202f2full,
0x722509203233752eull,0x2020203b30202c35ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x656c2e7074657309ull,0x702509203233732eull,0x202c347225202c31ull,0x092020203b357225ull,
0x702540090a202f2full,0x2409206172622031ull,0x3630375f305f744cull,0x20202020203b3835ull,
0x0a202f2f09202020ull,0x3233732e766f6d09ull,0x25202c3672250920ull,0x20202020203b3472ull,
0x0920202020202020ull,0x726873090a202f2full,0x722509203233732eull,0x202c337225202c37ull,
0x2020202020203b32ull,0x0a202f2f09202020ull,0x3233732e64646109ull,0x25202c3872250920ull,
0x3b377225202c3372ull,0x0920202020202020ull,0x2e646c090a202f2full,0x36752e6d61726170ull,
0x2c31647225092034ull,0x616475635f5f5b20ull,0x315a5f5f6d726170ull,0x656e697274655034ull,
0x506c656e72654b74ull,0x6969695f53665069ull,0x2f093b5d735f675full,0x3332363a6469202full,
0x70616475635f5f20ull,0x34315a5f5f6d7261ull,0x74656e6972746550ull,0x69506c656e72654bull,
0x5f6969695f536650ull,0x0a3078302b735f67ull,0x3233752e74766309ull,0x722509203631752eull,
0x6961746325202c39ull,0x092020203b782e64ull,0x6c756d090a202f2full,0x203233752e6f6c2eull,
0x25202c3031722509ull,0x3b397225202c3872ull,0x0a202f2f09202020ull,0x3436752e74766309ull,
0x722509203233752eull,0x30317225202c3264ull,0x092020202020203bull,0x6c756d090a202f2full,
0x203436752e6f6c2eull,0x25202c3364722509ull,0x203b34202c326472ull,0x0a202f2f09202020ull,
0x3436752e64646109ull,0x202c346472250920ull,0x7225202c31647225ull,0x09202020203b3364ull,
0x747663090a202f2full,0x3631752e3233752eull,0x202c313172250920ull,0x203b782e64697425ull,
0x0a202f2f09202020ull,0x3233732e766f6d09ull,0x202c323172250920ull,0x2020202020203b30ull,
0x0920202020202020ull,0x766f6d090a202f2full,0x722509203233732eull,0x3b367225202c3331ull,
0x2020202020202020ull,0x0a202f2f09202020ull,0x39345f305f744c24ull,0x2f2f200a3a363636ull,
0x4c203e706f6f6c3cull,0x79646f6220706f6full,0x303220656e696c20ull,0x697473656e202c38ull,
0x687470656420676eull,0x747365202c31203aull,0x6920646574616d69ull,0x6e6f697461726574ull,
0x6f6e6b6e75203a73ull,0x636f6c2e090a6e77ull,0x3009363709373109ull,0x33732e766f6d090aull,
0x2c34317225092032ull,0x3030333438363120ull,0x2020202020203b39ull,0x6873090a202f2f09ull,
0x2509203233622e6cull,0x317225202c353172ull,0x2020203b38202c32ull,0x202f2f0920202020ull,
0x33752e646461090aull,0x2c36317225092032ull,0x25202c3131722520ull,0x202020203b353172ull,
0x7663090a202f2f09ull,0x33752e3436752e74ull,0x2c35647225092032ull,0x20203b3631722520ull,
0x202f2f0920202020ull,0x6f6c2e6c756d090aull,0x722509203436752eull,0x35647225202c3664ull,
0x202020203b34202cull,0x6461090a202f2f09ull,0x2509203436752e64ull,0x647225202c376472ull,
0x3b36647225202c34ull,0x202f2f0920202020ull,0x6f6c672e7473090aull,0x203233732e6c6162ull,
0x302b376472255b09ull,0x3b34317225202c5dull,0x363a6469202f2f09ull,0x2e646461090a3532ull,
0x3172250920323373ull,0x2c32317225202c32ull,0x20202020203b3120ull,0x090a202f2f092020ull,
0x2e656e2e70746573ull,0x3270250920323373ull,0x25202c347225202cull,0x2f0920203b323172ull,
0x32702540090a202full,0x4c24092061726220ull,0x363639345f305f74ull,0x2020202020203b36ull,
0x090a202f2f092020ull,0x20696e752e617262ull,0x345f305f744c2409ull,0x2020203b34353139ull,
0x2f09202020202020ull,0x305f744c240a202full,0x0a3a38353630375full,0x3233732e72687309ull,
0x25202c3772250920ull,0x20203b32202c3372ull,0x0920202020202020ull,0x747663090a202f2full,
0x3631752e3233752eull,0x202c313172250920ull,0x203b782e64697425ull,0x0a202f2f09202020ull,
0x39345f305f744c24ull,0x6873090a3a343531ull,0x2509203233622e6cull,0x347225202c373172ull,
0x202020203b38202cull,0x202f2f0920202020ull,0x33732e627573090aull,0x2c38317225092032ull,
0x7225202c37722520ull,0x20202020203b3731ull,0x6573090a202f2f09ull,0x33752e65672e7074ull,
0x202c337025092032ull,0x7225202c31317225ull,0x202f2f09203b3831ull,0x622033702540090aull,
0x5f744c2409206172ull,0x3b38373130355f30ull,0x2020202020202020ull,0x6c2e090a202f2f09ull,
0x313809373109636full,0x2e766f6d090a3009ull,0x3172250920323373ull,0x3334383631202c39ull,
0x202020203b393030ull,0x090a202f2f092020ull,0x6d617261702e646cull,0x722509203436752eull,
0x635f5f5b202c3864ull,0x5f6d726170616475ull,0x7274655034315a5full,0x6e72654b74656e69ull,
0x5f53665069506c65ull,0x5d735f675f696969ull,0x3a6469202f2f093bull,0x75635f5f20333236ull,
0x5f5f6d7261706164ull,0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,
0x302b735f675f6969ull,0x2e747663090a3078ull,0x203631752e323375ull,0x25202c3032722509ull,
0x3b782e6469617463ull,0x090a202f2f092020ull,0x203233732e646461ull,0x25202c3132722509ull,
0x3b377225202c3372ull,0x2f09202020202020ull,0x2e6c756d090a202full,0x09203233752e6f6cull,
0x7225202c32327225ull,0x31327225202c3032ull,0x090a202f2f09203bull,0x2e3436752e747663ull,
0x6472250920323375ull,0x3b32327225202c39ull,0x2f09202020202020ull,0x2e6c756d090a202full,
0x09203436752e6f6cull,0x25202c3031647225ull,0x203b34202c396472ull,0x090a202f2f092020ull,
0x203436752e646461ull,0x202c313164722509ull,0x7225202c38647225ull,0x2f0920203b303164ull,
0x2e646461090a202full,0x3272250920323375ull,0x2c37317225202c33ull,0x20203b3131722520ull,
0x090a202f2f092020ull,0x2e3436752e747663ull,0x6472250920323375ull,0x33327225202c3231ull,
0x2f0920202020203bull,0x2e6c756d090a202full,0x09203436752e6f6cull,0x25202c3331647225ull,
0x3b34202c32316472ull,0x090a202f2f092020ull,0x203436752e646461ull,0x202c343164722509ull,
0x25202c3131647225ull,0x2f09203b33316472ull,0x672e7473090a202full,0x33732e6c61626f6cull,
0x316472255b092032ull,0x7225202c5d302b34ull,0x69202f2f093b3931ull,0x4c240a3632363a64ull,
0x373130355f305f74ull,0x636f6c2e090a3a38ull,0x3009343809373109ull,0x33752e766f6d090aull,
0x2c34327225092032ull,0x20202020203b3020ull,0x2020202020202020ull,0x6573090a202f2f09ull,
0x33752e71652e7074ull,0x202c347025092032ull,0x7225202c31317225ull,0x202f2f09203b3432ull,
0x203470252140090aull,0x744c240920617262ull,0x30393630355f305full,0x202020202020203bull,
0x6c2e090a202f2f09ull,0x363509363109636full,0x2e766f6d090a3009ull,0x6472250920343675ull,
0x203b746d202c3531ull,0x2020202020202020ull,0x090a202f2f092020ull,0x6d617261702e646cull,
0x722509203233732eull,0x635f5f5b202c3532ull,0x5f6d726170616475ull,0x7274655034315a5full,
0x6e72654b74656e69ull,0x5f53665069506c65ull,0x646565735f696969ull,0x6469202f2f093b5dull,
0x635f5f203732363aull,0x5f6d726170616475ull,0x7274655034315a5full,0x6e72654b74656e69ull,
0x5f53665069506c65ull,0x646565735f696969ull,0x7663090a3078302bull,0x31752e3233752e74ull,
0x2c36327225092036ull,0x2e64696174632520ull,0x202f2f0920203b78ull,0x33752e646461090aull,
0x2c37327225092032ull,0x25202c3532722520ull,0x202020203b363272ull,0x7473090a202f2f09ull,
0x2e6465726168732eull,0x746d5b0920323375ull,0x327225202c5d302bull,0x202f2f0920203b37ull,
0x6d203832363a6469ull,0x61090a3078302b74ull,0x09203436752e6464ull,0x25202c3631647225ull,
0x3b34202c35316472ull,0x2f2f092020202020ull,0x732e766f6d090a20ull,0x3832722509203233ull,
0x202020203b31202cull,0x2020202020202020ull,0x4c240a202f2f0920ull,0x313731355f305f74ull,
0x6c3c2f2f200a3a34ull,0x6f6f4c203e706f6full,0x6c2079646f622070ull,0x202c363520656e69ull,
0x20676e697473656eull,0x31203a6874706564ull,0x746172657469202cull,0x3236203a736e6f69ull,
0x09636f6c2e090a33ull,0x0a30093935093631ull,0x3436732e766f6d09ull,0x2c37316472250920ull,
0x203b363164722520ull,0x0920202020202020ull,0x2e646c090a202f2full,0x752e646572616873ull,
0x3932722509203233ull,0x37316472255b202cull,0x2f2f093b5d342d2bull,0x203932363a646920ull,
0x090a3078302b746dull,0x203233752e726873ull,0x25202c3033722509ull,0x3b3033202c393272ull,
0x2f09202020202020ull,0x2e726f78090a202full,0x3372250920323362ull,0x2c39327225202c31ull,
0x20203b3033722520ull,0x090a202f2f092020ull,0x2e3436752e747663ull,0x6472250920323375ull,
0x31337225202c3831ull,0x2f0920202020203bull,0x2e6c756d090a202full,0x09203436752e6f6cull,
0x25202c3931647225ull,0x3831202c38316472ull,0x3335323333343231ull,0x63090a202f2f093bull,
0x752e3233752e7476ull,0x3233722509203436ull,0x3b3931647225202cull,0x2f2f092020202020ull,
0x752e646461090a20ull,0x3333722509203233ull,0x202c38327225202cull,0x2020203b32337225ull,
0x73090a202f2f0920ull,0x6465726168732e74ull,0x255b09203233752eull,0x2c5d302b37316472ull,
0x2f093b3333722520ull,0x3033363a6469202full,0x0a3078302b746d20ull,0x3233732e64646109ull,
0x202c383272250920ull,0x3b31202c38327225ull,0x0920202020202020ull,0x646461090a202f2full,
0x722509203436752eull,0x647225202c363164ull,0x20203b34202c3731ull,0x0a202f2f09202020ull,
0x3233752e766f6d09ull,0x202c343372250920ull,0x202020203b343236ull,0x0920202020202020ull,
0x746573090a202f2full,0x3233732e656e2e70ull,0x25202c3570250920ull,0x337225202c383272ull,
0x0a202f2f09203b34ull,0x7262203570254009ull,0x305f744c24092061ull,0x203b34313731355full,
0x0920202020202020ull,0x5f744c240a202f2full,0x3a30393630355f30ull,0x36752e766f6d090aull,
0x3531647225092034ull,0x2020203b746d202cull,0x2020202020202020ull,0x6c2e090a202f2f09ull,
0x323609363109636full,0x2e726162090a3009ull,0x3b300920636e7973ull,0x2020202020202020ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x09373109636f6c2eull,0x6c090a3009303132ull,
0x2e6d617261702e64ull,0x3372250920323373ull,0x75635f5f5b202c35ull,0x5f5f6d7261706164ull,
0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,0x2f093b5d735f6969ull,
0x3831363a6469202full,0x70616475635f5f20ull,0x34315a5f5f6d7261ull,0x74656e6972746550ull,
0x69506c656e72654bull,0x5f6969695f536650ull,0x6d090a3078302b73ull,0x09203233752e766full,
0x3b30202c36337225ull,0x2020202020202020ull,0x2f2f092020202020ull,0x2e70746573090a20ull,
0x09203233732e656cull,0x337225202c367025ull,0x3b36337225202c35ull,0x40090a202f2f0920ull,
0x2061726220367025ull,0x375f305f744c2409ull,0x2020203b34313930ull,0x2f2f092020202020ull,
0x732e646461090a20ull,0x2c38722509203233ull,0x7225202c33722520ull,0x2020202020203b37ull,
0x6c090a202f2f0920ull,0x2e6d617261702e64ull,0x6472250920343675ull,0x635f5f5b202c3032ull,
0x5f6d726170616475ull,0x7274655034315a5full,0x6e72654b74656e69ull,0x5f53665069506c65ull,
0x5d735f675f696969ull,0x3a6469202f2f093bull,0x75635f5f20333236ull,0x5f5f6d7261706164ull,
0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,0x302b735f675f6969ull,
0x2e747663090a3078ull,0x203631752e323375ull,0x25202c3733722509ull,0x3b782e6469617463ull,
0x090a202f2f092020ull,0x752e6f6c2e6c756dull,0x3833722509203233ull,0x25202c387225202cull,
0x2f0920203b373372ull,0x2e747663090a202full,0x203233752e343675ull,0x202c313264722509ull,
0x2020203b38337225ull,0x090a202f2f092020ull,0x752e6f6c2e6c756dull,0x3264722509203436ull,
0x3132647225202c32ull,0x2f0920203b34202cull,0x2e646461090a202full,0x6472250920343675ull,
0x3032647225202c34ull,0x3b3232647225202cull,0x090a202f2f092020ull,0x203233732e646461ull,
0x25202c3933722509ull,0x3b317225202c3172ull,0x2f09202020202020ull,0x2e6c756d090a202full,
0x09203233732e6f6cull,0x7225202c30347225ull,0x3b317225202c3933ull,0x090a202f2f092020ull,
0x203233732e646461ull,0x25202c3134722509ull,0x313133202c353372ull,0x2f0920202020203bull,
0x2e766f6d090a202full,0x3472250920323373ull,0x313435312d202c32ull,0x203b313331333837ull,
0x090a202f2f092020ull,0x203233732e766f6dull,0x30202c3334722509ull,0x202020202020203bull,
0x2f09202020202020ull,0x70746573090a202full,0x203233732e746c2eull,0x7225202c37702509ull,
0x33347225202c3134ull,0x090a202f2f09203bull,0x203233732e736261ull,0x25202c3434722509ull,
0x202020203b313472ull,0x2f09202020202020ull,0x2e6c756d090a202full,0x09203233752e6968ull,
0x7225202c35347225ull,0x32347225202c3434ull,0x090a202f2f09203bull,0x203233752e627573ull,
0x25202c3634722509ull,0x347225202c343472ull,0x2f09202020203b35ull,0x2e726873090a202full,
0x3472250920323375ull,0x2c36347225202c37ull,0x20202020203b3120ull,0x090a202f2f092020ull,
0x203233752e646461ull,0x25202c3834722509ull,0x347225202c373472ull,0x2f09202020203b35ull,
0x2e726873090a202full,0x3472250920323373ull,0x2c38347225202c39ull,0x20202020203b3820ull,
0x090a202f2f092020ull,0x6275732037702540ull,0x722509203233732eull,0x33347225202c3934ull,
0x093b39347225202cull,0x766f6d090a202f2full,0x722509203233732eull,0x39347225202c3035ull,
0x202020202020203bull,0x0a202f2f09202020ull,0x3436752e74766309ull,0x722509203233752eull,
0x317225202c333264ull,0x0920202020203b31ull,0x766f6d090a202f2full,0x722509203233752eull,
0x3b363232202c3135ull,0x2020202020202020ull,0x0a202f2f09202020ull,0x656c2e7074657309ull,
0x702509203233752eull,0x2c31317225202c38ull,0x09203b3135722520ull,0x646461090a202f2full,
0x722509203233732eull,0x31317225202c3235ull,0x20203b343534202cull,0x0a202f2f09202020ull,
0x3233752e766f6d09ull,0x202c333572250920ull,0x202020203b363532ull,0x0920202020202020ull,
0x746573090a202f2full,0x3233752e746c2e70ull,0x25202c3970250920ull,0x357225202c313172ull,
0x0a202f2f09203b33ull,0x3233752e766f6d09ull,0x202c343572250920ull,0x202020203b313131ull,
0x0920202020202020ull,0x746573090a202f2full,0x3233752e656c2e70ull,0x202c303170250920ull,
0x7225202c31317225ull,0x0a202f2f093b3435ull,0x3233752e766f6d09ull,0x202c353572250920ull,
0x202020203b323131ull,0x0920202020202020ull,0x746573090a202f2full,0x3233752e746c2e70ull,
0x202c313170250920ull,0x7225202c31317225ull,0x0a202f2f093b3535ull,0x2e6f6c2e6c756d09ull,
0x6472250920343675ull,0x32647225202c3432ull,0x0920203b34202c33ull,0x766f6d090a202f2full,
0x722509203233732eull,0x3b323236202c3635ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x656c2e7074657309ull,0x702509203233732eull,0x32357225202c3231ull,0x093b36357225202cull,
0x646461090a202f2full,0x722509203436752eull,0x647225202c353264ull,0x31647225202c3432ull,
0x0a202f2f09203b35ull,0x3233732e766f6d09ull,0x202c373572250920ull,0x2020202020203b30ull,
0x0920202020202020ull,0x766f6d090a202f2full,0x722509203233732eull,0x30357225202c3835ull,
0x202020202020203bull,0x0a202f2f09202020ull,0x32355f305f744c24ull,0x2f2f200a3a383337ull,
0x4c203e706f6f6c3cull,0x79646f6220706f6full,0x313220656e696c20ull,0x697473656e202c30ull,
0x687470656420676eull,0x747365202c31203aull,0x6920646574616d69ull,0x6e6f697461726574ull,
0x6f6e6b6e75203a73ull,0x636f6c2e090a6e77ull,0x3009373909373109ull,0x203870252140090aull,
0x744c240920617262ull,0x34393932355f305full,0x202020202020203bull,0x2f2f200a202f2f09ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,
0x0936370936310963ull,0x68732e646c090a30ull,0x3233752e64657261ull,0x202c393572250920ull,
0x302b35326472255bull,0x6469202f2f093b5dull,0x2b746d203133363aull,0x646e61090a307830ull,
0x722509203233622eull,0x39357225202c3036ull,0x34373431322d202cull,0x2f093b3834363338ull,
0x732e646c090a202full,0x33752e6465726168ull,0x2c31367225092032ull,0x2b35326472255b20ull,
0x69202f2f093b5d34ull,0x746d203233363a64ull,0x6e61090a3078302bull,0x2509203233622e64ull,
0x367225202c323672ull,0x3437343132202c31ull,0x2f093b3734363338ull,0x622e726f090a202full,
0x3336722509203233ull,0x202c30367225202cull,0x2020203b32367225ull,0x090a202f2f092020ull,
0x09363109636f6c2eull,0x646c090a30093737ull,0x2e6465726168732eull,0x3672250920323375ull,
0x326472255b202c34ull,0x3b5d383835312b35ull,0x363a6469202f2f09ull,0x78302b746d203333ull,
0x752e726873090a30ull,0x3536722509203233ull,0x202c33367225202cull,0x2020202020203b31ull,
0x78090a202f2f0920ull,0x09203233622e726full,0x7225202c36367225ull,0x35367225202c3436ull,
0x2f2f09202020203bull,0x622e726f78090a20ull,0x3736722509203233ull,0x202c36367225202cull,
0x333834373237312dull,0x202f2f093b313836ull,0x33622e646e61090aull,0x2c38367225092032ull,
0x31202c3336722520ull,0x202020202020203bull,0x656e090a202f2f09ull,0x2509203233732e67ull,
0x367225202c393672ull,0x2020202020203b38ull,0x202f2f0920202020ull,0x752e74636c73090aull,
0x09203233732e3233ull,0x7225202c30377225ull,0x37367225202c3636ull,0x093b39367225202cull,
0x766f6d090a202f2full,0x722509203233732eull,0x30377225202c3137ull,0x202020202020203bull,
0x0a202f2f09202020ull,0x32355f305f744c24ull,0x2f2f200a3a343939ull,0x50203e706f6f6c3cull,
0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,
0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,0x0939370936310963ull,
0x732e726162090a30ull,0x203b300920636e79ull,0x2020202020202020ull,0x2020202020202020ull,
0x40090a202f2f0920ull,0x6172622038702521ull,0x5f305f744c240920ull,0x20203b3630353335ull,
0x2f2f092020202020ull,0x6f6c3c2f2f200a20ull,0x74726150203e706full,0x706f6f6c20666f20ull,
0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,
0x38333732355f305full,0x3109636f6c2e090aull,0x090a300932380936ull,0x203233732e766f6dull,
0x25202c3237722509ull,0x202020203b313772ull,0x2f09202020202020ull,0x732e7473090a202full,
0x33752e6465726168ull,0x326472255b092032ull,0x7225202c5d302b35ull,0x69202f2f093b3237ull,
0x746d203433363a64ull,0x744c240a3078302bull,0x36303533355f305full,0x6f6c3c2f2f200a3aull,
0x74726150203e706full,0x706f6f6c20666f20ull,0x696c2079646f6220ull,0x202c30313220656eull,
0x62616c2064616568ull,0x744c242064656c65ull,0x38333732355f305full,0x3109636f6c2e090aull,
0x090a300934380936ull,0x636e79732e726162ull,0x202020203b300920ull,0x2020202020202020ull,
0x2f09202020202020ull,0x70252140090a202full,0x2409206172622038ull,0x3034355f305f744cull,
0x20202020203b3831ull,0x200a202f2f092020ull,0x3e706f6f6c3c2f2full,0x666f207472615020ull,
0x6f6220706f6f6c20ull,0x20656e696c207964ull,0x616568202c303132ull,0x656c6562616c2064ull,
0x5f305f744c242064ull,0x2e090a3833373235ull,0x3909363109636f6cull,0x747663090a300930ull,
0x3233732e3436732eull,0x2c36326472250920ull,0x20203b3131722520ull,0x0a202f2f09202020ull,
0x2e6f6c2e6c756d09ull,0x6472250920343675ull,0x32647225202c3732ull,0x0920203b34202c36ull,
0x646461090a202f2full,0x722509203436752eull,0x647225202c383264ull,0x32647225202c3531ull,
0x0a202f2f09203b37ull,0x726168732e646c09ull,0x09203233752e6465ull,0x255b202c33377225ull,
0x3830392b38326472ull,0x6469202f2f093b5dull,0x2b746d203533363aull,0x646e61090a307830ull,
0x722509203233622eull,0x33377225202c3437ull,0x34373431322d202cull,0x2f093b3834363338ull,
0x732e646c090a202full,0x33752e6465726168ull,0x2c35377225092032ull,0x2b38326472255b20ull,
0x2f2f093b5d323139ull,0x203633363a646920ull,0x090a3078302b746dull,0x203233622e646e61ull,
0x25202c3637722509ull,0x343132202c353772ull,0x3b37343633383437ull,0x726f090a202f2f09ull,
0x722509203233622eull,0x34377225202c3336ull,0x203b36377225202cull,0x202f2f0920202020ull,
0x3109636f6c2e090aull,0x090a300931390936ull,0x65726168732e646cull,0x2509203233752e64ull,
0x72255b202c373772ull,0x093b5d302b353264ull,0x33363a6469202f2full,0x3078302b746d2037ull,
0x33752e726873090aull,0x2c38377225092032ull,0x31202c3336722520ull,0x202020202020203bull,
0x6f78090a202f2f09ull,0x2509203233622e72ull,0x377225202c393772ull,0x3b38377225202c37ull,
0x202f2f0920202020ull,0x33622e726f78090aull,0x2c30387225092032ull,0x2d202c3937722520ull,
0x3633383437323731ull,0x0a202f2f093b3138ull,0x3233622e646e6109ull,0x202c313872250920ull,
0x3b31202c33367225ull,0x0920202020202020ull,0x67656e090a202f2full,0x722509203233732eull,
0x31387225202c3238ull,0x202020202020203bull,0x0a202f2f09202020ull,0x33752e74636c7309ull,
0x2509203233732e32ull,0x377225202c333872ull,0x2c30387225202c39ull,0x2f093b3238722520ull,
0x2e766f6d090a202full,0x3772250920323373ull,0x3b33387225202c31ull,0x2020202020202020ull,
0x240a202f2f092020ull,0x3034355f305f744cull,0x3c2f2f200a3a3831ull,0x6150203e706f6f6cull,
0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,
0x242064656c656261ull,0x3732355f305f744cull,0x636f6c2e090a3833ull,0x3009333909363109ull,
0x79732e726162090aull,0x20203b300920636eull,0x2020202020202020ull,0x2020202020202020ull,
0x2140090a202f2f09ull,0x2061726220387025ull,0x355f305f744c2409ull,0x2020203b30333534ull,
0x202f2f0920202020ull,0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,0x20706f6f6c20666full,
0x6e696c2079646f62ull,0x68202c3031322065ull,0x6562616c20646165ull,0x5f744c242064656cull,
0x0a38333732355f30ull,0x363109636f6c2e09ull,0x6d090a3009363909ull,0x09203233732e766full,
0x7225202c34387225ull,0x20202020203b3137ull,0x2f2f092020202020ull,0x732e747663090a20ull,
0x09203233732e3436ull,0x25202c3932647225ull,0x202020203b313172ull,0x6d090a202f2f0920ull,
0x36752e6f6c2e6c75ull,0x3033647225092034ull,0x2c3932647225202cull,0x2f2f0920203b3420ull,
0x752e646461090a20ull,0x3364722509203436ull,0x3531647225202c31ull,0x3b3033647225202cull,
0x73090a202f2f0920ull,0x6465726168732e74ull,0x255b09203233752eull,0x3830392b31336472ull,
0x3b34387225202c5dull,0x363a6469202f2f09ull,0x78302b746d203833ull,0x5f305f744c240a30ull,
0x200a3a3033353435ull,0x3e706f6f6c3c2f2full,0x666f207472615020ull,0x6f6220706f6f6c20ull,
0x20656e696c207964ull,0x616568202c303132ull,0x656c6562616c2064ull,0x5f305f744c242064ull,
0x2e090a3833373235ull,0x3909363109636f6cull,0x726162090a300938ull,0x300920636e79732eull,
0x202020202020203bull,0x2020202020202020ull,0x0a202f2f09202020ull,0x2032317025214009ull,
0x744c240920617262ull,0x32343035355f305full,0x092020202020203bull,0x3c2f2f200a202f2full,
0x6150203e706f6f6cull,0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,
0x6c2064616568202cull,0x242064656c656261ull,0x3732355f305f744cull,0x636f6c2e090a3833ull,
0x0934303109363109ull,0x732e747663090a30ull,0x09203233732e3436ull,0x25202c3233647225ull,
0x202020203b313172ull,0x6d090a202f2f0920ull,0x36752e6f6c2e6c75ull,0x3333647225092034ull,
0x2c3233647225202cull,0x2f2f0920203b3420ull,0x752e646461090a20ull,0x3364722509203436ull,
0x3531647225202c34ull,0x3b3333647225202cull,0x6c090a202f2f0920ull,0x6465726168732e64ull,
0x722509203233752eull,0x6472255b202c3538ull,0x5d363138312b3433ull,0x3a6469202f2f093bull,
0x302b746d20393336ull,0x2e646e61090a3078ull,0x3872250920323362ull,0x2c35387225202c36ull,
0x3834373431322d20ull,0x2f2f093b38343633ull,0x68732e646c090a20ull,0x3233752e64657261ull,
0x202c373872250920ull,0x312b34336472255bull,0x2f2f093b5d303238ull,0x203034363a646920ull,
0x090a3078302b746dull,0x203233622e646e61ull,0x25202c3838722509ull,0x343132202c373872ull,
0x3b37343633383437ull,0x726f090a202f2f09ull,0x722509203233622eull,0x36387225202c3336ull,
0x203b38387225202cull,0x202f2f0920202020ull,0x3109636f6c2e090aull,0x0a30093530310936ull,
0x726168732e646c09ull,0x09203233752e6465ull,0x255b202c39387225ull,0x3830392b35326472ull,
0x6469202f2f093b5dull,0x2b746d203134363aull,0x726873090a307830ull,0x722509203233752eull,
0x33367225202c3039ull,0x202020203b31202cull,0x0a202f2f09202020ull,0x3233622e726f7809ull,
0x202c313972250920ull,0x7225202c39387225ull,0x09202020203b3039ull,0x726f78090a202f2full,
0x722509203233622eull,0x31397225202c3239ull,0x34373237312d202cull,0x2f093b3138363338ull,
0x2e646e61090a202full,0x3972250920323362ull,0x2c33367225202c33ull,0x20202020203b3120ull,
0x090a202f2f092020ull,0x203233732e67656eull,0x25202c3439722509ull,0x202020203b333972ull,
0x2f09202020202020ull,0x74636c73090a202full,0x3233732e3233752eull,0x202c353972250920ull,
0x7225202c31397225ull,0x34397225202c3239ull,0x6d090a202f2f093bull,0x09203233732e766full,
0x7225202c31377225ull,0x20202020203b3539ull,0x2f2f092020202020ull,0x5f305f744c240a20ull,
0x200a3a3234303535ull,0x3e706f6f6c3c2f2full,0x666f207472615020ull,0x6f6220706f6f6c20ull,
0x20656e696c207964ull,0x616568202c303132ull,0x656c6562616c2064ull,0x5f305f744c242064ull,
0x2e090a3833373235ull,0x3109363109636f6cull,0x6162090a30093730ull,0x0920636e79732e72ull,
0x2020202020203b30ull,0x2020202020202020ull,0x202f2f0920202020ull,0x323170252140090aull,
0x4c24092061726220ull,0x353535355f305f74ull,0x2020202020203b34ull,0x2f2f200a202f2f09ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,
0x3031310936310963ull,0x2e766f6d090a3009ull,0x3972250920323373ull,0x3b31377225202c36ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x2e3436732e747663ull,0x6472250920323373ull,
0x31317225202c3533ull,0x2f0920202020203bull,0x2e6c756d090a202full,0x09203436752e6f6cull,
0x25202c3633647225ull,0x3b34202c35336472ull,0x090a202f2f092020ull,0x203436752e646461ull,
0x202c373364722509ull,0x25202c3531647225ull,0x2f09203b36336472ull,0x732e7473090a202full,
0x33752e6465726168ull,0x336472255b092032ull,0x2c5d363138312b37ull,0x2f093b3639722520ull,
0x3234363a6469202full,0x0a3078302b746d20ull,0x35355f305f744c24ull,0x2f2f200a3a343535ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,
0x3231310936310963ull,0x2e726162090a3009ull,0x3b300920636e7973ull,0x2020202020202020ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x7262203470252140ull,0x305f744c24092061ull,
0x203b32323336355full,0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,
0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,0x616c206461656820ull,
0x4c242064656c6562ull,0x333732355f305f74ull,0x09636f6c2e090a38ull,0x3009373131093631ull,
0x6168732e646c090aull,0x203233752e646572ull,0x5b202c3739722509ull,0x20203b5d302b746dull,
0x363a6469202f2f09ull,0x78302b746d203832ull,0x622e646e61090a30ull,0x3839722509203233ull,
0x202c37397225202cull,0x3633383437343132ull,0x0a202f2f093b3734ull,0x726168732e646c09ull,
0x09203233752e6465ull,0x6d5b202c39397225ull,0x3b5d323934322b74ull,0x363a6469202f2f09ull,
0x78302b746d203334ull,0x646e61090a636239ull,0x722509203233622eull,0x397225202c303031ull,
0x373431322d202c39ull,0x093b383436333834ull,0x2e726f090a202f2full,0x3172250920323362ull,
0x38397225202c3130ull,0x3b3030317225202cull,0x0a202f2f09202020ull,0x3233732e766f6d09ull,
0x202c333672250920ull,0x20203b3130317225ull,0x0920202020202020ull,0x6f6c2e090a202f2full,
0x3831310936310963ull,0x2e726873090a3009ull,0x3172250920323375ull,0x30317225202c3230ull,
0x2020203b31202c31ull,0x090a202f2f092020ull,0x65726168732e646cull,0x2509203233752e64ull,
0x6d5b202c33303172ull,0x3b5d343835312b74ull,0x363a6469202f2f09ull,0x78302b746d203434ull,
0x726f78090a303336ull,0x722509203233622eull,0x317225202c343031ull,0x30317225202c3230ull,
0x0a202f2f09203b33ull,0x3233622e726f7809ull,0x2c35303172250920ull,0x202c343031722520ull,
0x333834373237312dull,0x202f2f093b313836ull,0x33622e646e61090aull,0x3630317225092032ull,
0x2c3130317225202cull,0x20202020203b3120ull,0x656e090a202f2f09ull,0x2509203233732e67ull,
0x7225202c37303172ull,0x202020203b363031ull,0x202f2f0920202020ull,0x752e74636c73090aull,
0x09203233732e3233ull,0x25202c3830317225ull,0x7225202c34303172ull,0x317225202c353031ull,
0x0a202f2f093b3730ull,0x726168732e747309ull,0x09203233752e6465ull,0x323934322b746d5bull,
0x3830317225202c5dull,0x3a6469202f2f093bull,0x302b746d20333436ull,0x7262090a63623978ull,
0x240920696e752e61ull,0x3036355f305f744cull,0x20202020203b3636ull,0x202f2f0920202020ull,
0x355f305f744c240aull,0x2f200a3a32323336ull,0x203e706f6f6c3c2full,0x20666f2074726150ull,
0x646f6220706f6f6cull,0x3220656e696c2079ull,0x64616568202c3031ull,0x64656c6562616c20ull,
0x355f305f744c2420ull,0x6f6d090a38333732ull,0x2509203233732e76ull,0x377225202c333672ull,
0x2020202020203b31ull,0x202f2f0920202020ull,0x355f305f744c240aull,0x2f200a3a36363036ull,
0x203e706f6f6c3c2full,0x20666f2074726150ull,0x646f6220706f6f6cull,0x3220656e696c2079ull,
0x64616568202c3031ull,0x64656c6562616c20ull,0x355f305f744c2420ull,0x6c2e090a38333732ull,
0x323109363109636full,0x726162090a300930ull,0x300920636e79732eull,0x202020202020203bull,
0x2020202020202020ull,0x0a202f2f09202020ull,0x363109636f6c2e09ull,0x090a300936323109ull,
0x203233752e726873ull,0x202c393031722509ull,0x3131202c33367225ull,0x2f0920202020203bull,
0x2e726f78090a202full,0x3172250920323362ull,0x30317225202c3031ull,0x3b33367225202c39ull,
0x090a202f2f092020ull,0x203233622e6c6873ull,0x202c313131722509ull,0x37202c3031317225ull,
0x2f0920202020203bull,0x2e646e61090a202full,0x3172250920323362ull,0x31317225202c3231ull,
0x383536312d202c31ull,0x093b363536383330ull,0x726f78090a202f2full,0x722509203233622eull,
0x317225202c333131ull,0x31317225202c3031ull,0x0a202f2f09203b32ull,0x3233622e6c687309ull,
0x2c34313172250920ull,0x202c333131722520ull,0x09202020203b3531ull,0x646e61090a202f2full,
0x722509203233622eull,0x317225202c353131ull,0x3237322d202c3431ull,0x093b343435363332ull,
0x726f78090a202f2full,0x722509203233622eull,0x317225202c363131ull,0x31317225202c3331ull,
0x0a202f2f09203b35ull,0x3233752e72687309ull,0x2c37313172250920ull,0x202c363131722520ull,
0x09202020203b3831ull,0x726f78090a202f2full,0x722509203233622eull,0x317225202c383131ull,
0x31317225202c3631ull,0x0a202f2f09203b37ull,0x3233732e766f6d09ull,0x202c313772250920ull,
0x20203b3831317225ull,0x0920202020202020ull,0x6f6c2e090a202f2full,0x3030310937310963ull,
0x732e646c090a3009ull,0x33752e6465726168ull,0x3931317225092032ull,0x35326472255b202cull,
0x202f2f093b5d302bull,0x6d203534363a6469ull,0x72090a3078302b74ull,0x09203233752e6d65ull,
0x25202c3032317225ull,0x7225202c39313172ull,0x2f2f0920203b3034ull,0x09636f6c2e090a20ull,
0x3009313031093731ull,0x203970252140090aull,0x744c240920617262ull,0x34333836355f305full,
0x202020202020203bull,0x2f2f200a202f2f09ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,
0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,0x2064656c6562616cull,
0x32355f305f744c24ull,0x6f6c2e090a383337ull,0x0938320937310963ull,0x732e766964090a30ull,
0x3231722509203233ull,0x3032317225202c31ull,0x20203b317225202cull,0x73090a202f2f0920ull,
0x09203233732e6275ull,0x25202c3232317225ull,0x203b31202c393372ull,0x2f2f092020202020ull,
0x732e766f6d090a20ull,0x3231722509203233ull,0x2020203b30202c33ull,0x2020202020202020ull,
0x73090a202f2f0920ull,0x732e71652e707465ull,0x3331702509203233ull,0x2c3132317225202cull,
0x093b333231722520ull,0x627573090a202f2full,0x722509203233732eull,0x317225202c343231ull,
0x20203b31202c3132ull,0x0a202f2f09202020ull,0x33732e706c657309ull,0x3532317225092032ull,
0x2c3232317225202cull,0x202c343231722520ull,0x2f2f093b33317025ull,0x732e766f6d090a20ull,
0x3231722509203233ull,0x3532317225202c36ull,0x202020202020203bull,0x72090a202f2f0920ull,
0x09203233732e6d65ull,0x25202c3732317225ull,0x7225202c30323172ull,0x2f2f092020203b31ull,
0x622e646e61090a20ull,0x3231722509203233ull,0x3132317225202c38ull,0x202020203b31202cull,
0x6d090a202f2f0920ull,0x09203233752e766full,0x30202c3932317225ull,0x202020202020203bull,
0x2f2f092020202020ull,0x2e70746573090a20ull,0x09203233732e7165ull,0x7225202c34317025ull,
0x317225202c383231ull,0x0a202f2f093b3932ull,0x6220343170254009ull,0x5f744c2409206172ull,
0x3b36343337355f30ull,0x0920202020202020ull,0x3c2f2f200a202f2full,0x6150203e706f6f6cull,
0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,
0x242064656c656261ull,0x3732355f305f744cull,0x636f6c2e090a3833ull,0x3009393209373109ull,
0x33732e646461090aull,0x3033317225092032ull,0x2c3732317225202cull,0x20202020203b3120ull,
0x6f6d090a202f2f09ull,0x2509203233732e76ull,0x3b30202c31333172ull,0x2020202020202020ull,
0x202f2f0920202020ull,0x33732e627573090aull,0x3233317225092032ull,0x31202c317225202cull,
0x202020202020203bull,0x6573090a202f2f09ull,0x33732e656e2e7074ull,0x2c35317025092032ull,
0x202c373231722520ull,0x2f093b3233317225ull,0x706c6573090a202full,0x722509203233732eull,
0x317225202c333331ull,0x33317225202c3033ull,0x3b35317025202c31ull,0x7262090a202f2f09ull,
0x240920696e752e61ull,0x3037355f305f744cull,0x20202020203b3039ull,0x202f2f0920202020ull,
0x355f305f744c240aull,0x2f200a3a36343337ull,0x203e706f6f6c3c2full,0x20666f2074726150ull,
0x646f6220706f6f6cull,0x3220656e696c2079ull,0x64616568202c3031ull,0x64656c6562616c20ull,
0x355f305f744c2420ull,0x6f6d090a38333732ull,0x2509203233732e76ull,0x7225202c33333172ull,
0x202020203b373231ull,0x202f2f0920202020ull,0x355f305f744c240aull,0x2f200a3a30393037ull,
0x203e706f6f6c3c2full,0x20666f2074726150ull,0x646f6220706f6f6cull,0x3220656e696c2079ull,
0x64616568202c3031ull,0x64656c6562616c20ull,0x355f305f744c2420ull,0x6f6d090a38333732ull,
0x2509203233732e76ull,0x7225202c34333172ull,0x202020203b333331ull,0x202f2f0920202020ull,
0x3109636f6c2e090aull,0x090a300930330937ull,0x203233732e646461ull,0x202c353331722509ull,
0x31202c3132317225ull,0x2f0920202020203bull,0x2e766f6d090a202full,0x3172250920323373ull,
0x20203b30202c3633ull,0x2020202020202020ull,0x090a202f2f092020ull,0x2e656e2e70746573ull,
0x3170250920323373ull,0x3232317225202c36ull,0x3b3132317225202cull,0x6573090a202f2f09ull,
0x09203233732e706cull,0x25202c3733317225ull,0x7225202c35333172ull,0x317025202c363331ull,
0x090a202f2f093b36ull,0x203233732e766f6dull,0x202c383331722509ull,0x20203b3733317225ull,
0x2f09202020202020ull,0x636f6c2e090a202full,0x3009313309373109ull,0x6f6c2e6c756d090aull,
0x722509203233732eull,0x317225202c393331ull,0x3b317225202c3532ull,0x6461090a202f2f09ull,
0x2509203233732e64ull,0x7225202c30343172ull,0x317225202c333331ull,0x202f2f09203b3933ull,
0x36752e747663090aull,0x2509203233732e34ull,0x7225202c38336472ull,0x202020203b303431ull,
0x6461090a202f2f09ull,0x2509203436752e64ull,0x7225202c39336472ull,0x647225202c383364ull,
0x202f2f0920203b34ull,0x6f6c672e646c090aull,0x092038732e6c6162ull,0x5b202c3134317225ull,
0x5d302b3933647225ull,0x3a6469202f2f093bull,0x766f6d090a383436ull,0x722509203233732eull,
0x317225202c323431ull,0x20202020203b3134ull,0x0a202f2f09202020ull,0x373109636f6c2e09ull,
0x6d090a3009323309ull,0x33732e6f6c2e6c75ull,0x3334317225092032ull,0x2c3132317225202cull,
0x2f2f093b31722520ull,0x732e646461090a20ull,0x3431722509203233ull,0x3732317225202c34ull,
0x3b3334317225202cull,0x63090a202f2f0920ull,0x732e3436752e7476ull,0x3464722509203233ull,
0x3434317225202c30ull,0x2f2f09202020203bull,0x752e646461090a20ull,0x3464722509203436ull,
0x3034647225202c31ull,0x203b34647225202cull,0x6c090a202f2f0920ull,0x6c61626f6c672e64ull,
0x317225092038732eull,0x6472255b202c3534ull,0x2f093b5d302b3134ull,0x3934363a6469202full,
0x33732e766f6d090aull,0x3634317225092032ull,0x3b3534317225202cull,0x2020202020202020ull,
0x6f6d090a202f2f09ull,0x2509203233732e76ull,0x3b30202c37343172ull,0x2020202020202020ull,
0x202f2f0920202020ull,0x74672e746573090aull,0x3233732e3233752eull,0x2c38343172250920ull,
0x202c353431722520ull,0x2f093b3734317225ull,0x2e67656e090a202full,0x3172250920323373ull,
0x34317225202c3934ull,0x2020202020203b38ull,0x090a202f2f092020ull,0x203233732e766f6dull,
0x202c303531722509ull,0x2020202020203b30ull,0x2f09202020202020ull,0x2e746573090a202full,
0x732e3233752e7467ull,0x3531722509203233ull,0x3134317225202c31ull,0x3b3035317225202cull,
0x656e090a202f2f09ull,0x2509203233732e67ull,0x7225202c32353172ull,0x202020203b313531ull,
0x202f2f0920202020ull,0x33622e646e61090aull,0x3335317225092032ull,0x2c3934317225202cull,
0x203b323531722520ull,0x6f6d090a202f2f09ull,0x2509203233752e76ull,0x3b30202c34353172ull,
0x2020202020202020ull,0x202f2f0920202020ull,0x652e70746573090aull,0x2509203233732e71ull,
0x317225202c373170ull,0x35317225202c3335ull,0x090a202f2f093b34ull,0x7262203731702540ull,
0x305f744c24092061ull,0x203b38353837355full,0x2f09202020202020ull,0x6c3c2f2f200a202full,
0x726150203e706f6full,0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,
0x616c206461656820ull,0x4c242064656c6562ull,0x333732355f305f74ull,0x09636f6c2e090a38ull,
0x0a30093633093731ull,0x3233732e64646109ull,0x2c35353172250920ull,0x37202c3735722520ull,
0x092020202020203bull,0x726873090a202f2full,0x722509203233732eull,0x347225202c363531ull,
0x2020203b32202c30ull,0x0a202f2f09202020ull,0x3436732e74766309ull,0x722509203233732eull,
0x317225202c323464ull,0x09202020203b3635ull,0x6c756d090a202f2full,0x203436752e6f6c2eull,
0x202c333464722509ull,0x34202c3234647225ull,0x0a202f2f0920203bull,0x3436752e64646109ull,
0x2c34346472250920ull,0x25202c3464722520ull,0x0920203b33346472ull,0x747663090a202f2full,
0x3233732e3436732eull,0x2c35346472250920ull,0x203b343431722520ull,0x0a202f2f09202020ull,
0x2e6f6c2e6c756d09ull,0x6472250920343675ull,0x34647225202c3634ull,0x0920203b34202c35ull,
0x646461090a202f2full,0x722509203436752eull,0x647225202c373464ull,0x34647225202c3434ull,
0x0a202f2f09203b36ull,0x626f6c672e747309ull,0x09203233732e6c61ull,0x302b37346472255bull,
0x3535317225202c5dull,0x3a6469202f2f093bull,0x766f6d090a303536ull,0x722509203233732eull,
0x203b31202c373531ull,0x2020202020202020ull,0x0a202f2f09202020ull,0x696e752e61726209ull,
0x5f305f744c240920ull,0x20203b3837353635ull,0x0920202020202020ull,0x5f744c240a202f2full,
0x3a38353837355f30ull,0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,0x20706f6f6c20666full,
0x6e696c2079646f62ull,0x68202c3031322065ull,0x6562616c20646165ull,0x5f744c242064656cull,
0x0a38333732355f30ull,0x3233732e766f6d09ull,0x2c37353172250920ull,0x20202020203b3020ull,
0x0920202020202020ull,0x617262090a202f2full,0x4c240920696e752eull,0x373536355f305f74ull,
0x2020202020203b38ull,0x0a202f2f09202020ull,0x36355f305f744c24ull,0x2f2f200a3a343338ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x766f6d090a383337ull,
0x722509203233732eull,0x203b30202c373531ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x36355f305f744c24ull,0x2f2f200a3a383735ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,
0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,0x2064656c6562616cull,
0x32355f305f744c24ull,0x6f6c2e090a383337ull,0x0939330937310963ull,0x732e726162090a30ull,
0x203b300920636e79ull,0x2020202020202020ull,0x2020202020202020ull,0x6d090a202f2f0920ull,
0x09203233732e766full,0x30202c3835317225ull,0x202020202020203bull,0x2f2f092020202020ull,
0x2e70746573090a20ull,0x09203233732e656eull,0x7225202c38317025ull,0x317225202c373531ull,
0x0a202f2f093b3835ull,0x2038317025214009ull,0x744c240920617262ull,0x34313138355f305full,
0x092020202020203bull,0x3c2f2f200a202f2full,0x6150203e706f6f6cull,0x6f6c20666f207472ull,
0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,0x242064656c656261ull,
0x3732355f305f744cull,0x636f6c2e090a3833ull,0x3009353409373109ull,0x33732e646461090aull,
0x3935317225092032ull,0x202c37357225202cull,0x2020202020203b37ull,0x6e61090a202f2f09ull,
0x2509203233622e64ull,0x7225202c30363172ull,0x203b31202c393531ull,0x202f2f0920202020ull,
0x33732e766964090aull,0x3136317225092032ull,0x2c3032317225202cull,0x2020203b31722520ull,
0x6e61090a202f2f09ull,0x2509203233622e64ull,0x7225202c32363172ull,0x203b31202c313631ull,
0x202f2f0920202020ull,0x652e70746573090aull,0x2509203233732e71ull,0x317225202c393170ull,
0x36317225202c3036ull,0x090a202f2f093b32ull,0x7262203931702540ull,0x305f744c24092061ull,
0x203b36363231315full,0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,
0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,0x616c206461656820ull,
0x4c242064656c6562ull,0x333732355f305f74ull,0x732e726873090a38ull,0x3631722509203233ull,
0x2c30347225202c33ull,0x20202020203b3220ull,0x63090a202f2f0920ull,0x732e3436732e7476ull,
0x3464722509203233ull,0x3336317225202c38ull,0x2f2f09202020203bull,0x6c2e6c756d090a20ull,
0x2509203436752e6full,0x7225202c39346472ull,0x203b34202c383464ull,0x61090a202f2f0920ull,
0x09203436752e6464ull,0x25202c3035647225ull,0x647225202c346472ull,0x2f2f0920203b3934ull,
0x732e766f6d090a20ull,0x3631722509203233ull,0x3433317225202c34ull,0x202020202020203bull,
0x6d090a202f2f0920ull,0x09203233732e766full,0x25202c3536317225ull,0x2020203b36323172ull,
0x2f2f092020202020ull,0x6c2e6c756d090a20ull,0x2509203233732e6full,0x7225202c36363172ull,
0x317225202c353631ull,0x61090a202f2f093bull,0x09203233732e6464ull,0x25202c3736317225ull,
0x7225202c34363172ull,0x2f2f09203b363631ull,0x732e747663090a20ull,0x09203233732e3436ull,
0x25202c3135647225ull,0x2020203b37363172ull,0x6d090a202f2f0920ull,0x36752e6f6c2e6c75ull,
0x3235647225092034ull,0x2c3135647225202cull,0x2f2f0920203b3420ull,0x752e646461090a20ull,
0x3564722509203436ull,0x3035647225202c33ull,0x3b3235647225202cull,0x6c090a202f2f0920ull,
0x6c61626f6c672e64ull,0x722509203233732eull,0x72255b202c383631ull,0x093b5d302b333564ull,
0x35363a6469202f2full,0x2e70746573090a31ull,0x09203233732e7165ull,0x7225202c30327025ull,
0x317225202c383631ull,0x0a202f2f093b3935ull,0x6220303270254009ull,0x5f744c2409206172ull,
0x3b36323431375f30ull,0x0920202020202020ull,0x3c2f2f200a202f2full,0x6150203e706f6f6cull,
0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,
0x242064656c656261ull,0x3732355f305f744cull,0x2e766f6d090a3833ull,0x3172250920323373ull,
0x33317225202c3936ull,0x2020202020203b38ull,0x090a202f2f092020ull,0x732e6f6c2e6c756dull,
0x3731722509203233ull,0x3936317225202c30ull,0x2f093b317225202cull,0x2e646461090a202full,
0x3172250920323373ull,0x37317225202c3137ull,0x203b317225202c30ull,0x090a202f2f092020ull,
0x2e3436732e747663ull,0x6472250920323373ull,0x37317225202c3435ull,0x2f09202020203b31ull,
0x2e6c756d090a202full,0x09203436752e6f6cull,0x25202c3535647225ull,0x3b34202c34356472ull,
0x090a202f2f092020ull,0x203436752e627573ull,0x202c363564722509ull,0x34202c3535647225ull,
0x2f0920202020203bull,0x2e646461090a202full,0x3172250920323373ull,0x37317225202c3237ull,
0x3436317225202c30ull,0x090a202f2f09203bull,0x2e3436732e747663ull,0x6472250920323373ull,
0x37317225202c3735ull,0x2f09202020203b32ull,0x2e6c756d090a202full,0x09203436752e6f6cull,
0x25202c3835647225ull,0x3b34202c37356472ull,0x090a202f2f092020ull,0x203436752e627573ull,
0x202c393564722509ull,0x34202c3835647225ull,0x2f0920202020203bull,0x2e766f6d090a202full,
0x3172250920323373ull,0x20203b30202c3337ull,0x2020202020202020ull,0x090a202f2f092020ull,
0x2e71652e70746573ull,0x3270250920323373ull,0x3436317225202c31ull,0x3b3337317225202cull,
0x6573090a202f2f09ull,0x09203436752e706cull,0x25202c3036647225ull,0x7225202c36356472ull,
0x327025202c393564ull,0x090a202f2f093b31ull,0x203436752e646461ull,0x202c313664722509ull,
0x25202c3036647225ull,0x2f09203b30356472ull,0x672e646c090a202full,0x33732e6c61626f6cull,
0x3437317225092032ull,0x31366472255b202cull,0x202f2f093b5d302bull,0x090a3235363a6469ull,
0x2e71652e70746573ull,0x3270250920323373ull,0x3437317225202c32ull,0x3b3935317225202cull,
0x2540090a202f2f09ull,0x2061726220323270ull,0x375f305f744c2409ull,0x2020203b36323431ull,
0x202f2f0920202020ull,0x315f305f744c240aull,0x2f200a3a36363231ull,0x203e706f6f6c3c2full,
0x20666f2074726150ull,0x646f6220706f6f6cull,0x3220656e696c2079ull,0x64616568202c3031ull,
0x64656c6562616c20ull,0x355f305f744c2420ull,0x6f6d090a38333732ull,0x2509203233732e76ull,
0x3b31202c35373172ull,0x2020202020202020ull,0x202f2f0920202020ull,0x6e752e617262090aull,
0x305f744c24092069ull,0x203b32303832315full,0x2020202020202020ull,0x744c240a202f2f09ull,
0x36323431375f305full,0x5f305f744c240a3aull,0x200a3a3232353131ull,0x3e706f6f6c3c2f2full,
0x666f207472615020ull,0x6f6220706f6f6c20ull,0x20656e696c207964ull,0x616568202c303132ull,
0x656c6562616c2064ull,0x5f305f744c242064ull,0x6d090a3833373235ull,0x09203233732e766full,
0x30202c3537317225ull,0x202020202020203bull,0x2f2f092020202020ull,0x5f305f744c240a20ull,
0x200a3a3230383231ull,0x3e706f6f6c3c2f2full,0x666f207472615020ull,0x6f6220706f6f6c20ull,
0x20656e696c207964ull,0x616568202c303132ull,0x656c6562616c2064ull,0x5f305f744c242064ull,
0x6d090a3833373235ull,0x09203233732e766full,0x30202c3637317225ull,0x202020202020203bull,
0x2f2f092020202020ull,0x2e70746573090a20ull,0x09203233732e656eull,0x7225202c38317025ull,
0x317225202c353731ull,0x0a202f2f093b3637ull,0x38355f305f744c24ull,0x2f2f200a3a343131ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x252140090a383337ull,
0x2061726220383170ull,0x355f305f744c2409ull,0x2020203b36323638ull,0x0a202f2f09202020ull,
0x706f6f6c3c2f2f20ull,0x6f2074726150203eull,0x6220706f6f6c2066ull,0x656e696c2079646full,
0x6568202c30313220ull,0x6c6562616c206461ull,0x305f744c24206465ull,0x090a38333732355full,
0x09373109636f6c2eull,0x6f6d090a30093435ull,0x2509203233732e76ull,0x7225202c37373172ull,
0x202020203b323431ull,0x202f2f0920202020ull,0x33732e627573090aull,0x3837317225092032ull,
0x2c3737317225202cull,0x20202020203b3120ull,0x6f6d090a202f2f09ull,0x2509203233732e76ull,
0x7225202c39373172ull,0x202020203b343331ull,0x202f2f0920202020ull,0x33732e766f6d090aull,
0x3038317225092032ull,0x3b3632317225202cull,0x2020202020202020ull,0x756d090a202f2f09ull,
0x3233732e6f6c2e6cull,0x2c31383172250920ull,0x202c303831722520ull,0x202f2f093b317225ull,
0x33732e646461090aull,0x3238317225092032ull,0x2c3937317225202cull,0x203b313831722520ull,
0x7663090a202f2f09ull,0x33732e3436752e74ull,0x3236647225092032ull,0x3b3238317225202cull,
0x202f2f0920202020ull,0x36752e646461090aull,0x3336647225092034ull,0x2c3236647225202cull,
0x20203b3464722520ull,0x7473090a202f2f09ull,0x2e6c61626f6c672eull,0x6472255b09203873ull,
0x25202c5d302b3336ull,0x2f2f093b38373172ull,0x0a3335363a646920ull,0x373109636f6c2e09ull,
0x6d090a3009353509ull,0x09203233732e766full,0x25202c3338317225ull,0x2020203b36343172ull,
0x2f2f092020202020ull,0x732e627573090a20ull,0x3831722509203233ull,0x3338317225202c34ull,
0x202020203b31202cull,0x64090a202f2f0920ull,0x09203233732e7669ull,0x25202c3538317225ull,
0x7225202c30323172ull,0x2f2f092020203b31ull,0x6c2e6c756d090a20ull,0x2509203233732e6full,
0x7225202c36383172ull,0x3538317225202c31ull,0x72090a202f2f093bull,0x09203233732e6d65ull,
0x25202c3738317225ull,0x7225202c30323172ull,0x2f2f092020203b31ull,0x732e646461090a20ull,
0x3831722509203233ull,0x3638317225202c38ull,0x3b3738317225202cull,0x63090a202f2f0920ull,
0x732e3436752e7476ull,0x3664722509203233ull,0x3838317225202c34ull,0x2f2f09202020203bull,
0x752e646461090a20ull,0x3664722509203436ull,0x3436647225202c35ull,0x203b34647225202cull,
0x73090a202f2f0920ull,0x6c61626f6c672e74ull,0x72255b092038732eull,0x202c5d302b353664ull,
0x2f093b3438317225ull,0x3435363a6469202full,0x355f305f744c240aull,0x2f200a3a36323638ull,
0x203e706f6f6c3c2full,0x20666f2074726150ull,0x646f6220706f6f6cull,0x3220656e696c2079ull,
0x64616568202c3031ull,0x64656c6562616c20ull,0x355f305f744c2420ull,0x6c2e090a38333732ull,
0x373509373109636full,0x2e726162090a3009ull,0x3b300920636e7973ull,0x2020202020202020ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x6220383170252140ull,0x5f744c2409206172ull,
0x3b38333139355f30ull,0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,
0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,0x616c206461656820ull,
0x4c242064656c6562ull,0x333732355f305f74ull,0x09636f6c2e090a38ull,0x0a30093036093731ull,
0x3233732e766f6d09ull,0x2c39383172250920ull,0x203b343331722520ull,0x0920202020202020ull,
0x766f6d090a202f2full,0x722509203233732eull,0x317225202c303931ull,0x20202020203b3833ull,
0x0a202f2f09202020ull,0x2e6f6c2e6c756d09ull,0x3172250920323373ull,0x39317225202c3139ull,
0x093b317225202c30ull,0x646461090a202f2full,0x722509203233732eull,0x317225202c323931ull,
0x39317225202c3938ull,0x0a202f2f09203b31ull,0x3436752e74766309ull,0x722509203233732eull,
0x317225202c363664ull,0x09202020203b3239ull,0x646461090a202f2full,0x722509203436752eull,
0x647225202c373664ull,0x34647225202c3636ull,0x0a202f2f0920203bull,0x626f6c672e646c09ull,
0x25092038732e6c61ull,0x72255b202c316872ull,0x093b5d302b373664ull,0x35363a6469202f2full,
0x732e646461090a35ull,0x3268722509203631ull,0x202c31687225202cull,0x2020202020203b31ull,
0x73090a202f2f0920ull,0x6c61626f6c672e74ull,0x72255b092038732eull,0x202c5d302b373664ull,
0x2f2f093b32687225ull,0x0a3635363a646920ull,0x373109636f6c2e09ull,0x72090a3009313609ull,
0x09203233732e6d65ull,0x25202c3732317225ull,0x7225202c30323172ull,0x2f2f092020203b31ull,
0x732e627573090a20ull,0x3931722509203233ull,0x202c317225202c33ull,0x2020202020203b31ull,
0x64090a202f2f0920ull,0x09203233732e7669ull,0x25202c3132317225ull,0x7225202c30323172ull,
0x2f2f092020203b31ull,0x2e70746573090a20ull,0x09203233732e656eull,0x7225202c33327025ull,
0x317225202c333931ull,0x0a202f2f093b3732ull,0x2e6f6c2e6c756d09ull,0x3172250920323373ull,
0x32317225202c3439ull,0x093b317225202c31ull,0x646461090a202f2full,0x722509203233732eull,
0x317225202c343431ull,0x32317225202c3439ull,0x0a202f2f09203b37ull,0x3436732e74766309ull,
0x722509203233732eull,0x317225202c383664ull,0x09202020203b3434ull,0x646461090a202f2full,
0x722509203436732eull,0x647225202c393664ull,0x20203b31202c3836ull,0x0a202f2f09202020ull,
0x3436732e74766309ull,0x722509203233732eull,0x317225202c303764ull,0x09202020203b3439ull,
0x6c6573090a202f2full,0x2509203436752e70ull,0x7225202c31376472ull,0x647225202c393664ull,
0x33327025202c3037ull,0x61090a202f2f093bull,0x09203436752e6464ull,0x25202c3237647225ull,
0x7225202c31376472ull,0x2f2f0920203b3464ull,0x6c672e646c090a20ull,0x2038732e6c61626full,
0x5b202c3368722509ull,0x5d302b3237647225ull,0x3a6469202f2f093bull,0x646461090a373536ull,
0x722509203631732eull,0x33687225202c3468ull,0x202020203b31202cull,0x0a202f2f09202020ull,
0x626f6c672e747309ull,0x5b092038732e6c61ull,0x5d302b3237647225ull,0x093b34687225202cull,
0x35363a6469202f2full,0x5f305f744c240a38ull,0x200a3a3833313935ull,0x3e706f6f6c3c2f2full,
0x666f207472615020ull,0x6f6220706f6f6c20ull,0x20656e696c207964ull,0x616568202c303132ull,
0x656c6562616c2064ull,0x5f305f744c242064ull,0x2e090a3833373235ull,0x3609373109636f6cull,
0x726162090a300933ull,0x300920636e79732eull,0x202020202020203bull,0x2020202020202020ull,
0x0a202f2f09202020ull,0x373109636f6c2e09ull,0x090a300935303109ull,0x65726168732e646cull,
0x2509203233752e64ull,0x255b202c35393172ull,0x3230312b35326472ull,0x69202f2f093b5d34ull,
0x746d203935363a64ull,0x6572090a3078302bull,0x2509203233752e6dull,0x7225202c30323172ull,
0x347225202c353931ull,0x202f2f0920203b30ull,0x3109636f6c2e090aull,0x0a30093630310937ull,
0x6220397025214009ull,0x5f744c2409206172ull,0x3b36303939355f30ull,0x0920202020202020ull,
0x3c2f2f200a202f2full,0x6150203e706f6f6cull,0x6f6c20666f207472ull,0x2079646f6220706full,
0x30313220656e696cull,0x6c2064616568202cull,0x242064656c656261ull,0x3732355f305f744cull,
0x636f6c2e090a3833ull,0x3009383209373109ull,0x33732e766964090aull,0x3639317225092032ull,
0x2c3032317225202cull,0x2020203b31722520ull,0x7573090a202f2f09ull,0x2509203233732e62ull,
0x7225202c32323172ull,0x20203b31202c3933ull,0x202f2f0920202020ull,0x33732e766f6d090aull,
0x3739317225092032ull,0x202020203b30202cull,0x2020202020202020ull,0x6573090a202f2f09ull,
0x33732e71652e7074ull,0x2c33317025092032ull,0x202c363931722520ull,0x2f093b3739317225ull,
0x2e627573090a202full,0x3172250920323373ull,0x39317225202c3432ull,0x2020203b31202c36ull,
0x090a202f2f092020ull,0x3233732e706c6573ull,0x2c35323172250920ull,0x202c323231722520ull,
0x25202c3432317225ull,0x202f2f093b333170ull,0x33732e766f6d090aull,0x3632317225092032ull,
0x3b3532317225202cull,0x2020202020202020ull,0x6572090a202f2f09ull,0x2509203233732e6dull,
0x7225202c38393172ull,0x317225202c303231ull,0x202f2f092020203bull,0x33622e646e61090aull,
0x3939317225092032ull,0x2c3639317225202cull,0x20202020203b3120ull,0x6f6d090a202f2f09ull,
0x2509203233752e76ull,0x3b30202c30303272ull,0x2020202020202020ull,0x202f2f0920202020ull,
0x652e70746573090aull,0x2509203233732e71ull,0x317225202c343270ull,0x30327225202c3939ull,
0x090a202f2f093b30ull,0x7262203432702540ull,0x305f744c24092061ull,0x203b38313430365full,
0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,0x6f6f6c20666f2074ull,
0x6c2079646f622070ull,0x2c30313220656e69ull,0x616c206461656820ull,0x4c242064656c6562ull,
0x333732355f305f74ull,0x09636f6c2e090a38ull,0x0a30093932093731ull,0x3233732e64646109ull,
0x2c31303272250920ull,0x202c383931722520ull,0x0920202020203b31ull,0x766f6d090a202f2full,
0x722509203233732eull,0x203b30202c323032ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x3233732e62757309ull,0x2c33303272250920ull,0x3b31202c31722520ull,0x0920202020202020ull,
0x746573090a202f2full,0x3233732e656e2e70ull,0x202c353270250920ull,0x25202c3839317225ull,
0x2f2f093b33303272ull,0x2e706c6573090a20ull,0x3272250920323373ull,0x30327225202c3430ull,
0x3230327225202c31ull,0x093b35327025202cull,0x617262090a202f2full,0x4c240920696e752eull,
0x363130365f305f74ull,0x2020202020203b32ull,0x0a202f2f09202020ull,0x30365f305f744c24ull,
0x2f2f200a3a383134ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,
0x313220656e696c20ull,0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,
0x766f6d090a383337ull,0x722509203233732eull,0x317225202c343032ull,0x20202020203b3839ull,
0x0a202f2f09202020ull,0x30365f305f744c24ull,0x2f2f200a3a323631ull,0x50203e706f6f6c3cull,
0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,
0x2064656c6562616cull,0x32355f305f744c24ull,0x766f6d090a383337ull,0x722509203233732eull,
0x327225202c343331ull,0x20202020203b3430ull,0x0a202f2f09202020ull,0x373109636f6c2e09ull,
0x61090a3009303309ull,0x09203233732e6464ull,0x25202c3530327225ull,0x3b31202c36393172ull,
0x2f2f092020202020ull,0x732e766f6d090a20ull,0x3032722509203233ull,0x2020203b30202c36ull,
0x2020202020202020ull,0x73090a202f2f0920ull,0x732e656e2e707465ull,0x3632702509203233ull,
0x2c3639317225202cull,0x093b323231722520ull,0x6c6573090a202f2full,0x2509203233732e70ull,
0x7225202c37303272ull,0x327225202c353032ull,0x36327025202c3630ull,0x6d090a202f2f093bull,
0x09203233732e766full,0x25202c3833317225ull,0x2020203b37303272ull,0x2f2f092020202020ull,
0x09636f6c2e090a20ull,0x0a30093133093731ull,0x2e6f6c2e6c756d09ull,0x3272250920323373ull,
0x32317225202c3830ull,0x093b317225202c35ull,0x646461090a202f2full,0x722509203233732eull,
0x327225202c393032ull,0x30327225202c3430ull,0x0a202f2f09203b38ull,0x3436752e74766309ull,
0x722509203233732eull,0x327225202c333764ull,0x09202020203b3930ull,0x646461090a202f2full,
0x722509203436752eull,0x647225202c343764ull,0x34647225202c3337ull,0x0a202f2f0920203bull,
0x626f6c672e646c09ull,0x25092038732e6c61ull,0x255b202c31343172ull,0x3b5d302b34376472ull,
0x363a6469202f2f09ull,0x2e766f6d090a3036ull,0x3172250920323373ull,0x34317225202c3234ull,
0x2020202020203b31ull,0x090a202f2f092020ull,0x09373109636f6c2eull,0x756d090a30093233ull,
0x3233732e6f6c2e6cull,0x2c30313272250920ull,0x202c363931722520ull,0x202f2f093b317225ull,
0x33732e646461090aull,0x3131327225092032ull,0x2c3839317225202cull,0x203b303132722520ull,
0x7663090a202f2f09ull,0x33732e3436752e74ull,0x3537647225092032ull,0x3b3131327225202cull,
0x202f2f0920202020ull,0x36752e646461090aull,0x3637647225092034ull,0x2c3537647225202cull,
0x20203b3464722520ull,0x646c090a202f2f09ull,0x2e6c61626f6c672eull,0x3431722509203873ull,
0x376472255b202c35ull,0x2f2f093b5d302b36ull,0x0a3136363a646920ull,0x3233732e766f6d09ull,
0x2c36343172250920ull,0x203b353431722520ull,0x0920202020202020ull,0x766f6d090a202f2full,
0x722509203233732eull,0x203b30202c323132ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x2e74672e74657309ull,0x203233732e323375ull,0x202c333132722509ull,0x25202c3534317225ull,
0x2f2f093b32313272ull,0x732e67656e090a20ull,0x3132722509203233ull,0x3331327225202c34ull,
0x202020202020203bull,0x6d090a202f2f0920ull,0x09203233732e766full,0x30202c3531327225ull,
0x202020202020203bull,0x2f2f092020202020ull,0x672e746573090a20ull,0x33732e3233752e74ull,
0x3631327225092032ull,0x2c3134317225202cull,0x093b353132722520ull,0x67656e090a202f2full,
0x722509203233732eull,0x327225202c373132ull,0x20202020203b3631ull,0x0a202f2f09202020ull,
0x3233622e646e6109ull,0x2c38313272250920ull,0x202c343132722520ull,0x09203b3731327225ull,
0x766f6d090a202f2full,0x722509203233752eull,0x203b30202c393132ull,0x2020202020202020ull,
0x0a202f2f09202020ull,0x71652e7074657309ull,0x702509203233732eull,0x31327225202c3732ull,
0x3931327225202c38ull,0x40090a202f2f093bull,0x6172622037327025ull,0x5f305f744c240920ull,
0x20203b3033393036ull,0x2f2f092020202020ull,0x6f6c3c2f2f200a20ull,0x74726150203e706full,
0x706f6f6c20666f20ull,0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,
0x744c242064656c65ull,0x38333732355f305full,0x3109636f6c2e090aull,0x090a300936330937ull,
0x203233732e646461ull,0x202c303232722509ull,0x3131202c37357225ull,0x2f0920202020203bull,
0x2e726873090a202full,0x3272250920323373ull,0x30347225202c3132ull,0x202020203b32202cull,
0x090a202f2f092020ull,0x2e3436732e747663ull,0x6472250920323373ull,0x32327225202c3737ull,
0x2f09202020203b31ull,0x2e6c756d090a202full,0x09203436752e6f6cull,0x25202c3837647225ull,
0x3b34202c37376472ull,0x090a202f2f092020ull,0x203436752e646461ull,0x202c393764722509ull,
0x7225202c34647225ull,0x2f0920203b383764ull,0x2e747663090a202full,0x203233732e343673ull,
0x202c303864722509ull,0x20203b3131327225ull,0x090a202f2f092020ull,0x752e6f6c2e6c756dull,
0x3864722509203436ull,0x3038647225202c31ull,0x2f0920203b34202cull,0x2e646461090a202full,
0x6472250920343675ull,0x37647225202c3238ull,0x3138647225202c39ull,0x090a202f2f09203bull,
0x61626f6c672e7473ull,0x5b09203233732e6cull,0x5d302b3238647225ull,0x3b3032327225202cull,
0x363a6469202f2f09ull,0x2e766f6d090a3236ull,0x3172250920323373ull,0x20203b31202c3735ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x20696e752e617262ull,0x355f305f744c2409ull,
0x2020203b30353639ull,0x2f09202020202020ull,0x305f744c240a202full,0x0a3a30333930365full,
0x706f6f6c3c2f2f20ull,0x6f2074726150203eull,0x6220706f6f6c2066ull,0x656e696c2079646full,
0x6568202c30313220ull,0x6c6562616c206461ull,0x305f744c24206465ull,0x090a38333732355full,
0x203233732e766f6dull,0x202c373531722509ull,0x2020202020203b30ull,0x2f09202020202020ull,
0x2e617262090a202full,0x744c240920696e75ull,0x30353639355f305full,0x202020202020203bull,
0x240a202f2f092020ull,0x3939355f305f744cull,0x3c2f2f200a3a3630ull,0x6150203e706f6f6cull,
0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,
0x242064656c656261ull,0x3732355f305f744cull,0x2e766f6d090a3833ull,0x3172250920323373ull,
0x20203b30202c3735ull,0x2020202020202020ull,0x240a202f2f092020ull,0x3639355f305f744cull,
0x3c2f2f200a3a3035ull,0x6150203e706f6f6cull,0x6f6c20666f207472ull,0x2079646f6220706full,
0x30313220656e696cull,0x6c2064616568202cull,0x242064656c656261ull,0x3732355f305f744cull,
0x636f6c2e090a3833ull,0x3009393309373109ull,0x79732e726162090aull,0x20203b300920636eull,
0x2020202020202020ull,0x2020202020202020ull,0x6f6d090a202f2f09ull,0x2509203233732e76ull,
0x3b30202c32323272ull,0x2020202020202020ull,0x202f2f0920202020ull,0x6e2e70746573090aull,
0x2509203233732e65ull,0x317225202c383170ull,0x32327225202c3735ull,0x090a202f2f093b32ull,
0x6220383170252140ull,0x5f744c2409206172ull,0x3b36383131365f30ull,0x2f09202020202020ull,
0x6c3c2f2f200a202full,0x726150203e706f6full,0x6f6f6c20666f2074ull,0x6c2079646f622070ull,
0x2c30313220656e69ull,0x616c206461656820ull,0x4c242064656c6562ull,0x333732355f305f74ull,
0x09636f6c2e090a38ull,0x0a30093534093731ull,0x3233732e64646109ull,0x2c33323272250920ull,
0x31202c3735722520ull,0x0920202020203b31ull,0x646e61090a202f2full,0x722509203233622eull,
0x327225202c343232ull,0x20203b31202c3332ull,0x0a202f2f09202020ull,0x3233732e76696409ull,
0x2c35323272250920ull,0x202c303231722520ull,0x092020203b317225ull,0x646e61090a202f2full,
0x722509203233622eull,0x327225202c363232ull,0x20203b31202c3532ull,0x0a202f2f09202020ull,
0x71652e7074657309ull,0x702509203233732eull,0x32327225202c3832ull,0x3632327225202c34ull,
0x40090a202f2f093bull,0x6172622038327025ull,0x5f305f744c240920ull,0x20203b3433343831ull,
0x2f2f092020202020ull,0x6f6c3c2f2f200a20ull,0x74726150203e706full,0x706f6f6c20666f20ull,
0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,
0x38333732355f305full,0x33732e726873090aull,0x3732327225092032ull,0x202c30347225202cull,
0x2020202020203b32ull,0x7663090a202f2f09ull,0x33732e3436732e74ull,0x3338647225092032ull,
0x3b3732327225202cull,0x202f2f0920202020ull,0x6f6c2e6c756d090aull,0x722509203436752eull,
0x647225202c343864ull,0x20203b34202c3338ull,0x6461090a202f2f09ull,0x2509203436752e64ull,
0x7225202c30356472ull,0x38647225202c3464ull,0x202f2f0920203b34ull,0x33732e766f6d090aull,
0x3436317225092032ull,0x3b3433317225202cull,0x2020202020202020ull,0x6f6d090a202f2f09ull,
0x2509203233732e76ull,0x7225202c38323272ull,0x202020203b363231ull,0x202f2f0920202020ull,
0x6f6c2e6c756d090aull,0x722509203233732eull,0x327225202c393232ull,0x3b317225202c3832ull,
0x6461090a202f2f09ull,0x2509203233732e64ull,0x7225202c30333272ull,0x327225202c343631ull,
0x202f2f09203b3932ull,0x36732e747663090aull,0x2509203233732e34ull,0x7225202c35386472ull,
0x202020203b303332ull,0x756d090a202f2f09ull,0x3436752e6f6c2e6cull,0x2c36386472250920ull,
0x202c353864722520ull,0x202f2f0920203b34ull,0x36752e646461090aull,0x3738647225092034ull,
0x2c3035647225202cull,0x203b363864722520ull,0x646c090a202f2f09ull,0x2e6c61626f6c672eull,
0x3272250920323373ull,0x6472255b202c3133ull,0x2f093b5d302b3738ull,0x3336363a6469202full,
0x652e70746573090aull,0x2509203233732e71ull,0x327225202c393270ull,0x32327225202c3133ull,
0x090a202f2f093b33ull,0x7262203932702540ull,0x305f744c24092061ull,0x203b34393132375full,
0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,0x6f6f6c20666f2074ull,
0x6c2079646f622070ull,0x2c30313220656e69ull,0x616c206461656820ull,0x4c242064656c6562ull,
0x333732355f305f74ull,0x732e766f6d090a38ull,0x3332722509203233ull,0x3833317225202c32ull,
0x202020202020203bull,0x6d090a202f2f0920ull,0x33732e6f6c2e6c75ull,0x3333327225092032ull,
0x2c3233327225202cull,0x2f2f093b31722520ull,0x732e646461090a20ull,0x3332722509203233ull,
0x3333327225202c34ull,0x20203b317225202cull,0x63090a202f2f0920ull,0x732e3436732e7476ull,
0x3864722509203233ull,0x3433327225202c38ull,0x2f2f09202020203bull,0x6c2e6c756d090a20ull,
0x2509203436752e6full,0x7225202c39386472ull,0x203b34202c383864ull,0x73090a202f2f0920ull,
0x09203436752e6275ull,0x25202c3039647225ull,0x3b34202c39386472ull,0x2f2f092020202020ull,
0x732e646461090a20ull,0x3332722509203233ull,0x3333327225202c35ull,0x3b3436317225202cull,
0x63090a202f2f0920ull,0x732e3436732e7476ull,0x3964722509203233ull,0x3533327225202c31ull,
0x2f2f09202020203bull,0x6c2e6c756d090a20ull,0x2509203436752e6full,0x7225202c32396472ull,
0x203b34202c313964ull,0x73090a202f2f0920ull,0x09203436752e6275ull,0x25202c3339647225ull,
0x3b34202c32396472ull,0x2f2f092020202020ull,0x732e766f6d090a20ull,0x3332722509203233ull,
0x2020203b30202c36ull,0x2020202020202020ull,0x73090a202f2f0920ull,0x732e71652e707465ull,
0x3033702509203233ull,0x2c3436317225202cull,0x093b363332722520ull,0x6c6573090a202f2full,
0x2509203436752e70ull,0x7225202c34396472ull,0x647225202c303964ull,0x30337025202c3339ull,
0x61090a202f2f093bull,0x09203436752e6464ull,0x25202c3539647225ull,0x7225202c34396472ull,
0x2f2f09203b303564ull,0x6c672e646c090a20ull,0x3233732e6c61626full,0x2c37333272250920ull,
0x2b35396472255b20ull,0x69202f2f093b5d30ull,0x73090a3436363a64ull,0x732e71652e707465ull,
0x3133702509203233ull,0x2c3733327225202cull,0x093b333232722520ull,0x702540090a202f2full,
0x0920617262203133ull,0x32375f305f744c24ull,0x202020203b343931ull,0x0a202f2f09202020ull,
0x38315f305f744c24ull,0x2f2f200a3a343334ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,
0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,0x2064656c6562616cull,
0x32355f305f744c24ull,0x766f6d090a383337ull,0x722509203233732eull,0x203b31202c383332ull,
0x2020202020202020ull,0x0a202f2f09202020ull,0x696e752e61726209ull,0x5f305f744c240920ull,
0x20203b3037393931ull,0x0920202020202020ull,0x5f744c240a202f2full,0x3a34393132375f30ull,
0x315f305f744c240aull,0x2f200a3a30393638ull,0x203e706f6f6c3c2full,0x20666f2074726150ull,
0x646f6220706f6f6cull,0x3220656e696c2079ull,0x64616568202c3031ull,0x64656c6562616c20ull,
0x355f305f744c2420ull,0x6f6d090a38333732ull,0x2509203233732e76ull,0x3b30202c38333272ull,
0x2020202020202020ull,0x202f2f0920202020ull,0x315f305f744c240aull,0x2f200a3a30373939ull,
0x203e706f6f6c3c2full,0x20666f2074726150ull,0x646f6220706f6f6cull,0x3220656e696c2079ull,
0x64616568202c3031ull,0x64656c6562616c20ull,0x355f305f744c2420ull,0x6f6d090a38333732ull,
0x2509203233732e76ull,0x3b30202c39333272ull,0x2020202020202020ull,0x202f2f0920202020ull,
0x6e2e70746573090aull,0x2509203233732e65ull,0x327225202c383170ull,0x33327225202c3833ull,
0x240a202f2f093b39ull,0x3131365f305f744cull,0x3c2f2f200a3a3638ull,0x6150203e706f6f6cull,
0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,
0x242064656c656261ull,0x3732355f305f744cull,0x70252140090a3833ull,0x0920617262203831ull,
0x31365f305f744c24ull,0x202020203b383936ull,0x200a202f2f092020ull,0x3e706f6f6c3c2f2full,
0x666f207472615020ull,0x6f6220706f6f6c20ull,0x20656e696c207964ull,0x616568202c303132ull,
0x656c6562616c2064ull,0x5f305f744c242064ull,0x2e090a3833373235ull,0x3509373109636f6cull,
0x766f6d090a300934ull,0x722509203233732eull,0x317225202c303432ull,0x20202020203b3234ull,
0x0a202f2f09202020ull,0x3233732e62757309ull,0x2c31343272250920ull,0x202c303432722520ull,
0x0920202020203b31ull,0x766f6d090a202f2full,0x722509203233732eull,0x317225202c323432ull,
0x20202020203b3433ull,0x0a202f2f09202020ull,0x3233732e766f6d09ull,0x2c33343272250920ull,
0x203b363231722520ull,0x0920202020202020ull,0x6c756d090a202f2full,0x203233732e6f6c2eull,
0x202c343432722509ull,0x25202c3334327225ull,0x0a202f2f093b3172ull,0x3233732e64646109ull,
0x2c35343272250920ull,0x202c323432722520ull,0x09203b3434327225ull,0x747663090a202f2full,
0x3233732e3436752eull,0x2c36396472250920ull,0x203b353432722520ull,0x0a202f2f09202020ull,
0x3436752e64646109ull,0x2c37396472250920ull,0x202c363964722520ull,0x0920203b34647225ull,
0x2e7473090a202f2full,0x732e6c61626f6c67ull,0x396472255b092038ull,0x7225202c5d302b37ull,
0x202f2f093b313432ull,0x090a3536363a6469ull,0x09373109636f6c2eull,0x6f6d090a30093535ull,
0x2509203233732e76ull,0x7225202c36343272ull,0x202020203b363431ull,0x202f2f0920202020ull,
0x33732e627573090aull,0x3734327225092032ull,0x2c3634327225202cull,0x20202020203b3120ull,
0x6964090a202f2f09ull,0x2509203233732e76ull,0x7225202c38343272ull,0x317225202c303231ull,
0x202f2f092020203bull,0x6f6c2e6c756d090aull,0x722509203233732eull,0x317225202c393432ull,
0x3b3834327225202cull,0x6572090a202f2f09ull,0x2509203233732e6dull,0x7225202c30353272ull,
0x317225202c303231ull,0x202f2f092020203bull,0x33732e646461090aull,0x3135327225092032ull,
0x2c3934327225202cull,0x203b303532722520ull,0x7663090a202f2f09ull,0x33732e3436752e74ull,
0x3839647225092032ull,0x3b3135327225202cull,0x202f2f0920202020ull,0x36752e646461090aull,
0x3939647225092034ull,0x2c3839647225202cull,0x20203b3464722520ull,0x7473090a202f2f09ull,
0x2e6c61626f6c672eull,0x6472255b09203873ull,0x25202c5d302b3939ull,0x2f2f093b37343272ull,
0x0a3636363a646920ull,0x31365f305f744c24ull,0x2f2f200a3a383936ull,0x50203e706f6f6c3cull,
0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,
0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,0x0937350937310963ull,
0x732e726162090a30ull,0x203b300920636e79ull,0x2020202020202020ull,0x2020202020202020ull,
0x40090a202f2f0920ull,0x7262203831702521ull,0x305f744c24092061ull,0x203b30313232365full,
0x2f2f092020202020ull,0x6f6c3c2f2f200a20ull,0x74726150203e706full,0x706f6f6c20666f20ull,
0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,
0x38333732355f305full,0x3109636f6c2e090aull,0x090a300930360937ull,0x203233732e766f6dull,
0x202c323532722509ull,0x20203b3433317225ull,0x2f09202020202020ull,0x2e766f6d090a202full,
0x3272250920323373ull,0x33317225202c3335ull,0x2020202020203b38ull,0x090a202f2f092020ull,
0x732e6f6c2e6c756dull,0x3532722509203233ull,0x3335327225202c34ull,0x2f093b317225202cull,
0x2e646461090a202full,0x3272250920323373ull,0x35327225202c3535ull,0x3435327225202c32ull,
0x090a202f2f09203bull,0x2e3436752e747663ull,0x6472250920323373ull,0x327225202c303031ull,
0x2f092020203b3535ull,0x2e646461090a202full,0x6472250920343675ull,0x31647225202c3736ull,
0x34647225202c3030ull,0x090a202f2f09203bull,0x61626f6c672e646cull,0x7225092038732e6cull,
0x6472255b202c3568ull,0x2f093b5d302b3736ull,0x3736363a6469202full,0x31732e646461090aull,
0x2c36687225092036ull,0x31202c3568722520ull,0x202020202020203bull,0x7473090a202f2f09ull,
0x2e6c61626f6c672eull,0x6472255b09203873ull,0x25202c5d302b3736ull,0x202f2f093b366872ull,
0x090a3836363a6469ull,0x09373109636f6c2eull,0x6572090a30093136ull,0x2509203233732e6dull,
0x7225202c38393172ull,0x317225202c303231ull,0x202f2f092020203bull,0x33732e627573090aull,
0x3339317225092032ull,0x31202c317225202cull,0x202020202020203bull,0x6964090a202f2f09ull,
0x2509203233732e76ull,0x7225202c36393172ull,0x317225202c303231ull,0x202f2f092020203bull,
0x6e2e70746573090aull,0x2509203233732e65ull,0x317225202c323370ull,0x39317225202c3839ull,
0x090a202f2f093b33ull,0x732e6f6c2e6c756dull,0x3532722509203233ull,0x3639317225202c36ull,
0x2f093b317225202cull,0x2e646461090a202full,0x3272250920323373ull,0x35327225202c3131ull,
0x3839317225202c36ull,0x090a202f2f09203bull,0x2e3436732e747663ull,0x6472250920323373ull,
0x327225202c313031ull,0x2f092020203b3131ull,0x2e646461090a202full,0x6472250920343673ull,
0x647225202c323031ull,0x203b31202c313031ull,0x090a202f2f092020ull,0x2e3436732e747663ull,
0x6472250920323373ull,0x327225202c333031ull,0x2f092020203b3635ull,0x706c6573090a202full,
0x722509203436752eull,0x7225202c34303164ull,0x7225202c32303164ull,0x7025202c33303164ull,
0x0a202f2f093b3233ull,0x3436752e64646109ull,0x2c32376472250920ull,0x2c34303164722520ull,
0x09203b3464722520ull,0x2e646c090a202f2full,0x732e6c61626f6c67ull,0x2c37687225092038ull,
0x2b32376472255b20ull,0x69202f2f093b5d30ull,0x61090a3936363a64ull,0x09203631732e6464ull,
0x7225202c38687225ull,0x20203b31202c3768ull,0x2f2f092020202020ull,0x6c672e7473090a20ull,
0x2038732e6c61626full,0x2b32376472255b09ull,0x38687225202c5d30ull,0x3a6469202f2f093bull,
0x5f744c240a303736ull,0x3a30313232365f30ull,0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,
0x20706f6f6c20666full,0x6e696c2079646f62ull,0x68202c3031322065ull,0x6562616c20646165ull,
0x5f744c242064656cull,0x0a38333732355f30ull,0x373109636f6c2e09ull,0x62090a3009333609ull,
0x20636e79732e7261ull,0x20202020203b3009ull,0x2020202020202020ull,0x2f2f092020202020ull,
0x09636f6c2e090a20ull,0x3009363031093731ull,0x303170252140090aull,0x4c24092061726220ull,
0x323732365f305f74ull,0x2020202020203b32ull,0x2f2f200a202f2f09ull,0x50203e706f6f6c3cull,
0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,
0x2064656c6562616cull,0x32355f305f744c24ull,0x6f6c2e090a383337ull,0x3231310937310963ull,
0x732e646c090a3009ull,0x33752e6465726168ull,0x3735327225092032ull,0x35326472255b202cull,
0x093b5d383430322bull,0x37363a6469202f2full,0x3078302b746d2031ull,0x33752e6d6572090aull,
0x3032317225092032ull,0x2c3735327225202cull,0x20203b3034722520ull,0x744c240a202f2f09ull,
0x32323732365f305full,0x6f6c3c2f2f200a3aull,0x74726150203e706full,0x706f6f6c20666f20ull,
0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,
0x38333732355f305full,0x3109636f6c2e090aull,0x0a30093431310937ull,0x2031317025214009ull,
0x744c240920617262ull,0x30393433365f305full,0x092020202020203bull,0x3c2f2f200a202f2full,
0x6150203e706f6f6cull,0x6f6c20666f207472ull,0x2079646f6220706full,0x30313220656e696cull,
0x6c2064616568202cull,0x242064656c656261ull,0x3732355f305f744cull,0x636f6c2e090a3833ull,
0x3009383209373109ull,0x33732e766964090aull,0x3835327225092032ull,0x2c3032317225202cull,
0x2020203b31722520ull,0x7573090a202f2f09ull,0x2509203233732e62ull,0x7225202c32323172ull,
0x20203b31202c3933ull,0x202f2f0920202020ull,0x33732e766f6d090aull,0x3935327225092032ull,
0x202020203b30202cull,0x2020202020202020ull,0x6573090a202f2f09ull,0x33732e71652e7074ull,
0x2c33317025092032ull,0x202c383532722520ull,0x2f093b3935327225ull,0x2e627573090a202full,
0x3172250920323373ull,0x35327225202c3432ull,0x2020203b31202c38ull,0x090a202f2f092020ull,
0x3233732e706c6573ull,0x2c35323172250920ull,0x202c323231722520ull,0x25202c3432317225ull,
0x202f2f093b333170ull,0x33732e766f6d090aull,0x3632317225092032ull,0x3b3532317225202cull,
0x2020202020202020ull,0x6572090a202f2f09ull,0x2509203233732e6dull,0x7225202c30363272ull,
0x317225202c303231ull,0x202f2f092020203bull,0x33622e646e61090aull,0x3136327225092032ull,
0x2c3835327225202cull,0x20202020203b3120ull,0x6f6d090a202f2f09ull,0x2509203233752e76ull,
0x3b30202c32363272ull,0x2020202020202020ull,0x202f2f0920202020ull,0x652e70746573090aull,
0x2509203233732e71ull,0x327225202c333370ull,0x36327225202c3136ull,0x090a202f2f093b32ull,
0x7262203333702540ull,0x305f744c24092061ull,0x203b32303034365full,0x2f09202020202020ull,
0x6c3c2f2f200a202full,0x726150203e706f6full,0x6f6f6c20666f2074ull,0x6c2079646f622070ull,
0x2c30313220656e69ull,0x616c206461656820ull,0x4c242064656c6562ull,0x333732355f305f74ull,
0x09636f6c2e090a38ull,0x0a30093932093731ull,0x3233732e64646109ull,0x2c33363272250920ull,
0x202c303632722520ull,0x0920202020203b31ull,0x766f6d090a202f2full,0x722509203233732eull,
0x203b30202c343632ull,0x2020202020202020ull,0x0a202f2f09202020ull,0x3233732e62757309ull,
0x2c35363272250920ull,0x3b31202c31722520ull,0x0920202020202020ull,0x746573090a202f2full,
0x3233732e656e2e70ull,0x202c343370250920ull,0x25202c3036327225ull,0x2f2f093b35363272ull,
0x2e706c6573090a20ull,0x3272250920323373ull,0x36327225202c3636ull,0x3436327225202c33ull,
0x093b34337025202cull,0x617262090a202f2full,0x4c240920696e752eull,0x343733365f305f74ull,
0x2020202020203b36ull,0x0a202f2f09202020ull,0x34365f305f744c24ull,0x2f2f200a3a323030ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x766f6d090a383337ull,
0x722509203233732eull,0x327225202c363632ull,0x20202020203b3036ull,0x0a202f2f09202020ull,
0x33365f305f744c24ull,0x2f2f200a3a363437ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,
0x79646f6220706f6full,0x313220656e696c20ull,0x2064616568202c30ull,0x2064656c6562616cull,
0x32355f305f744c24ull,0x766f6d090a383337ull,0x722509203233732eull,0x327225202c343331ull,
0x20202020203b3636ull,0x0a202f2f09202020ull,0x373109636f6c2e09ull,0x61090a3009303309ull,
0x09203233732e6464ull,0x25202c3736327225ull,0x3b31202c38353272ull,0x2f2f092020202020ull,
0x732e766f6d090a20ull,0x3632722509203233ull,0x2020203b30202c38ull,0x2020202020202020ull,
0x73090a202f2f0920ull,0x732e656e2e707465ull,0x3533702509203233ull,0x2c3835327225202cull,
0x093b323231722520ull,0x6c6573090a202f2full,0x2509203233732e70ull,0x7225202c39363272ull,
0x327225202c373632ull,0x35337025202c3836ull,0x6d090a202f2f093bull,0x09203233732e766full,
0x25202c3833317225ull,0x2020203b39363272ull,0x2f2f092020202020ull,0x09636f6c2e090a20ull,
0x0a30093133093731ull,0x2e6f6c2e6c756d09ull,0x3272250920323373ull,0x32317225202c3037ull,
0x093b317225202c35ull,0x646461090a202f2full,0x722509203233732eull,0x327225202c313732ull,
0x37327225202c3636ull,0x0a202f2f09203b30ull,0x3436752e74766309ull,0x722509203233732eull,
0x7225202c35303164ull,0x092020203b313732ull,0x646461090a202f2full,0x722509203436752eull,
0x7225202c36303164ull,0x7225202c35303164ull,0x0a202f2f093b3464ull,0x626f6c672e646c09ull,
0x25092038732e6c61ull,0x255b202c31343172ull,0x5d302b3630316472ull,0x3a6469202f2f093bull,
0x766f6d090a323736ull,0x722509203233732eull,0x317225202c323431ull,0x20202020203b3134ull,
0x0a202f2f09202020ull,0x373109636f6c2e09ull,0x6d090a3009323309ull,0x33732e6f6c2e6c75ull,
0x3237327225092032ull,0x2c3835327225202cull,0x2f2f093b31722520ull,0x732e646461090a20ull,
0x3732722509203233ull,0x3036327225202c33ull,0x3b3237327225202cull,0x63090a202f2f0920ull,
0x732e3436752e7476ull,0x3164722509203233ull,0x37327225202c3730ull,0x2f2f092020203b33ull,
0x752e646461090a20ull,0x3164722509203436ull,0x31647225202c3830ull,0x34647225202c3730ull,
0x6c090a202f2f093bull,0x6c61626f6c672e64ull,0x317225092038732eull,0x6472255b202c3534ull,
0x093b5d302b383031ull,0x37363a6469202f2full,0x732e766f6d090a33ull,0x3431722509203233ull,
0x3534317225202c36ull,0x202020202020203bull,0x6d090a202f2f0920ull,0x09203233732e766full,
0x30202c3437327225ull,0x202020202020203bull,0x2f2f092020202020ull,0x672e746573090a20ull,
0x33732e3233752e74ull,0x3537327225092032ull,0x2c3534317225202cull,0x093b343732722520ull,
0x67656e090a202f2full,0x722509203233732eull,0x327225202c363732ull,0x20202020203b3537ull,
0x0a202f2f09202020ull,0x3233732e766f6d09ull,0x2c37373272250920ull,0x20202020203b3020ull,
0x0920202020202020ull,0x746573090a202f2full,0x2e3233752e74672eull,0x3272250920323373ull,
0x34317225202c3837ull,0x3737327225202c31ull,0x6e090a202f2f093bull,0x09203233732e6765ull,
0x25202c3937327225ull,0x2020203b38373272ull,0x2f2f092020202020ull,0x622e646e61090a20ull,
0x3832722509203233ull,0x3637327225202c30ull,0x3b3937327225202cull,0x6d090a202f2f0920ull,
0x09203233752e766full,0x30202c3138327225ull,0x202020202020203bull,0x2f2f092020202020ull,
0x2e70746573090a20ull,0x09203233732e7165ull,0x7225202c36337025ull,0x327225202c303832ull,
0x0a202f2f093b3138ull,0x6220363370254009ull,0x5f744c2409206172ull,0x3b34313534365f30ull,
0x0920202020202020ull,0x3c2f2f200a202f2full,0x6150203e706f6f6cull,0x6f6c20666f207472ull,
0x2079646f6220706full,0x30313220656e696cull,0x6c2064616568202cull,0x242064656c656261ull,
0x3732355f305f744cull,0x636f6c2e090a3833ull,0x3009363309373109ull,0x33732e646461090aull,
0x3238327225092032ull,0x202c37357225202cull,0x20202020203b3331ull,0x6873090a202f2f09ull,
0x2509203233732e72ull,0x7225202c33383272ull,0x20203b32202c3034ull,0x202f2f0920202020ull,
0x36732e747663090aull,0x2509203233732e34ull,0x25202c3930316472ull,0x2020203b33383272ull,
0x756d090a202f2f09ull,0x3436752e6f6c2e6cull,0x3031316472250920ull,0x393031647225202cull,
0x202f2f093b34202cull,0x36752e646461090aull,0x3131647225092034ull,0x2c34647225202c31ull,
0x3b30313164722520ull,0x7663090a202f2f09ull,0x33732e3436732e74ull,0x3131647225092032ull,
0x3337327225202c32ull,0x202f2f092020203bull,0x6f6c2e6c756d090aull,0x722509203436752eull,
0x7225202c33313164ull,0x3b34202c32313164ull,0x6461090a202f2f09ull,0x2509203436752e64ull,
0x25202c3431316472ull,0x25202c3131316472ull,0x2f093b3331316472ull,0x672e7473090a202full,
0x33732e6c61626f6cull,0x316472255b092032ull,0x25202c5d302b3431ull,0x2f2f093b32383272ull,
0x0a3437363a646920ull,0x3233732e766f6d09ull,0x2c37353172250920ull,0x20202020203b3120ull,
0x0920202020202020ull,0x617262090a202f2full,0x4c240920696e752eull,0x333233365f305f74ull,
0x2020202020203b34ull,0x0a202f2f09202020ull,0x34365f305f744c24ull,0x2f2f200a3a343135ull,
0x50203e706f6f6c3cull,0x6c20666f20747261ull,0x79646f6220706f6full,0x313220656e696c20ull,
0x2064616568202c30ull,0x2064656c6562616cull,0x32355f305f744c24ull,0x766f6d090a383337ull,
0x722509203233732eull,0x203b30202c373531ull,0x2020202020202020ull,0x0a202f2f09202020ull,
0x696e752e61726209ull,0x5f305f744c240920ull,0x20203b3433323336ull,0x0920202020202020ull,
0x5f744c240a202f2full,0x3a30393433365f30ull,0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,
0x20706f6f6c20666full,0x6e696c2079646f62ull,0x68202c3031322065ull,0x6562616c20646165ull,
0x5f744c242064656cull,0x0a38333732355f30ull,0x3233732e766f6d09ull,0x2c37353172250920ull,
0x20202020203b3020ull,0x0920202020202020ull,0x5f744c240a202f2full,0x3a34333233365f30ull,
0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,0x20706f6f6c20666full,0x6e696c2079646f62ull,
0x68202c3031322065ull,0x6562616c20646165ull,0x5f744c242064656cull,0x0a38333732355f30ull,
0x373109636f6c2e09ull,0x62090a3009393309ull,0x20636e79732e7261ull,0x20202020203b3009ull,
0x2020202020202020ull,0x2f2f092020202020ull,0x732e766f6d090a20ull,0x3832722509203233ull,
0x2020203b30202c34ull,0x2020202020202020ull,0x73090a202f2f0920ull,0x732e656e2e707465ull,
0x3831702509203233ull,0x2c3735317225202cull,0x093b343832722520ull,0x252140090a202f2full,
0x2061726220383170ull,0x365f305f744c2409ull,0x2020203b30373734ull,0x0a202f2f09202020ull,
0x706f6f6c3c2f2f20ull,0x6f2074726150203eull,0x6220706f6f6c2066ull,0x656e696c2079646full,
0x6568202c30313220ull,0x6c6562616c206461ull,0x305f744c24206465ull,0x090a38333732355full,
0x09373109636f6c2eull,0x6461090a30093534ull,0x2509203233732e64ull,0x7225202c35383272ull,
0x203b3331202c3735ull,0x202f2f0920202020ull,0x33622e646e61090aull,0x3638327225092032ull,
0x2c3538327225202cull,0x20202020203b3120ull,0x6964090a202f2f09ull,0x2509203233732e76ull,
0x7225202c37383272ull,0x317225202c303231ull,0x202f2f092020203bull,0x33622e646e61090aull,
0x3838327225092032ull,0x2c3738327225202cull,0x20202020203b3120ull,0x6573090a202f2f09ull,
0x33732e71652e7074ull,0x2c37337025092032ull,0x202c363832722520ull,0x2f093b3838327225ull,
0x33702540090a202full,0x2409206172622037ull,0x3136325f305f744cull,0x20202020203b3431ull,
0x200a202f2f092020ull,0x3e706f6f6c3c2f2full,0x666f207472615020ull,0x6f6220706f6f6c20ull,
0x20656e696c207964ull,0x616568202c303132ull,0x656c6562616c2064ull,0x5f305f744c242064ull,
0x73090a3833373235ull,0x09203233732e7268ull,0x25202c3938327225ull,0x203b32202c303472ull,
0x2f2f092020202020ull,0x732e747663090a20ull,0x09203233732e3436ull,0x202c353131647225ull,
0x20203b3938327225ull,0x6d090a202f2f0920ull,0x36752e6f6c2e6c75ull,0x3131647225092034ull,
0x3131647225202c36ull,0x2f2f093b34202c35ull,0x752e646461090a20ull,0x3564722509203436ull,
0x2c34647225202c30ull,0x3b36313164722520ull,0x6d090a202f2f0920ull,0x09203233732e766full,
0x25202c3436317225ull,0x2020203b34333172ull,0x2f2f092020202020ull,0x732e766f6d090a20ull,
0x3932722509203233ull,0x3632317225202c30ull,0x202020202020203bull,0x6d090a202f2f0920ull,
0x33732e6f6c2e6c75ull,0x3139327225092032ull,0x2c3039327225202cull,0x2f2f093b31722520ull,
0x732e646461090a20ull,0x3932722509203233ull,0x3436317225202c32ull,0x3b3139327225202cull,
0x63090a202f2f0920ull,0x732e3436732e7476ull,0x3164722509203233ull,0x39327225202c3731ull,
0x2f2f092020203b32ull,0x6c2e6c756d090a20ull,0x2509203436752e6full,0x25202c3831316472ull,
0x34202c3731316472ull,0x61090a202f2f093bull,0x09203436752e6464ull,0x202c393131647225ull,
0x25202c3035647225ull,0x2f093b3831316472ull,0x672e646c090a202full,0x33732e6c61626f6cull,
0x3339327225092032ull,0x31316472255b202cull,0x2f2f093b5d302b39ull,0x0a3537363a646920ull,
0x71652e7074657309ull,0x702509203233732eull,0x39327225202c3833ull,0x3538327225202c33ull,
0x40090a202f2f093bull,0x6172622038337025ull,0x5f305f744c240920ull,0x20203b3236393237ull,
0x2f2f092020202020ull,0x6f6c3c2f2f200a20ull,0x74726150203e706full,0x706f6f6c20666f20ull,
0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,
0x38333732355f305full,0x33732e766f6d090aull,0x3439327225092032ull,0x3b3833317225202cull,
0x2020202020202020ull,0x756d090a202f2f09ull,0x3233732e6f6c2e6cull,0x2c35393272250920ull,
0x202c343932722520ull,0x202f2f093b317225ull,0x33732e646461090aull,0x3639327225092032ull,
0x2c3539327225202cull,0x2020203b31722520ull,0x7663090a202f2f09ull,0x33732e3436732e74ull,
0x3231647225092032ull,0x3639327225202c30ull,0x202f2f092020203bull,0x6f6c2e6c756d090aull,
0x722509203436752eull,0x7225202c31323164ull,0x3b34202c30323164ull,0x7573090a202f2f09ull,
0x2509203436752e62ull,0x25202c3232316472ull,0x34202c3132316472ull,0x202f2f092020203bull,
0x33732e646461090aull,0x3739327225092032ull,0x2c3539327225202cull,0x203b343631722520ull,
0x7663090a202f2f09ull,0x33732e3436732e74ull,0x3231647225092032ull,0x3739327225202c33ull,
0x202f2f092020203bull,0x6f6c2e6c756d090aull,0x722509203436752eull,0x7225202c34323164ull,
0x3b34202c33323164ull,0x7573090a202f2f09ull,0x2509203436752e62ull,0x25202c3532316472ull,
0x34202c3432316472ull,0x202f2f092020203bull,0x33732e766f6d090aull,0x3839327225092032ull,
0x202020203b30202cull,0x2020202020202020ull,0x6573090a202f2f09ull,0x33732e71652e7074ull,
0x2c39337025092032ull,0x202c343631722520ull,0x2f093b3839327225ull,0x706c6573090a202full,
0x722509203436752eull,0x7225202c36323164ull,0x7225202c32323164ull,0x7025202c35323164ull,
0x0a202f2f093b3933ull,0x3436752e64646109ull,0x3732316472250920ull,0x363231647225202cull,
0x3b3035647225202cull,0x646c090a202f2f09ull,0x2e6c61626f6c672eull,0x3272250920323373ull,
0x6472255b202c3939ull,0x093b5d302b373231ull,0x37363a6469202f2full,0x2e70746573090a36ull,
0x09203233732e7165ull,0x7225202c30347025ull,0x327225202c393932ull,0x0a202f2f093b3538ull,
0x6220303470254009ull,0x5f744c2409206172ull,0x3b32363932375f30ull,0x0920202020202020ull,
0x5f744c240a202f2full,0x3a34313136325f30ull,0x6f6f6c3c2f2f200aull,0x2074726150203e70ull,
0x20706f6f6c20666full,0x6e696c2079646f62ull,0x68202c3031322065ull,0x6562616c20646165ull,
0x5f744c242064656cull,0x0a38333732355f30ull,0x3233732e766f6d09ull,0x2c30303372250920ull,
0x20202020203b3120ull,0x0920202020202020ull,0x617262090a202f2full,0x4c240920696e752eull,
0x353637325f305f74ull,0x2020202020203b30ull,0x0a202f2f09202020ull,0x32375f305f744c24ull,
0x744c240a3a323639ull,0x30373336325f305full,0x6f6c3c2f2f200a3aull,0x74726150203e706full,
0x706f6f6c20666f20ull,0x696c2079646f6220ull,0x202c30313220656eull,0x62616c2064616568ull,
0x744c242064656c65ull,0x38333732355f305full,0x33732e766f6d090aull,0x3030337225092032ull,
0x202020203b30202cull,0x2020202020202020ull,0x744c240a202f2f09ull,0x30353637325f305full,
0x6f6c3c2f2f200a3aull,0x74726150203e706full,0x706f6f6c20666f20ull,0x696c2079646f6220ull,
0x202c30313220656eull,0x62616c2064616568ull,0x744c242064656c65ull,0x38333732355f305full,
0x33732e766f6d090aull,0x3130337225092032ull,0x202020203b30202cull,0x2020202020202020ull,
0x6573090a202f2f09ull,0x33732e656e2e7074ull,0x2c38317025092032ull,0x202c303033722520ull,
0x2f093b3130337225ull,0x305f744c240a202full,0x0a3a30373734365full,0x706f6f6c3c2f2f20ull,
0x6f2074726150203eull,0x6220706f6f6c2066ull,0x656e696c2079646full,0x6568202c30313220ull,
0x6c6562616c206461ull,0x305f744c24206465ull,0x090a38333732355full,0x6220383170252140ull,
0x5f744c2409206172ull,0x3b32383235365f30ull,0x2f09202020202020ull,0x6c3c2f2f200a202full,
0x726150203e706f6full,0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,
0x616c206461656820ull,0x4c242064656c6562ull,0x333732355f305f74ull,0x09636f6c2e090a38ull,
0x0a30093435093731ull,0x3233732e766f6d09ull,0x2c32303372250920ull,0x203b323431722520ull,
0x0920202020202020ull,0x627573090a202f2full,0x722509203233732eull,0x337225202c333033ull,
0x20203b31202c3230ull,0x0a202f2f09202020ull,0x3233732e766f6d09ull,0x2c34303372250920ull,
0x203b343331722520ull,0x0920202020202020ull,0x766f6d090a202f2full,0x722509203233732eull,
0x317225202c353033ull,0x20202020203b3632ull,0x0a202f2f09202020ull,0x2e6f6c2e6c756d09ull,
0x3372250920323373ull,0x30337225202c3630ull,0x093b317225202c35ull,0x646461090a202f2full,
0x722509203233732eull,0x337225202c373033ull,0x30337225202c3430ull,0x0a202f2f09203b36ull,
0x3436752e74766309ull,0x722509203233732eull,0x7225202c38323164ull,0x092020203b373033ull,
0x646461090a202f2full,0x722509203436752eull,0x7225202c39323164ull,0x7225202c38323164ull,
0x0a202f2f093b3464ull,0x626f6c672e747309ull,0x5b092038732e6c61ull,0x302b393231647225ull,
0x3330337225202c5dull,0x3a6469202f2f093bull,0x6f6c2e090a373736ull,0x0935350937310963ull,
0x732e766f6d090a30ull,0x3033722509203233ull,0x3634317225202c38ull,0x202020202020203bull,
0x73090a202f2f0920ull,0x09203233732e6275ull,0x25202c3930337225ull,0x3b31202c38303372ull,
0x2f2f092020202020ull,0x732e766964090a20ull,0x3133722509203233ull,0x3032317225202c30ull,
0x20203b317225202cull,0x6d090a202f2f0920ull,0x33732e6f6c2e6c75ull,0x3131337225092032ull,
0x25202c317225202cull,0x2f2f093b30313372ull,0x732e6d6572090a20ull,0x3133722509203233ull,
0x3032317225202c32ull,0x20203b317225202cull,0x61090a202f2f0920ull,0x09203233732e6464ull,
0x25202c3331337225ull,0x7225202c31313372ull,0x2f2f09203b323133ull,0x752e747663090a20ull,
0x09203233732e3436ull,0x202c303331647225ull,0x20203b3331337225ull,0x61090a202f2f0920ull,
0x09203436752e6464ull,0x202c313331647225ull,0x202c303331647225ull,0x2f2f093b34647225ull,
0x6c672e7473090a20ull,0x2038732e6c61626full,0x3133316472255b09ull,0x337225202c5d302bull,
0x69202f2f093b3930ull,0x4c240a3837363a64ull,0x383235365f305f74ull,0x6c3c2f2f200a3a32ull,
0x726150203e706f6full,0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c30313220656e69ull,
0x616c206461656820ull,0x4c242064656c6562ull,0x333732355f305f74ull,0x09636f6c2e090a38ull,
0x0a30093735093731ull,0x6e79732e72616209ull,0x2020203b30092063ull,0x2020202020202020ull,
0x0920202020202020ull,0x252140090a202f2full,0x2061726220383170ull,0x365f305f744c2409ull,
0x2020203b34393735ull,0x0a202f2f09202020ull,0x706f6f6c3c2f2f20ull,0x6f2074726150203eull,
0x6220706f6f6c2066ull,0x656e696c2079646full,0x6568202c30313220ull,0x6c6562616c206461ull,
0x305f744c24206465ull,0x090a38333732355full,0x09373109636f6c2eull,0x6f6d090a30093036ull,
0x2509203233732e76ull,0x7225202c34313372ull,0x202020203b343331ull,0x202f2f0920202020ull,
0x33732e766f6d090aull,0x3531337225092032ull,0x3b3833317225202cull,0x2020202020202020ull,
0x756d090a202f2f09ull,0x3233732e6f6c2e6cull,0x2c36313372250920ull,0x202c353133722520ull,
0x202f2f093b317225ull,0x33732e646461090aull,0x3731337225092032ull,0x2c3431337225202cull,
0x203b363133722520ull,0x7663090a202f2f09ull,0x33732e3436752e74ull,0x3331647225092032ull,
0x3731337225202c32ull,0x202f2f092020203bull,0x36752e646461090aull,0x3736647225092034ull,
0x323331647225202cull,0x203b34647225202cull,0x646c090a202f2f09ull,0x2e6c61626f6c672eull,
0x3968722509203873ull,0x37366472255b202cull,0x202f2f093b5d302bull,0x090a3937363a6469ull,
0x203631732e646461ull,0x202c303168722509ull,0x3b31202c39687225ull,0x2f09202020202020ull,
0x672e7473090a202full,0x38732e6c61626f6cull,0x37366472255b0920ull,0x687225202c5d302bull,
0x69202f2f093b3031ull,0x2e090a3038363a64ull,0x3609373109636f6cull,0x6d6572090a300931ull,
0x722509203233732eull,0x317225202c303632ull,0x3b317225202c3032ull,0x0a202f2f09202020ull,
0x3233732e62757309ull,0x2c33393172250920ull,0x3b31202c31722520ull,0x0920202020202020ull,
0x766964090a202f2full,0x722509203233732eull,0x317225202c383532ull,0x3b317225202c3032ull,
0x0a202f2f09202020ull,0x656e2e7074657309ull,0x702509203233732eull,0x36327225202c3134ull,
0x3339317225202c30ull,0x6d090a202f2f093bull,0x33732e6f6c2e6c75ull,0x3831337225092032ull,
0x2c3835327225202cull,0x2f2f093b31722520ull,0x732e646461090a20ull,0x3732722509203233ull,
0x3831337225202c33ull,0x3b3036327225202cull,0x63090a202f2f0920ull,0x732e3436732e7476ull,
0x3164722509203233ull,0x37327225202c3333ull,0x2f2f092020203b33ull,0x732e646461090a20ull,
0x3164722509203436ull,0x31647225202c3433ull,0x20203b31202c3333ull,0x63090a202f2f0920ull,
0x732e3436732e7476ull,0x3164722509203233ull,0x31337225202c3533ull,0x2f2f092020203b38ull,
0x2e706c6573090a20ull,0x6472250920343675ull,0x647225202c363331ull,0x647225202c343331ull,
0x347025202c353331ull,0x090a202f2f093b31ull,0x203436752e646461ull,0x202c323764722509ull,
0x202c363331647225ull,0x2f09203b34647225ull,0x672e646c090a202full,0x38732e6c61626f6cull,
0x2c31316872250920ull,0x2b32376472255b20ull,0x69202f2f093b5d30ull,0x61090a3138363a64ull,
0x09203631732e6464ull,0x25202c3231687225ull,0x3b31202c31316872ull,0x2f2f092020202020ull,
0x6c672e7473090a20ull,0x2038732e6c61626full,0x2b32376472255b09ull,0x31687225202c5d30ull,
0x6469202f2f093b32ull,0x744c240a3238363aull,0x34393735365f305full,0x6f6c3c2f2f200a3aull,
0x74726150203e706full,0x706f6f6c20666f20ull,0x696c2079646f6220ull,0x202c30313220656eull,
0x62616c2064616568ull,0x744c242064656c65ull,0x38333732355f305full,0x3109636f6c2e090aull,
0x090a300933360937ull,0x636e79732e726162ull,0x202020203b300920ull,0x2020202020202020ull,
0x2f09202020202020ull,0x636f6c2e090a202full,0x0934313109373109ull,0x732e646461090a30ull,
0x3735722509203233ull,0x202c37357225202cull,0x202020203b323133ull,0x73090a202f2f0920ull,
0x732e74672e707465ull,0x3234702509203233ull,0x202c35337225202cull,0x2f2f093b37357225ull,
0x3234702540090a20ull,0x4c24092061726220ull,0x333732355f305f74ull,0x2020202020203b38ull,
0x62090a202f2f0920ull,0x0920696e752e6172ull,0x32355f305f744c24ull,0x202020203b363232ull,
0x2f2f092020202020ull,0x5f305f744c240a20ull,0x090a3a3431393037ull,0x2e3436752e747663ull,
0x6472250920323375ull,0x317225202c373331ull,0x2f09202020203b31ull,0x2e6c756d090a202full,
0x09203436752e6f6cull,0x202c383331647225ull,0x202c373331647225ull,0x090a202f2f093b34ull,
0x203436752e646461ull,0x202c353264722509ull,0x25202c3531647225ull,0x2f093b3833316472ull,
0x305f744c240a202full,0x0a3a36323232355full,0x373109636f6c2e09ull,0x090a300931313209ull,
0x203233732e627573ull,0x202c393133722509ull,0x203b31202c347225ull,0x2f09202020202020ull,
0x2e766f6d090a202full,0x3372250920323375ull,0x20203b30202c3032ull,0x2020202020202020ull,
0x090a202f2f092020ull,0x2e746c2e70746573ull,0x3470250920323373ull,0x3931337225202c33ull,
0x3b3032337225202cull,0x6f6d090a202f2f09ull,0x2509203233732e76ull,0x3b30202c31323372ull,
0x2020202020202020ull,0x202f2f0920202020ull,0x33662e766f6d090aull,0x202c316625092032ull,
0x3030303030306630ull,0x20202020203b3030ull,0x40090a30202f2f09ull,0x6172622033347025ull,
0x5f305f744c240920ull,0x20203b3437343337ull,0x2f2f092020202020ull,0x732e766f6d090a20ull,
0x3233722509203233ull,0x203b347225202c32ull,0x2020202020202020ull,0x61090a202f2f0920ull,
0x09203233732e6464ull,0x337225202c387225ull,0x20203b377225202cull,0x2f2f092020202020ull,
0x61702e646c090a20ull,0x203436752e6d6172ull,0x2c39333164722509ull,0x616475635f5f5b20ull,
0x315a5f5f6d726170ull,0x656e697274655034ull,0x506c656e72654b74ull,0x6969695f53665069ull,
0x2f093b5d735f675full,0x3332363a6469202full,0x70616475635f5f20ull,0x34315a5f5f6d7261ull,
0x74656e6972746550ull,0x69506c656e72654bull,0x5f6969695f536650ull,0x0a3078302b735f67ull,
0x3233752e74766309ull,0x722509203631752eull,0x746325202c333233ull,0x09203b782e646961ull,
0x6c756d090a202f2full,0x203233752e6f6c2eull,0x202c343233722509ull,0x337225202c387225ull,
0x0a202f2f093b3332ull,0x3436752e74766309ull,0x722509203233752eull,0x7225202c30343164ull,
0x092020203b343233ull,0x6c756d090a202f2full,0x203436752e6f6c2eull,0x2c31343164722509ull,
0x2c30343164722520ull,0x0a202f2f093b3420ull,0x3436752e64646109ull,0x202c346472250920ull,
0x202c393331647225ull,0x093b313431647225ull,0x766f6d090a202f2full,0x722509203233732eull,
0x203b30202c353233ull,0x2020202020202020ull,0x0a202f2f09202020ull,0x3233732e766f6d09ull,
0x2c36323372250920ull,0x203b323233722520ull,0x0920202020202020ull,0x5f744c240a202f2full,
0x3a34373037365f30ull,0x6f6f6c3c2f2f200aull,0x20706f6f4c203e70ull,0x6e696c2079646f62ull,
0x6e202c3131322065ull,0x6420676e69747365ull,0x2c31203a68747065ull,0x74616d6974736520ull,
0x6172657469206465ull,0x75203a736e6f6974ull,0x090a6e776f6e6b6eull,0x09373109636f6c2eull,
0x73090a3009353331ull,0x09203233622e6c68ull,0x25202c3732337225ull,0x3b38202c35323372ull,
0x2f2f092020202020ull,0x752e646461090a20ull,0x3233722509203233ull,0x2c31317225202c38ull,
0x203b373233722520ull,0x63090a202f2f0920ull,0x752e3436752e7476ull,0x3164722509203233ull,
0x32337225202c3234ull,0x2f2f092020203b38ull,0x6c2e6c756d090a20ull,0x2509203436752e6full,
0x25202c3334316472ull,0x34202c3234316472ull,0x61090a202f2f093bull,0x09203436752e6464ull,
0x202c343431647225ull,0x7225202c34647225ull,0x2f2f093b33343164ull,0x6c672e646c090a20ull,
0x3233732e6c61626full,0x2c39323372250920ull,0x3434316472255b20ull,0x202f2f093b5d302bull,
0x090a3338363a6469ull,0x09373109636f6c2eull,0x61090a3009383331ull,0x09203233622e646eull,
0x25202c3033337225ull,0x3532202c39323372ull,0x2f2f092020203b35ull,0x6c2e6c756d090a20ull,
0x2509203233732e6full,0x7225202c31333372ull,0x337225202c303333ull,0x0a202f2f093b3033ull,
0x2e6e722e74766309ull,0x203233732e323366ull,0x7225202c32662509ull,0x092020203b313333ull,
0x646461090a202f2full,0x662509203233662eull,0x202c326625202c31ull,0x202020203b316625ull,
0x0a202f2f09202020ull,0x373109636f6c2e09ull,0x090a300939333109ull,0x203233732e78616dull,
0x202c313233722509ull,0x25202c3033337225ull,0x2f09203b31323372ull,0x636f6c2e090a202full,
0x0931343109373109ull,0x732e726873090a30ull,0x3333722509203233ull,0x3932337225202c32ull,
0x202020203b38202cull,0x61090a202f2f0920ull,0x09203233622e646eull,0x25202c3333337225ull,
0x3532202c32333372ull,0x2f2f092020203b35ull,0x6c2e6c756d090a20ull,0x2509203233732e6full,
0x7225202c34333372ull,0x337225202c333333ull,0x0a202f2f093b3333ull,0x2e6e722e74766309ull,
0x203233732e323366ull,0x7225202c33662509ull,0x092020203b343333ull,0x646461090a202f2full,
0x662509203233662eull,0x202c336625202c31ull,0x202020203b316625ull,0x0a202f2f09202020ull,
0x373109636f6c2e09ull,0x090a300932343109ull,0x203233732e78616dull,0x202c313233722509ull,
0x25202c3333337225ull,0x2f09203b31323372ull,0x636f6c2e090a202full,0x0934343109373109ull,
0x732e726873090a30ull,0x3333722509203233ull,0x3932337225202c35ull,0x2020203b3631202cull,
0x61090a202f2f0920ull,0x09203233622e646eull,0x25202c3633337225ull,0x3532202c35333372ull,
0x2f2f092020203b35ull,0x6c2e6c756d090a20ull,0x2509203233732e6full,0x7225202c37333372ull,
0x337225202c363333ull,0x0a202f2f093b3633ull,0x2e6e722e74766309ull,0x203233732e323366ull,
0x7225202c34662509ull,0x092020203b373333ull,0x646461090a202f2full,0x662509203233662eull,
0x202c346625202c31ull,0x202020203b316625ull,0x0a202f2f09202020ull,0x373109636f6c2e09ull,
0x090a300935343109ull,0x203233732e78616dull,0x202c313233722509ull,0x25202c3633337225ull,
0x2f09203b31323372ull,0x636f6c2e090a202full,0x0937343109373109ull,0x732e726873090a30ull,
0x3333722509203233ull,0x3932337225202c38ull,0x2020203b3432202cull,0x61090a202f2f0920ull,
0x09203233622e646eull,0x25202c3933337225ull,0x3532202c38333372ull,0x2f2f092020203b35ull,
0x6c2e6c756d090a20ull,0x2509203233732e6full,0x7225202c30343372ull,0x337225202c393333ull,
0x0a202f2f093b3933ull,0x2e6e722e74766309ull,0x203233732e323366ull,0x7225202c35662509ull,
0x092020203b303433ull,0x646461090a202f2full,0x662509203233662eull,0x202c356625202c31ull,
0x202020203b316625ull,0x0a202f2f09202020ull,0x373109636f6c2e09ull,0x090a300938343109ull,
0x203233732e78616dull,0x202c313233722509ull,0x25202c3933337225ull,0x2f09203b31323372ull,
0x2e646461090a202full,0x3372250920323373ull,0x32337225202c3532ull,0x2020203b31202c35ull,
0x090a202f2f092020ull,0x2e656e2e70746573ull,0x3470250920323373ull,0x202c347225202c34ull,
0x2f093b3532337225ull,0x34702540090a202full,0x2409206172622034ull,0x3037365f305f744cull,
0x20202020203b3437ull,0x090a202f2f092020ull,0x20696e752e617262ull,0x365f305f744c2409ull,
0x2020203b32363536ull,0x2f09202020202020ull,0x305f744c240a202full,0x0a3a34373433375full,
0x36365f305f744c24ull,0x6e61090a3a323635ull,0x2509203233622e64ull,0x7225202c31343372ull,
0x203b353532202c37ull,0x202f2f0920202020ull,0x33752e627573090aull,0x3234337225092032ull,
0x2c3134337225202cull,0x20202020203b3120ull,0x6573090a202f2f09ull,0x33752e74672e7074ull,
0x2c35347025092032ull,0x25202c3131722520ull,0x2f2f093b32343372ull,0x3534702540090a20ull,
0x4c24092061726220ull,0x383537365f305f74ull,0x2020202020203b36ull,0x2e090a202f2f0920ull,
0x3109373109636f6cull,0x646c090a30093735ull,0x752e6d617261702eull,0x3164722509203436ull,
0x635f5f5b202c3534ull,0x5f6d726170616475ull,0x7274655034315a5full,0x6e72654b74656e69ull,
0x5f53665069506c65ull,0x5d735f675f696969ull,0x3a6469202f2f093bull,0x75635f5f20333236ull,
0x5f5f6d7261706164ull,0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,
0x302b735f675f6969ull,0x2e747663090a3078ull,0x203631752e323375ull,0x202c333433722509ull,
0x782e646961746325ull,0x090a202f2f09203bull,0x203233732e646461ull,0x202c343433722509ull,
0x377225202c337225ull,0x2f0920202020203bull,0x2e6c756d090a202full,0x09203233752e6f6cull,
0x25202c3534337225ull,0x7225202c33343372ull,0x202f2f093b343433ull,0x36752e747663090aull,
0x2509203233752e34ull,0x25202c3634316472ull,0x2020203b35343372ull,0x756d090a202f2f09ull,
0x3436752e6f6c2e6cull,0x3734316472250920ull,0x363431647225202cull,0x202f2f093b34202cull,
0x36752e646461090aull,0x3431647225092034ull,0x3431647225202c38ull,0x3431647225202c35ull,
0x090a202f2f093b37ull,0x732e6f6c2e6c756dull,0x3433722509203233ull,0x202c347225202c36ull,
0x2f0920203b363532ull,0x2e646461090a202full,0x3372250920323375ull,0x31317225202c3734ull,
0x3b3634337225202cull,0x090a202f2f092020ull,0x2e3436752e747663ull,0x6472250920323375ull,
0x337225202c393431ull,0x2f092020203b3734ull,0x2e6c756d090a202full,0x09203436752e6f6cull,
0x202c303531647225ull,0x202c393431647225ull,0x090a202f2f093b34ull,0x203436752e646461ull,
0x2c31353164722509ull,0x2c38343164722520ull,0x3b30353164722520ull,0x646c090a202f2f09ull,
0x2e6c61626f6c672eull,0x3372250920323373ull,0x6472255b202c3834ull,0x093b5d302b313531ull,
0x38363a6469202f2full,0x09636f6c2e090a34ull,0x3009303631093731ull,0x33622e646e61090aull,
0x3934337225092032ull,0x2c3834337225202cull,0x2020203b35353220ull,0x756d090a202f2f09ull,
0x3233732e6f6c2e6cull,0x2c30353372250920ull,0x202c393433722520ull,0x2f093b3934337225ull,
0x2e747663090a202full,0x732e3233662e6e72ull,0x2c36662509203233ull,0x203b303533722520ull,
0x090a202f2f092020ull,0x203233662e646461ull,0x6625202c31662509ull,0x203b316625202c36ull,
0x2f09202020202020ull,0x636f6c2e090a202full,0x0931363109373109ull,0x732e78616d090a30ull,
0x3233722509203233ull,0x3934337225202c31ull,0x3b3132337225202cull,0x2e090a202f2f0920ull,
0x3109373109636f6cull,0x6873090a30093336ull,0x2509203233732e72ull,0x7225202c31353372ull,
0x203b38202c383433ull,0x202f2f0920202020ull,0x33622e646e61090aull,0x3235337225092032ull,
0x2c3135337225202cull,0x2020203b35353220ull,0x756d090a202f2f09ull,0x3233732e6f6c2e6cull,
0x2c33353372250920ull,0x202c323533722520ull,0x2f093b3235337225ull,0x2e747663090a202full,
0x732e3233662e6e72ull,0x2c37662509203233ull,0x203b333533722520ull,0x090a202f2f092020ull,
0x203233662e646461ull,0x6625202c31662509ull,0x203b316625202c37ull,0x2f09202020202020ull,
0x636f6c2e090a202full,0x0934363109373109ull,0x732e78616d090a30ull,0x3233722509203233ull,
0x3235337225202c31ull,0x3b3132337225202cull,0x2e090a202f2f0920ull,0x3109373109636f6cull,
0x6873090a30093636ull,0x2509203233732e72ull,0x7225202c34353372ull,0x3b3631202c383433ull,
0x202f2f0920202020ull,0x33622e646e61090aull,0x3535337225092032ull,0x2c3435337225202cull,
0x2020203b35353220ull,0x756d090a202f2f09ull,0x3233732e6f6c2e6cull,0x2c36353372250920ull,
0x202c353533722520ull,0x2f093b3535337225ull,0x2e747663090a202full,0x732e3233662e6e72ull,
0x2c38662509203233ull,0x203b363533722520ull,0x090a202f2f092020ull,0x203233662e646461ull,
0x6625202c31662509ull,0x203b316625202c38ull,0x2f09202020202020ull,0x636f6c2e090a202full,
0x0937363109373109ull,0x732e78616d090a30ull,0x3233722509203233ull,0x3535337225202c31ull,
0x3b3132337225202cull,0x2e090a202f2f0920ull,0x3109373109636f6cull,0x6873090a30093936ull,
0x2509203233732e72ull,0x7225202c37353372ull,0x3b3432202c383433ull,0x202f2f0920202020ull,
0x33622e646e61090aull,0x3835337225092032ull,0x2c3735337225202cull,0x2020203b35353220ull,
0x756d090a202f2f09ull,0x3233732e6f6c2e6cull,0x2c39353372250920ull,0x202c383533722520ull,
0x2f093b3835337225ull,0x2e747663090a202full,0x732e3233662e6e72ull,0x2c39662509203233ull,
0x203b393533722520ull,0x090a202f2f092020ull,0x203233662e646461ull,0x6625202c31662509ull,
0x203b316625202c39ull,0x2f09202020202020ull,0x636f6c2e090a202full,0x0930373109373109ull,
0x732e78616d090a30ull,0x3233722509203233ull,0x3835337225202c31ull,0x3b3132337225202cull,
0x4c240a202f2f0920ull,0x383537365f305f74ull,0x636f6c2e090a3a36ull,0x0933373109373109ull,
0x68732e7473090a30ull,0x3233662e64657261ull,0x35326472255b0920ull,0x316625202c5d302bull,
0x3a6469202f2f093bull,0x302b746d20353836ull,0x636f6c2e090a3078ull,0x0934373109373109ull,
0x68732e7473090a30ull,0x3233752e64657261ull,0x35326472255b0920ull,0x202c5d343230312bull,
0x2f093b3132337225ull,0x3638363a6469202full,0x0a3078302b746d20ull,0x373109636f6c2e09ull,
0x090a300935373109ull,0x636e79732e726162ull,0x202020203b300920ull,0x2020202020202020ull,
0x2f09202020202020ull,0x2e766f6d090a202full,0x3372250920323373ull,0x3b383231202c3532ull,
0x2020202020202020ull,0x240a202f2f092020ull,0x3638365f305f744cull,0x3c2f2f200a3a3031ull,
0x6f4c203e706f6f6cull,0x2079646f6220706full,0x35373120656e696cull,0x6e697473656e202cull,
0x3a68747065642067ull,0x69747365202c3120ull,0x746920646574616dull,0x736e6f6974617265ull,
0x776f6e6b6e75203aull,0x2e70746573090a6eull,0x09203233752e656cull,0x7225202c36347025ull,
0x317225202c353233ull,0x090a202f2f093b31ull,0x7262203634702540ull,0x305f744c24092061ull,
0x203b38373339365full,0x2f09202020202020ull,0x6c3c2f2f200a202full,0x726150203e706f6full,
0x6f6f6c20666f2074ull,0x6c2079646f622070ull,0x2c35373120656e69ull,0x616c206461656820ull,
0x4c242064656c6562ull,0x313638365f305f74ull,0x09636f6c2e090a30ull,0x3009313831093731ull,
0x33752e646461090aull,0x3036337225092032ull,0x2c3532337225202cull,0x20203b3131722520ull,
0x7663090a202f2f09ull,0x33752e3436752e74ull,0x3531647225092032ull,0x3036337225202c32ull,
0x202f2f092020203bull,0x6f6c2e6c756d090aull,0x722509203436752eull,0x7225202c33353164ull,
0x3b34202c32353164ull,0x6461090a202f2f09ull,0x2509203436752e64ull,0x25202c3435316472ull,
0x7225202c35316472ull,0x2f2f093b33353164ull,0x68732e646c090a20ull,0x3233662e64657261ull,
0x202c303166250920ull,0x2b3435316472255bull,0x69202f2f093b5d30ull,0x746d203838363a64ull,
0x646c090a3078302bull,0x2e6465726168732eull,0x3166250920323366ull,0x326472255b202c31ull,
0x2f2f093b5d302b35ull,0x203738363a646920ull,0x090a3078302b746dull,0x203233662e646461ull,
0x25202c3231662509ull,0x316625202c303166ull,0x2f09202020203b31ull,0x732e7473090a202full,
0x33662e6465726168ull,0x326472255b092032ull,0x6625202c5d302b35ull,0x69202f2f093b3231ull,
0x746d203938363a64ull,0x646c090a3078302bull,0x2e6465726168732eull,0x3372250920323375ull,
0x6472255b202c3136ull,0x343230312b343531ull,0x6469202f2f093b5dull,0x2b746d203039363aull,
0x2e646c090a307830ull,0x752e646572616873ull,0x3633722509203233ull,0x326472255b202c32ull,
0x3b5d343230312b35ull,0x363a6469202f2f09ull,0x78302b746d203139ull,0x2e70746573090a30ull,
0x09203233752e6567ull,0x7225202c37347025ull,0x337225202c323633ull,0x0a202f2f093b3136ull,
0x6220373470254009ull,0x5f744c2409206172ull,0x3b38373339365f30ull,0x0920202020202020ull,
0x3c2f2f200a202f2full,0x6150203e706f6f6cull,0x6f6c20666f207472ull,0x2079646f6220706full,
0x35373120656e696cull,0x6c2064616568202cull,0x242064656c656261ull,0x3638365f305f744cull,
0x636f6c2e090a3031ull,0x0933383109373109ull,0x68732e7473090a30ull,0x3233752e64657261ull,
0x35326472255b0920ull,0x202c5d343230312bull,0x2f093b3136337225ull,0x3239363a6469202full,
0x0a3078302b746d20ull,0x39365f305f744c24ull,0x744c240a3a383733ull,0x36363838365f305full,
0x6f6c3c2f2f200a3aull,0x74726150203e706full,0x706f6f6c20666f20ull,0x696c2079646f6220ull,
0x202c35373120656eull,0x62616c2064616568ull,0x744c242064656c65ull,0x30313638365f305full,
0x3109636f6c2e090aull,0x0a30093538310937ull,0x6e79732e72616209ull,0x2020203b30092063ull,
0x2020202020202020ull,0x0920202020202020ull,0x6f6c2e090a202f2full,0x3737310937310963ull,
0x2e726873090a3009ull,0x3372250920323373ull,0x32337225202c3532ull,0x2020203b31202c35ull,
0x090a202f2f092020ull,0x203233752e766f6dull,0x202c333633722509ull,0x2020202020203b30ull,
0x2f09202020202020ull,0x70746573090a202full,0x203233732e74672eull,0x25202c3834702509ull,
0x7225202c35323372ull,0x202f2f093b333633ull,0x203834702540090aull,0x744c240920617262ull,
0x30313638365f305full,0x202020202020203bull,0x2140090a202f2f09ull,0x2061726220347025ull,
0x375f305f744c2409ull,0x2020203b36343130ull,0x202f2f0920202020ull,0x3109636f6c2e090aull,
0x0a30093039310937ull,0x3233752e74766309ull,0x722509203631752eull,0x746325202c343633ull,
0x09203b782e646961ull,0x747663090a202f2full,0x3233752e3436752eull,0x3535316472250920ull,
0x3b3436337225202cull,0x0a202f2f09202020ull,0x2e6f6c2e6c756d09ull,0x6472250920343675ull,
0x647225202c363531ull,0x093b34202c353531ull,0x2e646c090a202f2full,0x662e646572616873ull,
0x3331662509203233ull,0x5d302b746d5b202cull,0x69202f2f0920203bull,0x746d203339363a64ull,
0x7663090a3078302bull,0x3233662e6e722e74ull,0x662509203233732eull,0x3b337225202c3431ull,
0x202f2f0920202020ull,0x70612e766964090aull,0x3233662e786f7270ull,0x202c353166250920ull,
0x6625202c33316625ull,0x0a202f2f093b3431ull,0x3233662e766f6d09ull,0x202c363166250920ull,
0x3030303866626630ull,0x09202020203b3030ull,0x61090a312d202f2full,0x09203233662e6464ull,
0x6625202c37316625ull,0x36316625202c3531ull,0x2f2f09202020203bull,0x61702e646c090a20ull,
0x203436752e6d6172ull,0x2c37353164722509ull,0x616475635f5f5b20ull,0x315a5f5f6d726170ull,
0x656e697274655034ull,0x506c656e72654b74ull,0x6969695f53665069ull,0x2f093b5d765f675full,
0x3439363a6469202full,0x70616475635f5f20ull,0x34315a5f5f6d7261ull,0x74656e6972746550ull,
0x69506c656e72654bull,0x5f6969695f536650ull,0x0a3078302b765f67ull,0x3436752e64646109ull,
0x3835316472250920ull,0x373531647225202cull,0x363531647225202cull,0x73090a202f2f093bull,
0x6c61626f6c672e74ull,0x255b09203233662eull,0x5d302b3835316472ull,0x093b37316625202cull,
0x39363a6469202f2full,0x09636f6c2e090a35ull,0x3009323931093731ull,0x6168732e646c090aull,
0x203233732e646572ull,0x202c353633722509ull,0x343230312b746d5bull,0x6469202f2f093b5dull,
0x2b746d203639363aull,0x6c090a3030347830ull,0x2e6d617261702e64ull,0x6472250920343675ull,
0x5f5f5b202c393531ull,0x6d72617061647563ull,0x74655034315a5f5full,0x72654b74656e6972ull,
0x53665069506c656eull,0x6d5f675f6969695full,0x6469202f2f093b5dull,0x635f5f203739363aull,
0x5f6d726170616475ull,0x7274655034315a5full,0x6e72654b74656e69ull,0x5f53665069506c65ull,
0x2b6d5f675f696969ull,0x646461090a307830ull,0x722509203436752eull,0x7225202c30363164ull,
0x7225202c39353164ull,0x2f2f093b36353164ull,0x6c672e7473090a20ull,0x3233732e6c61626full,
0x36316472255b0920ull,0x7225202c5d302b30ull,0x202f2f093b353633ull,0x240a3839363a6469ull,
0x3130375f305f744cull,0x6f6c2e090a3a3634ull,0x3231320937310963ull,0x74697865090a3009ull,
0x202020202020203bull,0x2020202020202020ull,0x2020202020202020ull,0x240a202f2f092020ull,
0x5f5f646e6557444cull,0x697274655034315aull,0x656e72654b74656eull,0x695f53665069506cull,
0x2f207d090a3a6969ull,0x655034315a5f202full,0x654b74656e697274ull,0x665069506c656e72ull,
0x000a0a6969695f53ull
};


}


static __cudaFatPtxEntry __ptxEntries [] = {{(char*)"compute_13",(char*)__deviceText_$compute_13$},{0,0}};
static __cudaFatCubinEntry __cubinEntries[] = {{(char*)"sm_13",(char*)__deviceText_$sm_13$},{0,0}};
static __cudaFatDebugEntry __debugEntries[] = {{0,0}};



static __cudaFatCudaBinary __fatDeviceText __attribute__ ((section (".nvFatBinSegment")))= {0x1ee55a01,0x00000003,0x8ecc680c,(char*)"9079ddbf9c169c73",(char*)"main.cu",(char*)" ",__ptxEntries,__cubinEntries,__debugEntries,0,0,0,0,0,0xaa3c4621};
# 5 "/tmp/tmpxft_00006e97_00000000-1_main.cudafe1.stub.c" 2
# 1 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 1
# 85 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
# 1 "/usr/local/cuda/bin/../include/host_defines.h" 1
# 86 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2
# 111 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
extern "C" {


extern void** __cudaRegisterFatBinary(
  void *fatCubin
);

extern void __cudaUnregisterFatBinary(
  void **fatCubinHandle
);

extern void __cudaRegisterVar(
        void **fatCubinHandle,
        char *hostVar,
        char *deviceAddress,
  const char *deviceName,
        int ext,
        int size,
        int constant,
        int global
);

extern void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext
);

extern void __cudaRegisterShared(
  void **fatCubinHandle,
  void **devicePtr
);

extern void __cudaRegisterSharedVar(
  void **fatCubinHandle,
  void **devicePtr,
  size_t size,
  size_t alignment,
  int storage
);

extern void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize
);


}




extern int atexit(void(*)(void)) throw();







static void **__cudaFatCubinHandle;

static void __cudaUnregisterBinaryUtil(void)
{
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}
# 226 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
# 1 "/usr/local/cuda/bin/../include/common_functions.h" 1
# 68 "/usr/local/cuda/bin/../include/common_functions.h"
# 1 "/usr/local/cuda/bin/../include/crt/func_macro.h" 1 3
# 69 "/usr/local/cuda/bin/../include/common_functions.h" 2

static clock_t __cuda_clock(void)
{
  return clock();
}

static void *__cuda_memset(void *s, int c, size_t n)
{
  return memset(s, c, n);
}

static void *__cuda_memcpy(void *d, const void *s, size_t n)
{
  return memcpy(d, s, n);
}
# 93 "/usr/local/cuda/bin/../include/common_functions.h"
# 1 "/usr/local/cuda/bin/../include/math_functions.h" 1 3
# 864 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) int __cuda_error_not_implememted(void); int __cuda_error_not_implememted(void);
# 920 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda_abs(int a)
{
  return abs(a);
}

static float __cuda_fabsf(float a)
{
  return fabsf(a);
}

static long long int __cuda_llabs(long long int a)
{

  return ::llabs(a);



}

static float __cuda_exp2f(float a)
{
  return exp2f(a);
}

# 1 "/usr/local/cuda/bin/../include/device_functions.h" 1 3
# 426 "/usr/local/cuda/bin/../include/device_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/math_constants.h" 1 3
# 427 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 2367 "/usr/local/cuda/bin/../include/device_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/common_types.h" 1 3
# 37 "/usr/local/cuda/bin/../include/common_types.h" 3
union __cudart_FloatUintCvt {
    float f;
    unsigned int i;
};

union __cudart_FloatIntCvt {
    float f;
    int i;
};

union __cudart_DoubleUlonglongCvt {
    double d;
    unsigned long long int i;
};

union __cudart_DoubleLonglongCvt {
    double d;
    long long int i;
};

union __cudart_DoubleInthiloCvt {
    double d;
    signed int i[2];
};

struct __cudart_UintUint {
    unsigned int lo;
    unsigned int hi;
};
# 2368 "/usr/local/cuda/bin/../include/device_functions.h" 2 3

static const unsigned char __internal_rcpTab[128] =
{
  0xff, 0xfd, 0xfb, 0xf9, 0xf7, 0xf5, 0xf4, 0xf2,
  0xf0, 0xee, 0xed, 0xeb, 0xe9, 0xe8, 0xe6, 0xe4,
  0xe3, 0xe1, 0xe0, 0xde, 0xdd, 0xdb, 0xda, 0xd8,
  0xd7, 0xd5, 0xd4, 0xd3, 0xd1, 0xd0, 0xcf, 0xcd,
  0xcc, 0xcb, 0xca, 0xc8, 0xc7, 0xc6, 0xc5, 0xc4,
  0xc2, 0xc1, 0xc0, 0xbf, 0xbe, 0xbd, 0xbc, 0xbb,
  0xba, 0xb9, 0xb8, 0xb7, 0xb6, 0xb5, 0xb4, 0xb3,
  0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xac, 0xab,
  0xaa, 0xa9, 0xa8, 0xa8, 0xa7, 0xa6, 0xa5, 0xa4,
  0xa3, 0xa3, 0xa2, 0xa1, 0xa0, 0x9f, 0x9f, 0x9e,
  0x9d, 0x9c, 0x9c, 0x9b, 0x9a, 0x99, 0x99, 0x98,
  0x97, 0x97, 0x96, 0x95, 0x95, 0x94, 0x93, 0x93,
  0x92, 0x91, 0x91, 0x90, 0x8f, 0x8f, 0x8e, 0x8e,
  0x8d, 0x8c, 0x8c, 0x8b, 0x8b, 0x8a, 0x89, 0x89,
  0x88, 0x88, 0x87, 0x87, 0x86, 0x85, 0x85, 0x84,
  0x84, 0x83, 0x83, 0x82, 0x82, 0x81, 0x81, 0x80
};

static const unsigned int __internal_invSqrtCubeTab[96] =
{
  0xfa0bf8fe, 0xee6b28fa, 0xe5f024f7, 0xdaf268f3,
  0xd2f000f0, 0xc890c0ec, 0xc10378e9, 0xb9a758e6,
  0xb4da40e4, 0xadcea0e1, 0xa6f278de, 0xa279c0dc,
  0x9beb48d9, 0x97a5c4d7, 0x916340d4, 0x8d4fc8d2,
  0x895000d0, 0x8563b8ce, 0x818ac0cc, 0x7dc4e8ca,
  0x7a1200c8, 0x7671d8c6, 0x72e440c4, 0x6f6908c2,
  0x6db240c1, 0x6a523cbf, 0x670424bd, 0x6563c0bc,
  0x623028ba, 0x609ce8b9, 0x5d8364b7, 0x5bfd18b6,
  0x58fd40b4, 0x5783a8b3, 0x560e48b2, 0x533000b0,
  0x51c70caf, 0x506238ae, 0x4da4c0ac, 0x4c4c10ab,
  0x4af768aa, 0x49a6b8a9, 0x485a00a8, 0x471134a7,
  0x45cc58a6, 0x434e40a4, 0x4214f8a3, 0x40df88a2,
  0x3fade0a1, 0x3e8000a0, 0x3d55dc9f, 0x3c2f789e,
  0x3c2f789e, 0x3b0cc49d, 0x39edc09c, 0x38d2609b,
  0x37baa89a, 0x36a68899, 0x35960098, 0x34890497,
  0x34890497, 0x337f9896, 0x3279ac95, 0x31774094,
  0x30784893, 0x30784893, 0x2f7cc892, 0x2e84b091,
  0x2d900090, 0x2d900090, 0x2c9eac8f, 0x2bb0b88e,
  0x2bb0b88e, 0x2ac6148d, 0x29dec08c, 0x29dec08c,
  0x28fab08b, 0x2819e88a, 0x2819e88a, 0x273c5889,
  0x273c5889, 0x26620088, 0x258ad487, 0x258ad487,
  0x24b6d886, 0x24b6d886, 0x23e5fc85, 0x23184084,
  0x23184084, 0x224d9883, 0x224d9883, 0x21860882,
  0x21860882, 0x20c18081, 0x20c18081, 0x20000080
};

static float __internal_frcp_kernel (float x,enum cudaRoundMode mode)
{
  unsigned long long prod;
  volatile union __cudart_FloatUintCvt arg;
  unsigned int expo;
  unsigned int sign;
  unsigned f, y;

  arg.f = x;
  sign = arg.i & 0x80000000;
  expo = (arg.i >> 23);
  expo = expo & 0xff;
  f = expo - 1;

  if (f <= 0xFD) {
    y = (arg.i << 8);
    y = y | 0x80000000;

    arg.i = __internal_rcpTab[(y >> 24) - 128];

    f = arg.i * arg.i;
    f = f << 16;
    prod = ((unsigned long long)y) * f;
    arg.i = (arg.i << 24) - (unsigned)(prod >> 32);

    f = arg.i + arg.i;
    prod = ((unsigned long long)y) * f;
    f = (unsigned)(-(int)(prod >> 32));
    prod = ((unsigned long long)arg.i) * f;
    y = y >> 8;

    expo = (2 * 127) - expo - 2;
    arg.i = (unsigned)(prod >> 32);
    if (mode == cudaRoundNearest) {
      arg.i = arg.i >> 6;
    } else {
      arg.i = (arg.i + 32) >> 6;
    }
    if ((int)expo >= 0) {
      f = y * arg.i;
      arg.i = ((expo << 23) + arg.i) | sign;
    } else {

      expo = -(int)expo;
      arg.i = arg.i >> expo;
      f = y * arg.i;
      arg.i = arg.i | sign;
    }
    if (mode == cudaRoundNearest) {
      expo = f + y;
      if ((int)f < 0) f = (unsigned)(-(int)f);
      if ((int)expo < 0) expo = (unsigned)(-(int)expo);
      if (expo < f) arg.i++;
    } else if (mode == cudaRoundZero) {
      if ((int)f > 0) arg.i = arg.i - 1;
    } else if (mode == cudaRoundPosInf) {
      if (((int)f > 0) && sign) arg.i = arg.i - 1;
      if (((int)f < 0) && !sign) arg.i = arg.i + 1;
    } else {
      if (((int)f > 0) && !sign) arg.i = arg.i - 1;
      if (((int)f < 0) && sign) arg.i = arg.i + 1;
    }
    return arg.f;
  } else {

    if (!(arg.i << 1)) {
      arg.i = 0x7F800000 | arg.i;
      return arg.f;
    }

    if ((arg.i << 1) == 0xff000000) {
      arg.i &= 0x80000000;
      return arg.f;
    }

    if ((arg.i << 1) > 0xff000000) {
      arg.i |= 0x00400000;
      return arg.f;
    }

    f = 0;
    arg.i <<= 8;
    do {
      f++;
      arg.i <<= 1;
    } while ((int)arg.i > 0);
    arg.i >>= 8;
    arg.i |= sign;
    arg.f = __internal_frcp_kernel (arg.f, mode);
    expo = ((arg.i << 1) >> 24);
    if ((expo + f) < 255) {
      arg.i = (arg.i + (f << 23));
      return arg.f;
    }
    if (mode == cudaRoundNearest) {
      arg.i = (arg.i & 0x80000000) | 0x7f800000;
    } else if (mode == cudaRoundZero) {
      arg.i = (arg.i & 0x80000000) | 0x7f7fffff;
    } else if (mode == cudaRoundPosInf) {
      arg.i = (arg.i & 0x80000000) | ((sign) ? 0x7f7fffff : 0x7f800000);
    } else {
      arg.i = (arg.i & 0x80000000) | ((sign) ? 0x7f800000 : 0x7f7fffff);
    }
    return arg.f;
  }
}

static float __internal_fsqrt_kernel (float radicand, enum cudaRoundMode mode)

{
  unsigned long long prod;
  volatile union __cudart_FloatUintCvt arg;
  unsigned int expo;
  unsigned int s, f, x;

  arg.f = radicand;
  expo = arg.i >> 23;
  expo = expo & 0xff;
  f = expo - 1;

  if ((arg.i <= 0x80000000) && (f <= 0xFD)) {

    x = (arg.i << 8) | 0x80000000;
    x = x >> (expo & 1);

    arg.i = f = __internal_invSqrtCubeTab[((unsigned)x >> 25) - 32];

    prod = ((unsigned long long)x) * f;
    arg.i = ((arg.i * 3) << 22) - (unsigned)(prod >> 32);

    prod = ((unsigned long long)arg.i) * arg.i;
    s = (unsigned)(prod >> 32);
    prod = ((unsigned long long)x) * s;
    f = 0x30000000 - (unsigned)(prod >> 32);
    prod = ((unsigned long long)f) * arg.i;
    arg.i = (unsigned)(prod >> 32);

    prod = ((unsigned long long)x) * arg.i;
    arg.i = (unsigned)(prod >> 32);
    if (mode == cudaRoundNearest) {
      arg.i = arg.i >> 3;
    } else {
      arg.i = (arg.i + 4) >> 3;
    }
    x = (x << 16) - (arg.i * arg.i);

    if (mode == cudaRoundNearest) {
      f = x - (2 * arg.i + 1);
      if ((int)f < 0) f = (unsigned)(-(int)f);
      if ((int)x < 0) x = (unsigned)(-(int)x);
      if (f < x) arg.i ++;
    } else if ((mode == cudaRoundZero) || (mode == cudaRoundMinInf)) {
      if ((int)x < 0) arg.i--;
    } else if (mode == cudaRoundPosInf) {
      if ((int)x > 0) arg.i++;
    }
    arg.i = arg.i + (((expo + 125) & ~0x1) << 22);
    return arg.f;
  } else {

    if (!(arg.i << 1) || (arg.i == 0x7F800000)) {
      return arg.f;
    }

    if ((arg.i << 1) > 0xFF000000) {
      arg.i |= 0x00400000;
      return arg.f;
    }

    if (arg.i & 0x80000000) {
      arg.i = 0xFFC00000;
      return arg.f;
    }

    x = 0;
    arg.i <<= 8;
    do {
      x++;
      arg.i <<= 1;
    } while ((int)arg.i > 0);
    arg.i >>= 8;
    arg.i += (x & 1) << 23;
    x += (x & 1);
    arg.f = __internal_fsqrt_kernel (arg.f, mode);
    arg.i -= ((x >> 1) << 23);
    return arg.f;
  }
}

static float __internal_fdiv_kernel (float dividend, float divisor, enum cudaRoundMode mode)

{
  unsigned long long prod;
  unsigned r, f, x, y, expox, expoy, sign;
  volatile union __cudart_FloatUintCvt cvtx, cvty, res;

  cvtx.f = dividend;
  cvty.f = divisor;
  expox = ((cvtx.i >> 23) & 0xff) - 1;
  expoy = ((cvty.i >> 23) & 0xff) - 1;
  sign = ((cvtx.i ^ cvty.i) & 0x80000000);

  if ((expox <= 0xFD) && (expoy <= 0xFD)) {
divide:
    expox = expox - expoy + 127 - 1;
    expoy = expox;

    y = (cvty.i << 8) | 0x80000000;
    x = (cvtx.i & 0x00ffffff) | 0x00800000;

    r = __internal_rcpTab[(y >> 24) - 128];

    f = r * r;
    prod = ((unsigned long long)y) * (f << 16);
    r = (r << 24) - (unsigned)(prod >> 32);

    prod = ((unsigned long long)y) * (r << 1);
    f = (unsigned)-(int)(prod >> 32);
    prod = ((unsigned long long)f) * (r << 1);
    r = (unsigned)(prod >> 32);

    prod = ((unsigned long long)x) * (r << 1);

    if (((int)((prod >> 32) << 8)) > 0) {
      expox--;
      prod = prod + prod;
    }
    if (mode == cudaRoundNearest) {

      r = (unsigned)(prod >> 32);
      y = y >> 8;

      if (expox <= 0xFD) {
        int rem0, rem1, inc;

        prod = ((unsigned long long)y) * r;
        x = x << (23 + ((prod >> 32) >> 15));
        rem1 = x - (unsigned)(prod & 0xffffffff);
        rem0 = rem1 - y;
        inc = abs(rem0) < abs(rem1);

        res.i = sign | ((expox << 23) + r + inc);
        return res.f;
      } else if ((int)expox >= 254) {

        res.i = sign | 0x7f800000;
        return res.f;
      } else {

        int shift = -(int)expox;
        if (shift > 23) {

          r = (shift < 25) && ((x != y) || (r > 0x00ff0000));
          res.i = sign | r;
          return res.f;
        }
        if (x == y) {

          shift = -(int)expoy;
          r = 0x00800000 >> shift;
          res.i = sign | r;
          return res.f;
        }
        {
          unsigned long long tempx;
          long long remlo, remhi;

          r = r >> shift;
          prod = ((unsigned long long)y) * r;
          tempx = ((unsigned long long)x) << (23 - shift);
          remlo = 2 * tempx - 2 * prod - y;
          remhi = remlo + 2 * tempx;
          if (remlo < 0) remlo = -remlo;
          if (remhi < 0) remhi = -remhi;
          if (remhi < remlo) tempx = 2 * tempx;
          remlo = tempx - prod;
          remhi = remlo - y;
          if (remlo < 0) remlo = -remlo;
          if (remhi < 0) remhi = -remhi;
          if ((remhi < remlo) || ((remhi == remlo) && (r & 1))) r++;
          res.i = sign | r;
          return res.f;
        }
      }
    } else if (mode == cudaRoundZero) {

      prod += 0x0000000080000000ULL;
      r = (unsigned)(prod >> 32);
      y = y >> 8;

      if (expox <= 0xFD) {
        int rem1;
        prod = ((unsigned long long)y) * r;
        x = x << (23 + ((prod >> 32) >> 15));
        rem1 = x - (unsigned)(prod & 0xffffffff);
        if (rem1 < 0) r--;
        r = (expox << 23) + r;
        if (r == 0x7f800000) r = 0x7f7fffff;
        res.i = sign | r;
        return res.f;
      } else if ((int)expox >= 254) {

        res.i = sign | 0x7f7fffff;
        return res.f;
      } else {

        int shift = -(int)expox;
        if ((x == y) && (shift < 31)) {
          shift = -(int)expoy;
          r = 0x00800000 >> shift;
          res.i = sign | r;
          return res.f;
        }
        if (shift > 23) {
          r = 0;
          res.i = sign | r;
          return res.f;
        }
        {
          unsigned long long tempx;
          long long remlo, remhi;

          r = r >> shift;
          prod = ((unsigned long long)y) * r;
          tempx = ((unsigned long long)x) << (23 - shift);
          remlo = 2 * tempx - 2 * prod - y;
          remhi = remlo + 2 * tempx;
          if (remlo < 0) remlo = -remlo;
          if (remhi < 0) remhi = -remhi;
          if (remhi < remlo) tempx = 2 * tempx;
          remlo = tempx - prod;
          if ((remlo < 0) & (r != 0)) r--;
          res.i = sign | r;
          return res.f;
        }
      }
    } else if (mode == cudaRoundPosInf) {

      prod += 0x0000000080000000ULL;
      r = (unsigned)(prod >> 32);
      y = y >> 8;

      if (expox <= 0xFD) {
        int rem1;
        prod = ((unsigned long long)y) * r;
        x = x << (23 + ((prod >> 32) >> 15));
        rem1 = x - (unsigned)(prod & 0xffffffff);
        if ((rem1 < 0) && (sign)) r--;
        if ((rem1 > 0) && (!sign)) r++;
        r = (expox << 23) + r;
        if ((r == 0x7f800000) && (sign)) r = 0x7f7fffff;
        res.i = sign | r;
        return res.f;
      } else if ((int)expox >= 254) {

        r = sign ? 0x7f7fffff : 0x7f800000;
        res.i = sign | r;
        return res.f;
      } else {

        int shift = -(int)expox;
        if ((x == y) && (shift <= 24)) {
          shift = -(int)expoy;
          r = 0x00800000 >> shift;
          if (r == 0) r = !sign;
          res.i = sign | r;
          return res.f;
        }
        if (shift > 23) {
          r = !sign;
          res.i = sign | r;
          return res.f;
        }
        {
          unsigned long long tempx;
          long long remlo, remhi;

          r = r >> shift;
          prod = ((unsigned long long)y) * r;
          tempx = ((unsigned long long)x) << (23 - shift);
          remlo = 2 * tempx - 2 * prod - y;
          remhi = remlo + 2 * tempx;
          if (remlo < 0) remlo = -remlo;
          if (remhi < 0) remhi = -remhi;
          if (remhi < remlo) tempx = 2 * tempx;
          remlo = tempx - prod;
          if ((remlo < 0) && (r != 0) && (sign)) r--;
          if ((remlo > 0) && (!sign)) r++;
          res.i = sign | r;
          return res.f;
        }
      }
    } else if (mode == cudaRoundMinInf) {

      prod += 0x0000000080000000ULL;
      r = (unsigned)(prod >> 32);
      y = y >> 8;

      if (expox <= 0xFD) {
        int rem1;
        prod = ((unsigned long long)y) * r;
        x = x << (23 + ((prod >> 32) >> 15));
        rem1 = x - (unsigned)(prod & 0xffffffff);
        if ((rem1 < 0) && (!sign)) r--;
        if ((rem1 > 0) && (sign)) r++;
        r = (expox << 23) + r;
        if ((r == 0x7f800000) && (!sign)) r = 0x7f7fffff;
        res.i = sign | r;
        return res.f;
      } else if ((int)expox >= 254) {

        r = sign ? 0x7f800000 : 0x7f7fffff;
        res.i = sign | r;
        return res.f;
      } else {

        int shift = -(int)expox;
        if ((x == y) && (shift <= 24)) {
          shift = -(int)expoy;
          r = 0x00800000 >> shift;
          if (r == 0) r = !!sign;
          res.i = sign | r;
          return res.f;
        }
        if (shift > 23) {
          r = !!sign;
          res.i = sign | r;
          return res.f;
        }
        {
          unsigned long long tempx;
          long long remlo, remhi;

          r = r >> shift;
          prod = ((unsigned long long)y) * r;
          tempx = ((unsigned long long)x) << (23 - shift);
          remlo = 2 * tempx - 2 * prod - y;
          remhi = remlo + 2 * tempx;
          if (remlo < 0) remlo = -remlo;
          if (remhi < 0) remhi = -remhi;
          if (remhi < remlo) tempx = 2 * tempx;
          remlo = tempx - prod;
          if ((remlo < 0) && (r != 0) && (!sign)) r--;
          if ((remlo > 0) && (sign)) r++;
          res.i = sign | r;
          return res.f;
        }
      }
    }
  }
  {
    int xzero, yzero, xinf, yinf, xnan, ynan;

    xnan = (cvtx.i << 1) > 0xff000000;
    ynan = (cvty.i << 1) > 0xff000000;

    if (xnan) {
      res.i = cvtx.i | 0x00400000;
      return res.f;
    }
    if (ynan) {
      res.i = cvty.i | 0x00400000;
      return res.f;
    }
    xzero = (cvtx.i << 1) == 0x00000000;
    yzero = (cvty.i << 1) == 0x00000000;
    xinf = (cvtx.i << 1) == 0xff000000;
    yinf = (cvty.i << 1) == 0xff000000;

    if ((xzero & yzero) | (xinf & yinf)) {
      res.i = 0xffc00000;
      return res.f;
    }

    if (xzero | yinf) {
      res.i = sign;
      return res.f;
    }

    if (yzero | xinf) {
      res.i = sign | 0x7f800000;
      return res.f;
    }

    if ((int)expox < 0) {
      cvtx.i = cvtx.i << 9;
      while ((int)cvtx.i >= 0) {
        expox--;
        cvtx.i = cvtx.i + cvtx.i;
      }
      cvtx.i = cvtx.i >> 8;
    }
    if ((int)expoy < 0) {
      cvty.i = cvty.i << 9;
      while ((int)cvty.i >= 0) {
        expoy--;
        cvty.i = cvty.i + cvty.i;
      }
      cvty.i = cvty.i >> 8;
    }
    goto divide;
  }
}

static float __internal_fmul_kernel2 (float a, float b, enum cudaRoundMode mode)

{
  unsigned long long product;
  volatile union __cudart_FloatUintCvt xx, yy;

  unsigned expo_x, expo_y;

  xx.f = a;
  yy.f = b;

  expo_y = 0xFF;
  expo_x = expo_y & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {
multiply:
    expo_x = expo_x + expo_y;
    expo_y = xx.i ^ yy.i;
    xx.i = xx.i & 0x00ffffff;
    yy.i = yy.i << 8;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i | 0x80000000;

    product = ((unsigned long long)xx.i) * yy.i;
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xx.i = (unsigned int)(product >> 32);
    yy.i = (unsigned int)(product & 0xffffffff);

    if (xx.i < 0x00800000) {
      xx.i = (xx.i << 1) | (yy.i >> 31);
      yy.i = (yy.i << 1);
      expo_x--;
    }
    if (expo_x <= 0xFD) {
      xx.i = xx.i | expo_y;
      xx.i = xx.i + (expo_x << 23);

      if (mode == cudaRoundNearest) {
        if (yy.i < 0x80000000) return xx.f;
        xx.i += ((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31));
      } else if (mode == cudaRoundZero) {
      } else if (mode == cudaRoundPosInf) {
        xx.i += (yy.i && !expo_y);
      } else if (mode == cudaRoundMinInf) {
        xx.i += (yy.i && expo_y);
      }
      return xx.f;
    } else if ((int)expo_x >= 254) {

      if (mode == cudaRoundNearest) {
         xx.i = expo_y | 0x7F800000;
      } else if (mode == cudaRoundZero) {
         xx.i = expo_y | 0x7F7FFFFF;
      } else if (mode == cudaRoundPosInf) {
         xx.i = (expo_y ? 0xff7fffff : 0x7F800000);
      } else {
         xx.i = (expo_y ? 0xFF800000 : 0x7f7fffff);
      }
      return xx.f;
    } else {

      expo_x = ((unsigned int)-((int)expo_x));
      if (mode == cudaRoundNearest) {
        if (expo_x > 25) {

          xx.i = expo_y;
          return xx.f;
        } else {
          yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
          xx.i = expo_y + (xx.i >> expo_x);
          xx.i += ((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31));
          return xx.f;
        }
      } else if (mode == cudaRoundZero) {
        if (expo_x > 25) expo_x = 25;
        xx.i = expo_y + (xx.i >> expo_x);
        return xx.f;
      } else if (mode == cudaRoundPosInf) {
        if (expo_x > 25) expo_x = 25;
        yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
        xx.i = expo_y + (xx.i >> expo_x);
        xx.i += (yy.i && !expo_y);
        return xx.f;
      } else {
        if (expo_x > 25) expo_x = 25;
        yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
        xx.i = expo_y + (xx.i >> expo_x);
        xx.i += (yy.i && expo_y);
        return xx.f;
      }
    }
  } else {
    product = xx.i ^ yy.i;
    product = product & 0x80000000;
    if (!(xx.i & 0x7fffffff)) {
      if (expo_y != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_y = yy.i << 1;
      if (expo_y == 0xFF000000) {
        xx.i = expo_y | 0x00C00000;
      } else {
        xx.i = yy.i | 0x00400000;
      }
      return xx.f;
    }
    if (!(yy.i & 0x7fffffff)) {
      if (expo_x != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_x = xx.i << 1;
      if (expo_x == 0xFF000000) {
        xx.i = expo_x | 0x00C00000;
      } else {
        xx.i = xx.i | 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {
      expo_y++;
      expo_x++;
      if (expo_x == 0) {
        expo_y |= xx.i & 0x80000000;




        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        xx.i = (xx.i >> 8) | (expo_y & 0x80000000);
        expo_y &= ~0x80000000;
        expo_y--;
        goto multiply;
      }
      if (expo_y == 0) {
        expo_x |= yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        yy.i = (yy.i >> 8) | (expo_x & 0x80000000);
        expo_x &= ~0x80000000;
        expo_x--;
        goto multiply;
      }
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xFF000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xFF000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    xx.i = (unsigned int)product | 0x7f800000;
    return xx.f;
  }
}

static float __internal_fmaf_kernel (float a, float b, float c, enum cudaRoundMode mode)

{
  unsigned long long product;
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;
  volatile union __cudart_FloatUintCvt cvt;

  cvt.f = a;
  xx = cvt.i;
  cvt.f = b;
  yy = cvt.i;
  cvt.f = c;
  zz = cvt.i;

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if (!((expo_x <= 0xFD) &&
        (expo_y <= 0xFD) &&
        (expo_z <= 0xFD))) {




    if ((yy << 1) > 0xff000000) {
      return b + b;
    }
    if ((zz << 1) > 0xff000000) {
      return c + c;
    }
    if ((xx << 1) > 0xff000000) {
      return a + a;
    }
# 3148 "/usr/local/cuda/bin/../include/device_functions.h" 3
    if ((((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
        (((yy << 1) == 0) && ((xx << 1) == 0xff000000))) {
      cvt.i = 0xffc00000;
      return cvt.f;
    }
    if ((zz << 1) == 0xff000000) {
      if (((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000)) {
        if ((int)(xx ^ yy ^ zz) < 0) {
          cvt.i = 0xffc00000;
          return cvt.f;
        }
      }
    }




    if ((xx << 1) == 0xff000000) {
      xx = xx ^ (yy & 0x80000000);
      cvt.i = xx;
      return cvt.f;
    }
    if ((yy << 1) == 0xff000000) {
      yy = yy ^ (xx & 0x80000000);
      cvt.i = yy;
      return cvt.f;
    }
    if ((zz << 1) == 0xff000000) {
      cvt.i = zz;
      return cvt.f;
    }





    if (zz == 0x80000000) {
      if (((xx << 1) == 0) || ((yy << 1) == 0)) {
        if ((int)(xx ^ yy) < 0) {
          cvt.i = zz;
          return cvt.f;
        }
      }
    }



    if (((zz << 1) == 0) &&
        (((xx << 1) == 0) || ((yy << 1) == 0))) {
      if (mode == cudaRoundMinInf) {
        zz = 0x80000000 & (xx ^ yy ^ zz);
      } else {
        zz &= 0x7fffffff;
      }
      cvt.i = zz;
      return cvt.f;
    }



    if (((xx << 1) == 0) || ((yy << 1) == 0)) {
      cvt.i = zz;
      return cvt.f;
    }

    if (expo_x == (unsigned)-1) {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while (!(xx & 0x80000000)) {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }

    if (expo_y == (unsigned)-1) {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while (!(yy & 0x80000000)) {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }

    if ((expo_z == (unsigned)-1) && ((zz << 1) != 0)) {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while (!(zz & 0x80000000)) {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  product = ((unsigned long long)xx) * yy;
  xx = (unsigned)(product >> 32);
  yy = (unsigned)(product & 0xffffffff);

  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;

  if (xx < 0x00800000) {
    xx = (xx << 1) | (yy >> 31);
    yy = (yy << 1);
    expo_x--;
  }
  temp = 0;

  if ((zz << 1) != 0) {
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;

    if ((int)expo_z > (int)expo_x) {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }


    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 49) {

      temp = 0;
      while (expo_z >= 32) {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        temp = ((temp >> expo_z) | (ww << (32 - expo_z)) |
                ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }

    } else {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if ((int)u < 0) {

      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if (!(xx | yy | temp)) {

        if (mode == cudaRoundMinInf) {
          xx = 0x80000000;
        }
        cvt.i = xx;
        return cvt.f;
      }
      if ((int)xx < 0) {


        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if (++temp == 0) {
          if (++yy == 0) {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }

      while (!(xx & 0x00800000)) {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    } else {

      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if (xx & 0x01000000) {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if (expo_x <= 0xFD) {

    xx |= expo_y;
    if (mode == cudaRoundNearest) {
      s = xx & 1;
      xx += (temp == 0x80000000) ? s : (temp >> 31);
    } else if (mode == cudaRoundPosInf) {
      xx += temp && !expo_y;
    } else if (mode == cudaRoundMinInf) {
      xx += temp && expo_y;
    }
    xx = xx + (expo_x << 23);
    cvt.i = xx;
    return cvt.f;
  } else if ((int)expo_x >= 126) {

    if (mode == cudaRoundNearest) {
      xx = expo_y | 0x7f800000;
    } else if (mode == cudaRoundZero) {
      xx = expo_y | 0x7F7FFFFF;
    } else if (mode == cudaRoundPosInf) {
      xx = expo_y ? 0xFF7FFFFF : 0x7f800000;
    } else if (mode == cudaRoundMinInf) {
      xx = expo_y ? 0xff800000 : 0x7f7fffff;
    }
    cvt.i = xx;
    return cvt.f;
  }

  expo_x = (unsigned int)(-(int)expo_x);
  if (expo_x > 25) {

    xx = 0;
    if (mode == cudaRoundPosInf) {
      xx += !expo_y;
    } else if (mode == cudaRoundMinInf) {
      xx += !!expo_y;
    }
    cvt.i = expo_y | xx;
    return cvt.f;
  }
  temp = (xx << (32 - expo_x)) | ((temp) ? 1 : 0);
  xx = xx >> expo_x;
  if (mode == cudaRoundNearest) {
    xx = xx + ((temp == 0x80000000) ? (xx & 1) : (temp >> 31));
  } else if (mode == cudaRoundPosInf) {
    xx = xx + (!expo_y && temp);
  } else if (mode == cudaRoundMinInf) {
    xx = xx + (expo_y && temp);
  }
  xx = expo_y + xx;
  cvt.i = xx;
  return cvt.f;
}


static float __internal_fadd_kernel2 (float a, float b, enum cudaRoundMode mode)

{
  volatile union __cudart_FloatUintCvt xx, yy;
  unsigned int expo_x;
  unsigned int expo_y;
  unsigned int temp;

  xx.f = a;
  yy.f = b;


  expo_y = yy.i << 1;
  if (expo_y > (xx.i << 1)) {
    expo_y = xx.i;
    xx.i = yy.i;
    yy.i = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {
add:
    expo_y = expo_x - expo_y;
    if (expo_y > 25) {
      expo_y = 31;
    }
    temp = xx.i ^ yy.i;
    xx.i = xx.i & ~0x7f000000;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i & ~0xff000000;
    yy.i = yy.i | 0x00800000;

    if ((int)temp < 0) {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      temp = (unsigned)(-((int)temp));
      xx.i = xx.i - (yy.i >> expo_y) - (temp ? 1 : 0);
      if (xx.i & 0x00800000) {
        if (expo_x <= 0xFD) {
          xx.i = xx.i + (expo_x << 23);
          if (mode == cudaRoundMinInf) {
            xx.i += (temp && (xx.i & 0x80000000));
          } else if (mode == cudaRoundPosInf) {
            xx.i += (temp && !(xx.i & 0x80000000));
          }
          return xx.f;
        }
      } else {
        if ((temp | (xx.i << 1)) == 0) {

          if (mode == cudaRoundMinInf) {
            xx.i = 0x80000000;
          } else if (mode == cudaRoundPosInf) {
            xx.i = 0;
          }
          return xx.f;
        }

        yy.i = xx.i & 0x80000000;
        do {
          xx.i = (xx.i << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while (!(xx.i & 0x00800000));
        xx.i = xx.i | yy.i;
      }
    } else {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      xx.i = xx.i + (yy.i >> expo_y);
      if (!(xx.i & 0x01000000)) {
        if (expo_x <= 0xFD) {
          xx.i = xx.i + (expo_x << 23);
          if (mode == cudaRoundMinInf) {
            xx.i += (temp && (xx.i & 0x80000000));
          } else if (mode == cudaRoundPosInf) {
            xx.i += (temp && !(xx.i & 0x80000000));
          }
          return xx.f;
        }
      } else {

        temp = (xx.i << 31) | (temp >> 1);
        xx.i = ((xx.i & 0x80000000) | (xx.i >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if (expo_x <= 0xFD) {
      if (mode == cudaRoundMinInf) {
        xx.i += (temp && (xx.i & 0x80000000));
      } else if (mode == cudaRoundPosInf) {
        xx.i += (temp && !(xx.i & 0x80000000));
      }
      xx.i = xx.i + (expo_x << 23);
      return xx.f;
    }
    if ((int)expo_x >= 254) {

      temp = xx.i & 0x80000000;
      if (mode == cudaRoundMinInf) {
        xx.i = (temp ? 0xFF800000 : 0x7f7fffff);
      } else if (mode == cudaRoundPosInf) {
        xx.i = (temp ? 0xff7fffff : 0x7F800000);
      }
      return xx.f;
    }

    expo_y = expo_x + 32;
    yy.i = xx.i & 0x80000000;
    xx.i = xx.i & ~0xff000000;
    expo_x = (unsigned)(-((int)expo_x));
    temp = xx.i << expo_y | ((temp) ? 1 : 0);
    xx.i = yy.i | (xx.i >> expo_x);
    if (mode == cudaRoundMinInf) {
      xx.i += (temp && yy.i);
    } else if (mode == cudaRoundPosInf) {
      xx.i += (temp && !yy.i);
    }
    return xx.f;
  } else {

    if (!(yy.i << 1)) {
      if (mode == cudaRoundMinInf) {
        if (!(xx.i << 1)) {
          xx.i = xx.i | yy.i;
        }
      } else if (mode == cudaRoundPosInf) {
        if (xx.i == 0x80000000) {
          xx.i = yy.i;
        }
      }
      if ((xx.i << 1) > 0xff000000) {
        xx.i |= 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {

      if (expo_x == (unsigned int) -1) {
        temp = xx.i & 0x80000000;
        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        expo_x++;
        xx.i = (xx.i >> 8) | temp;
      }
      if (expo_y == (unsigned int) -1) {
        temp = yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        expo_y++;
        yy.i = (yy.i >> 8) | temp;
      }
      goto add;
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xff000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xff000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    if ((expo_x == 0xff000000) && (expo_y == 0xff000000)) {




      expo_x = xx.i ^ yy.i;
      xx.i = xx.i | ((expo_x) ? 0xffc00000 : 0);
      return xx.f;
    }

    if (expo_y == 0xff000000) {
      xx.i = yy.i;
    }
    return xx.f;
  }
}

static float __frcp_rn (float a)
{
  return __internal_frcp_kernel (a, cudaRoundNearest);
}

static float __frcp_rz (float a)
{
  return __internal_frcp_kernel (a, cudaRoundZero);
}

static float __frcp_rd (float a)
{
  return __internal_frcp_kernel (a, cudaRoundMinInf);
}

static float __frcp_ru (float a)
{
  return __internal_frcp_kernel (a, cudaRoundPosInf);
}

static float __fsqrt_rn (float a)
{
  return __internal_fsqrt_kernel (a, cudaRoundNearest);
}

static float __fsqrt_rz (float a)
{
  return __internal_fsqrt_kernel (a, cudaRoundZero);
}

static float __fsqrt_rd (float a)
{
  return __internal_fsqrt_kernel (a, cudaRoundMinInf);
}

static float __fsqrt_ru (float a)
{
  return __internal_fsqrt_kernel (a, cudaRoundPosInf);
}

static float __fdiv_rn (float a, float b)
{
  return __internal_fdiv_kernel (a, b, cudaRoundNearest);
}

static float __fdiv_rz (float a, float b)
{
  return __internal_fdiv_kernel (a, b, cudaRoundZero);
}

static float __fdiv_rd (float a, float b)
{
  return __internal_fdiv_kernel (a, b, cudaRoundMinInf);
}

static float __fdiv_ru (float a, float b)
{
  return __internal_fdiv_kernel (a, b, cudaRoundPosInf);
}

static float __fadd_rd (float a, float b)
{
  return __internal_fadd_kernel2 (a, b, cudaRoundMinInf);
}

static float __fadd_ru (float a, float b)
{
  return __internal_fadd_kernel2 (a, b, cudaRoundPosInf);
}

static float __fmul_rd (float a, float b)
{
  return __internal_fmul_kernel2 (a, b, cudaRoundMinInf);
}

static float __fmul_ru (float a, float b)
{
  return __internal_fmul_kernel2 (a, b, cudaRoundPosInf);
}

static float __fmaf_rn (float a, float b, float c)
{
  return __internal_fmaf_kernel (a, b, c, cudaRoundNearest);
}

static float __fmaf_rz (float a, float b, float c)
{
  return __internal_fmaf_kernel (a, b, c, cudaRoundZero);
}

static float __fmaf_ru (float a, float b, float c)
{
  return __internal_fmaf_kernel (a, b, c, cudaRoundPosInf);
}

static float __fmaf_rd (float a, float b, float c)
{
  return __internal_fmaf_kernel (a, b, c, cudaRoundMinInf);
}

static int __cuda___isnan(double a);
static int __cuda___isnanf(float a);
static int __double2int_rz(double);
static unsigned int __double2uint_rz(double);
static long long int __double2ll_rz(double);
static unsigned long long int __double2ull_rz(double);
# 3737 "/usr/local/cuda/bin/../include/device_functions.h" 3
static int __mulhi(int a, int b)
{
  long long int c = (long long int)a * (long long int)b;

  return (int)(c >> 32);
}

static unsigned int __umulhi(unsigned int a, unsigned int b)
{
  unsigned long long int c = (unsigned long long int)a * (unsigned long long int)b;

  return (unsigned int)(c >> 32);
}

static unsigned long long int __umul64hi(unsigned long long int a, unsigned long long int b)
{
  unsigned int a_lo = (unsigned int)a;
  unsigned long long int a_hi = a >> 32;
  unsigned int b_lo = (unsigned int)b;
  unsigned long long int b_hi = b >> 32;
  unsigned long long int m1 = a_lo * b_hi;
  unsigned long long int m2 = a_hi * b_lo;
  unsigned int carry;

  carry = (0ULL + __umulhi(a_lo, b_lo) + (unsigned int)m1 + (unsigned int)m2) >> 32;

  return a_hi * b_hi + (m1 >> 32) + (m2 >> 32) + carry;
}

static long long int __mul64hi(long long int a, long long int b)
{
  long long int res;
  res = __umul64hi(a, b);
  if (a < 0LL) res = res - b;
  if (b < 0LL) res = res - a;
  return res;
}

static float __saturatef(float a)
{
  if (__cuda___isnanf(a)) return 0.0f;
  return a >= 1.0f ? 1.0f : a <= 0.0f ? 0.0f : a;
}

static unsigned int __sad(int a, int b, unsigned int c)
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

static unsigned int __usad(unsigned int a, unsigned int b, unsigned int c)
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

static int __mul24(int a, int b)
{

  a &= 0xffffff;
  a = (a & 0x800000) != 0 ? a | ~0xffffff : a;
  b &= 0xffffff;
  b = (b & 0x800000) != 0 ? b | ~0xffffff : b;


  return a * b;
}

static unsigned int __umul24(unsigned int a, unsigned int b)
{

  a &= 0xffffff;
  b &= 0xffffff;


  return a * b;
}

static float __int_as_float(int a)
{
  volatile union __cudart_FloatIntCvt u;

  u.i = a;
  return u.f;
}

static int __float_as_int(float a)
{
  volatile union __cudart_FloatIntCvt u;

  u.f = a;
  return u.i;
}

static long long int __internal_float2ll_kernel(float a, long long int max, long long int min, long long int nan, enum cudaRoundMode rndMode)
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= min) return min;
  ia = __float_as_int(a);
  shift = 189 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)(((ia << 8) | 0x80000000) >> 1) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundMinInf && t != 0ULL && ia > 0x80000000) {
    res++;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL && (int)ia > 0) {
    res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)-(long long int)res;
  return (long long int)res;
}

static int __internal_float2int(float a, enum cudaRoundMode rndMode)
{
  return (int)__internal_float2ll_kernel(a, 2147483647LL, -2147483648LL, 0LL, rndMode);
}

static int __float2int_rz(float a)
{



  return __internal_float2int(a, cudaRoundZero);

}

static int __float2int_ru(float a)
{
  return __internal_float2int(a, cudaRoundPosInf);
}

static int __float2int_rd(float a)
{
  return __internal_float2int(a, cudaRoundMinInf);
}

static int __float2int_rn(float a)
{
  return __internal_float2int(a, cudaRoundNearest);
}

static long long int __internal_float2ll(float a, enum cudaRoundMode rndMode)
{
  return __internal_float2ll_kernel(a, 9223372036854775807LL, -9223372036854775807LL -1LL, -9223372036854775807LL -1LL, rndMode);
}

static long long int __float2ll_rz(float a)
{



  return __internal_float2ll(a, cudaRoundZero);

}

static long long int __float2ll_ru(float a)
{
  return __internal_float2ll(a, cudaRoundPosInf);
}

static long long int __float2ll_rd(float a)
{
  return __internal_float2ll(a, cudaRoundMinInf);
}

static long long int __float2ll_rn(float a)
{
  return __internal_float2ll(a, cudaRoundNearest);
}

static unsigned long long int __internal_float2ull_kernel(float a, unsigned long long int max, unsigned long long int nan, enum cudaRoundMode rndMode)
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= 0LL) return 0LL;
  ia = __float_as_int(a);
  shift = 190 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)((ia << 8) | 0x80000000) << 32;
  if (shift >= 64) {
    t = res >> (int)(shift > 64);
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL) {
    res++;
  }
  return res;
}

static unsigned int __internal_float2uint(float a, enum cudaRoundMode rndMode)
{
  return (unsigned int)__internal_float2ull_kernel(a, 4294967295U, 0U, rndMode);
}

static unsigned int __float2uint_rz(float a)
{



  return __internal_float2uint(a, cudaRoundZero);

}

static unsigned int __float2uint_ru(float a)
{
  return __internal_float2uint(a, cudaRoundPosInf);
}

static unsigned int __float2uint_rd(float a)
{
  return __internal_float2uint(a, cudaRoundMinInf);
}

static unsigned int __float2uint_rn(float a)
{
  return __internal_float2uint(a, cudaRoundNearest);
}

static unsigned long long int __internal_float2ull(float a, enum cudaRoundMode rndMode)
{
  return __internal_float2ull_kernel(a, 18446744073709551615ULL, 9223372036854775808ULL, rndMode);
}

static unsigned long long int __float2ull_rz(float a)
{



  return __internal_float2ull(a, cudaRoundZero);

}

static unsigned long long int __float2ull_ru(float a)
{
  return __internal_float2ull(a, cudaRoundPosInf);
}

static unsigned long long int __float2ull_rd(float a)
{
  return __internal_float2ull(a, cudaRoundMinInf);
}

static unsigned long long int __float2ull_rn(float a)
{
  return __internal_float2ull(a, cudaRoundNearest);
}

static int __internal_normalize64(unsigned long long int *a)
{
  int lz = 0;

  if ((*a & 0xffffffff00000000ULL) == 0ULL) {
    *a <<= 32;
    lz += 32;
  }
  if ((*a & 0xffff000000000000ULL) == 0ULL) {
    *a <<= 16;
    lz += 16;
  }
  if ((*a & 0xff00000000000000ULL) == 0ULL) {
    *a <<= 8;
    lz += 8;
  }
  if ((*a & 0xf000000000000000ULL) == 0ULL) {
    *a <<= 4;
    lz += 4;
  }
  if ((*a & 0xC000000000000000ULL) == 0ULL) {
    *a <<= 2;
    lz += 2;
  }
  if ((*a & 0x8000000000000000ULL) == 0ULL) {
    *a <<= 1;
    lz += 1;
  }
  return lz;
}

static int __internal_normalize(unsigned int *a)
{
  unsigned long long int t = (unsigned long long int)*a;
  int lz = __internal_normalize64(&t);

  *a = (unsigned int)(t >> 32);

  return lz - 32;
}

static float __internal_int2float_kernel(int a, enum cudaRoundMode rndMode)
{
  volatile union __cudart_FloatUintCvt res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  if (a < 0) res.i = (unsigned int)-a;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if (a < 0) res.i |= 0x80000000;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundMinInf) && t && (a < 0)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && (a > 0)) {
    res.i++;
  }
  return res.f;
}

static float __int2float_rz(int a)
{
  return __internal_int2float_kernel(a, cudaRoundZero);
}

static float __int2float_ru(int a)
{
  return __internal_int2float_kernel(a, cudaRoundPosInf);
}

static float __int2float_rd(int a)
{
  return __internal_int2float_kernel(a, cudaRoundMinInf);
}

static float __int2float_rn(int a)
{



  return __internal_int2float_kernel(a, cudaRoundNearest);

}

static float __internal_uint2float_kernel(unsigned int a, enum cudaRoundMode rndMode)
{
  volatile union __cudart_FloatUintCvt res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.f;
}

static float __uint2float_rz(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundZero);
}

static float __uint2float_ru(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundPosInf);
}

static float __uint2float_rd(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundMinInf);
}

static float __uint2float_rn(unsigned int a)
{



  return __internal_uint2float_kernel(a, cudaRoundNearest);

}

static float __ll2float_rn(long long int a)
{
  return (float)a;
}

static float __ull2float_rn(unsigned long long int a)
{



  unsigned long long int temp;
  unsigned int res, t;
  int shift;
  if (a == 0ULL) return 0.0f;
  temp = a;
  shift = __internal_normalize64(&temp);
  temp = (temp >> 8) | ((temp & 0xffULL) ? 1ULL : 0ULL);
  res = (unsigned int)(temp >> 32);
  t = (unsigned int)temp;
  res += (127 + 62 - shift) << 23;
  res += t == 0x80000000 ? res & 1 : t >> 31;
  return __int_as_float(res);

}

static float __internal_fmul_kernel(float a, float b, int rndNearest)
{
  unsigned long long product;
  volatile union __cudart_FloatUintCvt xx, yy;
  unsigned expo_x, expo_y;

  xx.f = a;
  yy.f = b;

  expo_y = 0xFF;
  expo_x = expo_y & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {
multiply:
    expo_x = expo_x + expo_y;
    expo_y = xx.i ^ yy.i;
    xx.i = xx.i & 0x00ffffff;
    yy.i = yy.i << 8;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i | 0x80000000;

    product = ((unsigned long long)xx.i) * yy.i;
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xx.i = (unsigned int)(product >> 32);
    yy.i = (unsigned int)(product & 0xffffffff);

    if (xx.i < 0x00800000) {
      xx.i = (xx.i << 1) | (yy.i >> 31);
      yy.i = (yy.i << 1);
      expo_x--;
    }
    if (expo_x <= 0xFD) {
      xx.i = xx.i | expo_y;
      xx.i = xx.i + (expo_x << 23);

      if (yy.i < 0x80000000) return xx.f;
      xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31))
               && rndNearest);
      return xx.f;
    } else if ((int)expo_x >= 254) {

      xx.i = (expo_y | 0x7F800000) - (!rndNearest);
      return xx.f;
    } else {

      expo_x = ((unsigned int)-((int)expo_x));
      if (expo_x > 25) {

        xx.i = expo_y;
        return xx.f;
      } else {
        yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
        xx.i = expo_y + (xx.i >> expo_x);
        xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31))
                 && rndNearest);
        return xx.f;
      }
    }
  } else {
    product = xx.i ^ yy.i;
    product = product & 0x80000000;
    if (!(xx.i & 0x7fffffff)) {
      if (expo_y != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_y = yy.i << 1;
      if (expo_y == 0xFF000000) {
        xx.i = expo_y | 0x00C00000;
      } else {
        xx.i = yy.i | 0x00400000;
      }
      return xx.f;
    }
    if (!(yy.i & 0x7fffffff)) {
      if (expo_x != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_x = xx.i << 1;
      if (expo_x == 0xFF000000) {
        xx.i = expo_x | 0x00C00000;
      } else {
        xx.i = xx.i | 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {
      expo_y++;
      expo_x++;
      if (expo_x == 0) {
        expo_y |= xx.i & 0x80000000;




        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        xx.i = (xx.i >> 8) | (expo_y & 0x80000000);
        expo_y &= ~0x80000000;
        expo_y--;
        goto multiply;
      }
      if (expo_y == 0) {
        expo_x |= yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        yy.i = (yy.i >> 8) | (expo_x & 0x80000000);
        expo_x &= ~0x80000000;
        expo_x--;
        goto multiply;
      }
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xFF000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xFF000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    xx.i = (unsigned int)product | 0x7f800000;
    return xx.f;
  }
}

static float __internal_fadd_kernel(float a, float b, int rndNearest)
{
  volatile union __cudart_FloatUintCvt xx, yy;
  unsigned int expo_x;
  unsigned int expo_y;
  unsigned int temp;

  xx.f = a;
  yy.f = b;


  expo_y = yy.i << 1;
  if (expo_y > (xx.i << 1)) {
    expo_y = xx.i;
    xx.i = yy.i;
    yy.i = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {

add:
    expo_y = expo_x - expo_y;
    if (expo_y > 25) {
      expo_y = 31;
    }
    temp = xx.i ^ yy.i;
    xx.i = xx.i & ~0x7f000000;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i & ~0xff000000;
    yy.i = yy.i | 0x00800000;

    if ((int)temp < 0) {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xx.i = xx.i - (yy.i >> expo_y) - (temp ? 1 : 0);
      if (xx.i & 0x00800000) {
        if (expo_x <= 0xFD) {
          xx.i = xx.i & ~0x00800000;
          xx.i = (xx.i + (expo_x << 23)) + 0x00800000;
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31))
                   && rndNearest);
          return xx.f;
        }
      } else {
        if ((temp | (xx.i << 1)) == 0) {

          xx.i = 0;
          return xx.f;
        }

        yy.i = xx.i & 0x80000000;
        do {
          xx.i = (xx.i << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while (!(xx.i & 0x00800000));
        xx.i = xx.i | yy.i;
      }
    } else {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      xx.i = xx.i + (yy.i >> expo_y);
      if (!(xx.i & 0x01000000)) {
        if (expo_x <= 0xFD) {
          expo_y = xx.i & 1;
          xx.i = xx.i + (expo_x << 23);
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31))
                   && rndNearest);
          return xx.f;
        }
      } else {

        temp = (xx.i << 31) | (temp >> 1);

        xx.i = ((xx.i & 0x80000000) | (xx.i >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if (expo_x <= 0xFD) {
      expo_y = xx.i & 1;
      xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31))
               && rndNearest);
      xx.i = xx.i + (expo_x << 23);
      return xx.f;
    }
    if ((int)expo_x >= 254) {

        xx.i = ((xx.i & 0x80000000) | 0x7f800000) - (!rndNearest);
        return xx.f;
    }

    expo_y = expo_x + 32;
    yy.i = xx.i & 0x80000000;
    xx.i = xx.i & ~0xff000000;

    expo_x = (unsigned int)(-((int)expo_x));
    temp = xx.i << expo_y | ((temp) ? 1 : 0);
    xx.i = yy.i | (xx.i >> expo_x);
    xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31))
             && rndNearest);
    return xx.f;
  } else {

    if (!(yy.i << 1)) {
      if (xx.i == 0x80000000) {
        xx.i = yy.i;
      }
      if ((xx.i << 1) > 0xff000000) {
        xx.i |= 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {

      if (expo_x == (unsigned int) -1) {
        temp = xx.i & 0x80000000;
        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        expo_x++;
        xx.i = (xx.i >> 8) | temp;
      }
      if (expo_y == (unsigned int) -1) {
        temp = yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        expo_y++;
        yy.i = (yy.i >> 8) | temp;
      }
      goto add;
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xff000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xff000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    if ((expo_x == 0xff000000) && (expo_y == 0xff000000)) {




      expo_x = xx.i ^ yy.i;
      xx.i = xx.i | ((expo_x) ? 0xffc00000 : 0);
      return xx.f;
    }

    if (expo_y == 0xff000000) {
      xx.i = yy.i;
    }
    return xx.f;
  }
}

static float __fadd_rz(float a, float b)
{
  return __internal_fadd_kernel(a, b, 0);
}

static float __fmul_rz(float a, float b)
{
  return __internal_fmul_kernel(a, b, 0);
}

static float __fadd_rn(float a, float b)
{
  return __internal_fadd_kernel(a, b, 1);
}

static float __fmul_rn(float a, float b)
{
  return __internal_fmul_kernel(a, b, 1);
}

static void __brkpt(int c)
{

}
# 4518 "/usr/local/cuda/bin/../include/device_functions.h" 3
extern int __cudaSynchronizeThreads(void**, void*);



static inline __attribute__((always_inline)) void __syncthreads(void)
{
  volatile int _ = 0;
  L: if (__cudaSynchronizeThreads((void**)&&L, (void*)&_)) goto L;
}
# 4537 "/usr/local/cuda/bin/../include/device_functions.h" 3
static void __prof_trigger(int a)
{
}

static void __threadfence(void)
{
}

static void __threadfence_block(void)
{
}



static void __trap(void)
{
  __builtin_trap();
}
# 4573 "/usr/local/cuda/bin/../include/device_functions.h" 3
static float __fdividef(float a, float b)
{






  if (__cuda_fabsf(b) > 8.507059173e37f) {
    if (__cuda_fabsf(a) <= 3.402823466e38f) {
      return ((a / b) / 3.402823466e38f) / 3.402823466e38f;
    } else {
      return __int_as_float(0xffc00000);
    }
  } else {
    return a / b;
  }

}

static float __sinf(float a)
{
  return sinf(a);
}

static float __cosf(float a)
{
  return cosf(a);
}

static float __log2f(float a)
{
  return log2f(a);
}







static float __internal_accurate_fdividef(float a, float b)
{
  if (__cuda_fabsf(b) > 8.507059173e37f) {
    a *= .25f;
    b *= .25f;
  }
  return __fdividef(a, b);
}

static float __tanf(float a)
{



  return __fdividef (__sinf(a), __cosf(a));

}

static void __sincosf(float a, float *sptr, float *cptr)
{



  *sptr = __sinf(a);
  *cptr = __cosf(a);

}

static float __expf(float a)
{



  return __cuda_exp2f(a * 1.442695041f);

}

static float __exp10f(float a)
{



  return __cuda_exp2f(a * 3.321928094f);

}

static float __log10f(float a)
{



  return 0.301029996f * __log2f(a);

}

static float __logf(float a)
{



  return 0.693147181f * __log2f(a);

}

static float __powf(float a, float b)
{



  return __cuda_exp2f(b * __log2f(a));

}

static float fdividef(float a, float b)
{





  return __internal_accurate_fdividef(a, b);

}

static int __clz(int a)
{
  return (a)?(158-(__float_as_int(__uint2float_rz((unsigned int)a))>>23)):32;
}

static int __clzll(long long int a)
{
  int ahi = ((int)((unsigned long long)a >> 32));
  int alo = ((int)((unsigned long long)a & 0xffffffffULL));
  int res;
  if (ahi) {
    res = 0;
  } else {
    res = 32;
    ahi = alo;
  }
  res = res + __clz(ahi);
  return res;
}

static int __popc(unsigned int a)
{
  a = a - ((a >> 1) & 0x55555555);
  a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
  a = (a + (a >> 4)) & 0x0f0f0f0f;
  a = ((__umul24(a, 0x808080) << 1) + a) >> 24;
  return a;
}

static int __popcll(unsigned long long int a)
{
  unsigned int ahi = ((unsigned int)(a >> 32));
  unsigned int alo = ((unsigned int)(a & 0xffffffffULL));
  alo = alo - ((alo >> 1) & 0x55555555);
  alo = (alo & 0x33333333) + ((alo >> 2) & 0x33333333);
  ahi = ahi - ((ahi >> 1) & 0x55555555);
  ahi = (ahi & 0x33333333) + ((ahi >> 2) & 0x33333333);
  alo = alo + ahi;
  alo = (alo & 0x0f0f0f0f) + ((alo >> 4) & 0x0f0f0f0f);
  alo = ((__umul24(alo, 0x808080) << 1) + alo) >> 24;
  return alo;
}

static unsigned int __brev(unsigned int a)
{
  a = ((a >> 1) & 0x55555555) + ((a & 0x55555555) << 1);
  a = ((a >> 2) & 0x33333333) + ((a & 0x33333333) << 2);
  a = ((a >> 4) & 0x0F0F0F0F) + ((a & 0x0F0F0F0F) << 4);
  a = ((a >> 8) & 0x00FF00FF) + ((a & 0x00FF00FF) << 8);
  a = ( a >> 16 ) + ( a << 16);
  return a;
}

static unsigned long long int __brevll(unsigned long long int a)
{
  unsigned int hi = (unsigned int)(a >> 32);
  unsigned int lo = (unsigned int)(a & 0xffffffffULL);
  unsigned int t;
  t = __brev(lo);
  lo = __brev(hi);
  return ((unsigned long long int)t << 32) + (unsigned long long int)lo;
}

static int __ffs(int a)
{
  return 32 - __clz (a & -a);
}

static int __ffsll(long long int a)
{
  return 64 - __clzll (a & -a);
}
# 4812 "/usr/local/cuda/bin/../include/device_functions.h" 3
static double fdivide(double a, double b)
{
  return a / b;
}



static int __internal_double2int(double a, enum cudaRoundMode rndMode);
static unsigned int __internal_double2uint(double a, enum cudaRoundMode rndMode);
static long long int __internal_double2ll(double a, enum cudaRoundMode rndMode);
static unsigned long long int __internal_double2ull(double a, enum cudaRoundMode rndMode);

static int __double2int_rz(double a)
{
  return __internal_double2int(a, cudaRoundZero);
}

static unsigned int __double2uint_rz(double a)
{
  return __internal_double2uint(a, cudaRoundZero);
}

static long long int __double2ll_rz(double a)
{
  return __internal_double2ll(a, cudaRoundZero);
}

static unsigned long long int __double2ull_rz(double a)
{
  return __internal_double2ull(a, cudaRoundZero);
}
# 4856 "/usr/local/cuda/bin/../include/device_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/sm_11_atomic_functions.h" 1 3
# 257 "/usr/local/cuda/bin/../include/sm_11_atomic_functions.h" 3
extern void __cudaMutexOperation(int lock);






static int __iAtomicAdd(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old + val; __cudaMutexOperation(0);



  return old;
}

static unsigned int __uAtomicAdd(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old + val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicExch(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicExch(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = val; __cudaMutexOperation(0);




  return old;
}

static float __fAtomicExch(float *address, float val)
{
  float old;

  __cudaMutexOperation(1); old = *address; *address = val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicMin(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old < val ? old : val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicMin(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old < val ? old : val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicMax(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old > val ? old : val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicMax(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old > val ? old : val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicInc(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = (old >= val) ? 0 : old + 1; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicDec(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = ((old == 0) | (old > val)) ? val : (old - 1); __cudaMutexOperation(0);




  return old;
}

static int __iAtomicAnd(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old & val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicAnd(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old & val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicOr(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old | val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicOr(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old | val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicXor(int *address, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old ^ val; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicXor(unsigned int *address, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old ^ val; __cudaMutexOperation(0);




  return old;
}

static int __iAtomicCAS(int *address, int compare, int val)
{
  int old;

  __cudaMutexOperation(1); old = *address; *address = old == compare ? val : old; __cudaMutexOperation(0);




  return old;
}

static unsigned int __uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  unsigned int old;

  __cudaMutexOperation(1); old = *address; *address = old == compare ? val : old; __cudaMutexOperation(0);




  return old;
}
# 4857 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/sm_12_atomic_functions.h" 1 3
# 118 "/usr/local/cuda/bin/../include/sm_12_atomic_functions.h" 3
extern void __cudaMutexOperation(int lock);






static unsigned long long int __ullAtomicAdd(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old;

  __cudaMutexOperation(1); old = *address; *address = old + val; __cudaMutexOperation(0);



  return old;
}

static unsigned long long int __ullAtomicExch(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old;

  __cudaMutexOperation(1); old = *address; *address = val; __cudaMutexOperation(0);



  return old;
}

static unsigned long long int __ullAtomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val)
{
  unsigned long long int old;

  __cudaMutexOperation(1); old = *address; *address = old == compare ? val : old; __cudaMutexOperation(0);



  return old;
}





static int __any(int cond)
{
  return cond;
}

static int __all(int cond)
{
  return cond;
}
# 4858 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 1 3
# 267 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 3
static double __longlong_as_double(long long int a)
{
  volatile union __cudart_DoubleLonglongCvt u;
  u.i = a;
  return u.d;
}

static long long int __double_as_longlong(double a)
{
  volatile union __cudart_DoubleLonglongCvt u;
  u.d = a;
  return u.i;
}

static float __internal_double2float_kernel(double a)
{
  volatile union __cudart_DoubleUlonglongCvt xx;
  volatile union __cudart_FloatUintCvt res;

  int shift;
  xx.d = a;
  if (xx.i == 0) return 0.0f;
  res.i = (((unsigned int) (xx.i >> 32)) & 0x80000000);
  if ((xx.i & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) {
    if ((xx.i & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {

      res.i = 0x7f8fffff;
    } else {

      res.i |= 0x7f800000;
    }
    return res.f;
  }
  shift = ((int) ((xx.i >> 52) & 0x7ff)) - 1023;

  xx.i = (xx.i & 0x000fffffffffffffULL);
  if (shift >= 128) {
    res.i |= 0x7f7fffff;
    return res.f;
  }
  if (shift <= -127) {
    if (shift < -180) {

      xx.i = 0;
    } else {
      xx.i |= 0x0010000000000000ULL;
      xx.i >>= 127 + shift;
    }
  } else {
    res.i |= (unsigned int) (127 + shift) << 23;
  }
  res.i |= ((unsigned int) (xx.i >> 29)) & 0x007fffff;
  xx.i &= 0x1fffffff;
  return res.f;
}

static double __internal_ll2double_kernel(long long int a, enum cudaRoundMode rndMode)
{
  volatile union __cudart_DoubleUlonglongCvt res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.d;
  if (a < 0) res.i = (unsigned long long int)-a;
  shift = __internal_normalize64((unsigned long long int*)&res.i);
  t = ((unsigned int) res.i) << 21;
  res.i >>= 11;
  res.i += ((unsigned long long int)(1023 + 62 - shift)) << 52;
  if (a < 0) res.i |= 0x8000000000000000ULL;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundMinInf) && t && (a < 0)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && (a > 0)) {
    res.i++;
  }
  return res.d;
}

static double __internal_ull2double_kernel(unsigned long long int a, enum cudaRoundMode rndMode)
{
  volatile union __cudart_DoubleUlonglongCvt res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.d;
  shift = __internal_normalize64((unsigned long long int *)&res.i);
  t = ((unsigned int) res.i) << 21;
  res.i >>= 11;
  res.i += ((unsigned long long int)(1023 + 62 - shift)) << 52;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.d;
}

static long long int __internal_double2ll_kernel(double a, long long int max, long long int min, long long int nan, enum cudaRoundMode rndMode)
{
  volatile union __cudart_DoubleUlonglongCvt xx, res;
  unsigned long long int t = 0;
  int shift;

  xx.d = a;
  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= min) return min;
  shift = (int) (1023 + 62 - ((xx.i >> 52) & 0x7ff));
  res.i = ((xx.i << 11) | 0x8000000000000000ULL) >> 1;
  if (shift >= 64) {
    t = res.i;
    res.i = 0;
  } else if (shift) {
    t = res.i << (64 - shift);
    res.i = res.i >> shift;
  }
  if ((rndMode == cudaRoundNearest) && (t >= 0x8000000000000000ULL)) {
    res.i += (t == 0x8000000000000000ULL) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundMinInf) && t &&
          (xx.i > 0x8000000000000000ULL)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && ((long long int)xx.i > 0)) {
    res.i++;
  }
  if ((long long int)xx.i < 0) {
    res.i = (unsigned long long int)(-(long long int)res.i);
  }
  return res.i;
}

static unsigned long long int __internal_double2ull_kernel(double a, unsigned long long int max, unsigned long long int nan, enum cudaRoundMode rndMode)
{
  volatile union __cudart_DoubleUlonglongCvt xx, res;
  unsigned long long int t = 0;
  int shift;

  xx.d = a;
  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= 0LL) return 0LL;

  if (a == 0.0) return 0LL;
  shift = (int) (1023 + 63 - ((xx.i >> 52) & 0x7ff));
  res.i = ((xx.i << 11) | 0x8000000000000000ULL);
  if (shift >= 64) {
    t = res.i >> (int)(shift > 64);
    res.i = 0;
  } else if (shift) {
    t = res.i << (64 - shift);
    res.i = res.i >> shift;
  }
  if ((rndMode == cudaRoundNearest) && (t >= 0x8000000000000000ULL)) {
    res.i += (t == 0x8000000000000000ULL) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.i;
}

static int __double2hiint(double a)
{
  volatile union __cudart_DoubleInthiloCvt cvt;

  cvt.d = a;
  return cvt.i[1];
}

static int __double2loint(double a)
{
  volatile union __cudart_DoubleInthiloCvt cvt;

  cvt.d = a;
  return cvt.i[0];
}

static double __hiloint2double(int a, int b)
{
  volatile union __cudart_DoubleInthiloCvt cvt;

  cvt.i[0] = b;
  cvt.i[1] = a;
  return cvt.d;
}

static float __double2float_rn(double a)
{
  return (float)a;
}

static float __double2float_rz(double a)
{
  return __internal_double2float_kernel(a);
}

static int __internal_double2int(double a, enum cudaRoundMode rndMode)
{
  return (int)__internal_double2ll_kernel(a, 2147483647LL, -2147483648LL, -2147483648LL, rndMode);
}

static int __double2int_rn(double a)
{
  return __internal_double2int(a, cudaRoundNearest);
}

static int __double2int_ru(double a)
{
  return __internal_double2int(a, cudaRoundPosInf);
}

static int __double2int_rd(double a)
{
  return __internal_double2int(a, cudaRoundMinInf);
}

static unsigned int __internal_double2uint(double a, enum cudaRoundMode rndMode)
{
  return (unsigned int)__internal_double2ull_kernel(a, 4294967295ULL, 2147483648ULL, rndMode);
}

static unsigned int __double2uint_rn(double a)
{
  return __internal_double2uint(a, cudaRoundNearest);
}

static unsigned int __double2uint_ru(double a)
{
  return __internal_double2uint(a, cudaRoundPosInf);
}

static unsigned int __double2uint_rd(double a)
{
  return __internal_double2uint(a, cudaRoundMinInf);
}

static long long int __internal_double2ll(double a, enum cudaRoundMode rndMode)
{
  return __internal_double2ll_kernel(a, 9223372036854775807LL, -9223372036854775807LL -1LL, -9223372036854775807LL -1LL, rndMode);
}

static long long int __double2ll_rn(double a)
{
  return __internal_double2ll(a, cudaRoundNearest);
}

static long long int __double2ll_ru(double a)
{
  return __internal_double2ll(a, cudaRoundPosInf);
}

static long long int __double2ll_rd(double a)
{
  return __internal_double2ll(a, cudaRoundMinInf);
}

static unsigned long long int __internal_double2ull(double a, enum cudaRoundMode rndMode)
{
  return __internal_double2ull_kernel(a, 18446744073709551615ULL, 9223372036854775808ULL, rndMode);
}

static unsigned long long int __double2ull_rn(double a)
{
  return __internal_double2ull(a, cudaRoundNearest);
}

static unsigned long long int __double2ull_ru(double a)
{
  return __internal_double2ull(a, cudaRoundPosInf);
}

static unsigned long long int __double2ull_rd(double a)
{
  return __internal_double2ull(a, cudaRoundMinInf);
}

static double __int2double_rn(int a)
{
  return (double)a;
}

static double __uint2double_rn(unsigned int a)
{
  return (double)a;
}

static double __ll2double_rn(long long int a)
{
  return (double)a;
}

static double __ll2double_rz(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundZero);
}

static double __ll2double_rd(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundMinInf);
}

static double __ll2double_ru(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundPosInf);
}

static double __ull2double_rn(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundNearest);
}

static double __ull2double_rz(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundZero);
}

static double __ull2double_rd(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundMinInf);
}

static double __ull2double_ru(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundPosInf);
}





static double __internal_fma_kernel(double x, double y, double z, enum cudaRoundMode rndMode)
{



  struct __cudart_UintUint xx, yy, zz, ww;
  unsigned int s, t, u, prod0, prod1, prod2, prod3, expo_x, expo_y, expo_z;

  xx.hi = __double2hiint(x);
  xx.lo = __double2loint(x);
  yy.hi = __double2hiint(y);
  yy.lo = __double2loint(y);
  zz.hi = __double2hiint(z);
  zz.lo = __double2loint(z);

  expo_z = 0x7FF;
  t = xx.hi >> 20;
  expo_x = expo_z & t;
  expo_x = expo_x - 1;
  t = yy.hi >> 20;
  expo_y = expo_z & t;
  expo_y = expo_y - 1;
  t = zz.hi >> 20;
  expo_z = expo_z & t;
  expo_z = expo_z - 1;

  if (!((expo_x <= 0x7FD) &&
        (expo_y <= 0x7FD) &&
        (expo_z <= 0x7FD))) {





    if (((yy.hi << 1) | (yy.lo != 0)) > 0xffe00000) {
      yy.hi |= 0x00080000;
      return __hiloint2double(yy.hi, yy.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) > 0xffe00000) {
      zz.hi |= 0x00080000;
      return __hiloint2double(zz.hi, zz.lo);
    }
    if (((xx.hi << 1) | (xx.lo != 0)) > 0xffe00000) {
      xx.hi |= 0x00080000;
      return __hiloint2double(xx.hi, xx.lo);
    }
# 656 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 3
    if (((((xx.hi << 1) | xx.lo) == 0) &&
         (((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000)) ||
        ((((yy.hi << 1) | yy.lo) == 0) &&
         (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000))) {
      xx.hi = 0xfff80000;
      xx.lo = 0x00000000;
      return __hiloint2double(xx.hi, xx.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) == 0xffe00000) {
      if ((((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000) ||
          (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000)) {
        if ((int)(xx.hi ^ yy.hi ^ zz.hi) < 0) {
          xx.hi = 0xfff80000;
          xx.lo = 0x00000000;
          return __hiloint2double(xx.hi, xx.lo);
        }
      }
    }




    if (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000) {
      xx.hi = xx.hi ^ (yy.hi & 0x80000000);
      return __hiloint2double(xx.hi, xx.lo);
    }
    if (((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000) {
      yy.hi = yy.hi ^ (xx.hi & 0x80000000);
      return __hiloint2double(yy.hi, yy.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) == 0xffe00000) {
      return __hiloint2double(zz.hi, zz.lo);
    }





    if ((zz.hi == 0x80000000) && (zz.lo == 0)) {
      if ((((xx.hi << 1) | xx.lo) == 0) ||
          (((yy.hi << 1) | yy.lo) == 0)) {
        if ((int)(xx.hi ^ yy.hi) < 0) {
          return __hiloint2double(zz.hi, zz.lo);
        }
      }
    }



    if ((((zz.hi << 1) | zz.lo) == 0) &&
        ((((xx.hi << 1) | xx.lo) == 0) ||
         (((yy.hi << 1) | yy.lo) == 0))) {
      if (rndMode == cudaRoundMinInf) {
        return __hiloint2double((xx.hi ^ yy.hi ^ zz.hi) & 0x80000000, zz.lo);
      } else {
        zz.hi &= 0x7fffffff;
        return __hiloint2double(zz.hi, zz.lo);
      }
    }




    if ((((xx.hi << 1) | xx.lo) == 0) ||
        (((yy.hi << 1) | yy.lo) == 0)) {
      return __hiloint2double(zz.hi, zz.lo);
    }

    if (expo_x == 0xffffffff) {
      expo_x++;
      t = xx.hi & 0x80000000;
      s = xx.lo >> 21;
      xx.lo = xx.lo << 11;
      xx.hi = xx.hi << 11;
      xx.hi = xx.hi | s;
      if (!xx.hi) {
        xx.hi = xx.lo;
        xx.lo = 0;
        expo_x -= 32;
      }
      while ((int)xx.hi > 0) {
        s = xx.lo >> 31;
        xx.lo = xx.lo + xx.lo;
        xx.hi = xx.hi + xx.hi;
        xx.hi = xx.hi | s;
        expo_x--;
      }
      xx.lo = (xx.lo >> 11);
      xx.lo |= (xx.hi << 21);
      xx.hi = (xx.hi >> 11) | t;
    }
    if (expo_y == 0xffffffff) {
      expo_y++;
      t = yy.hi & 0x80000000;
      s = yy.lo >> 21;
      yy.lo = yy.lo << 11;
      yy.hi = yy.hi << 11;
      yy.hi = yy.hi | s;
      if (!yy.hi) {
        yy.hi = yy.lo;
        yy.lo = 0;
        expo_y -= 32;
      }
      while ((int)yy.hi > 0) {
        s = yy.lo >> 31;
        yy.lo = yy.lo + yy.lo;
        yy.hi = yy.hi + yy.hi;
        yy.hi = yy.hi | s;
        expo_y--;
      }
      yy.lo = (yy.lo >> 11);
      yy.lo |= (yy.hi << 21);
      yy.hi = (yy.hi >> 11) | t;
    }
    if (expo_z == 0xffffffff) {
      expo_z++;
      t = zz.hi & 0x80000000;
      s = zz.lo >> 21;
      zz.lo = zz.lo << 11;
      zz.hi = zz.hi << 11;
      zz.hi = zz.hi | s;
      if (!zz.hi) {
        zz.hi = zz.lo;
        zz.lo = 0;
        expo_z -= 32;
      }
      while ((int)zz.hi > 0) {
        s = zz.lo >> 31;
        zz.lo = zz.lo + zz.lo;
        zz.hi = zz.hi + zz.hi;
        zz.hi = zz.hi | s;
        expo_z--;
      }
      zz.lo = (zz.lo >> 11);
      zz.lo |= (zz.hi << 21);
      zz.hi = (zz.hi >> 11) | t;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx.hi ^ yy.hi;
  t = xx.lo >> 21;
  xx.lo = xx.lo << 11;
  xx.hi = xx.hi << 11;
  xx.hi = xx.hi | t;
  yy.hi = yy.hi & 0x000fffff;
  xx.hi = xx.hi | 0x80000000;
  yy.hi = yy.hi | 0x00100000;

  prod0 = xx.lo * yy.lo;
  prod1 = __umulhi (xx.lo, yy.lo);
  prod2 = xx.hi * yy.lo;
  prod3 = xx.lo * yy.hi;
  prod1 += prod2;
  t = prod1 < prod2;
  prod1 += prod3;
  t += prod1 < prod3;
  prod2 = __umulhi (xx.hi, yy.lo);
  prod3 = __umulhi (xx.lo, yy.hi);
  prod2 += prod3;
  s = prod2 < prod3;
  prod3 = xx.hi * yy.hi;
  prod2 += prod3;
  s += prod2 < prod3;
  prod2 += t;
  s += prod2 < t;
  prod3 = __umulhi (xx.hi, yy.hi) + s;

  yy.lo = prod0;
  yy.hi = prod1;
  xx.lo = prod2;
  xx.hi = prod3;
  expo_x = expo_x - (1023 - 2);
  expo_y = expo_y & 0x80000000;

  if (xx.hi < 0x00100000) {
    s = xx.lo >> 31;
    s = (xx.hi << 1) + s;
    xx.hi = s;
    s = yy.hi >> 31;
    s = (xx.lo << 1) + s;
    xx.lo = s;
    s = yy.lo >> 31;
    s = (yy.hi << 1) + s;
    yy.hi = s;
    s = yy.lo << 1;
    yy.lo = s;
    expo_x--;
  }

  t = 0;
  if (((zz.hi << 1) | zz.lo) != 0) {

    s = zz.hi & 0x80000000;

    zz.hi &= 0x000fffff;
    zz.hi |= 0x00100000;
    ww.hi = 0;
    ww.lo = 0;


    if ((int)expo_z > (int)expo_x) {
      t = expo_z;
      expo_z = expo_x;
      expo_x = t;
      t = zz.hi;
      zz.hi = xx.hi;
      xx.hi = t;
      t = zz.lo;
      zz.lo = xx.lo;
      xx.lo = t;
      t = ww.hi;
      ww.hi = yy.hi;
      yy.hi = t;
      t = ww.lo;
      ww.lo = yy.lo;
      yy.lo = t;
      t = expo_y;
      expo_y = s;
      s = t;
    }



    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 107) {

      t = 0;
      while (expo_z >= 32) {
        t = ww.lo | (t != 0);
        ww.lo = ww.hi;
        ww.hi = zz.lo;
        zz.lo = zz.hi;
        zz.hi = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        t = (t >> expo_z) | (ww.lo << (32 - expo_z)) |
                ((t << (32 - expo_z)) != 0);
        ww.lo = (ww.lo >> expo_z) | (ww.hi << (32 - expo_z));
        ww.hi = (ww.hi >> expo_z) | (zz.lo << (32 - expo_z));
        zz.lo = (zz.lo >> expo_z) | (zz.hi << (32 - expo_z));
        zz.hi = (zz.hi >> expo_z);
      }
    } else {
      t = 1;
      ww.lo = 0;
      ww.hi = 0;
      zz.lo = 0;
      zz.hi = 0;
    }
    if ((int)u < 0) {

      t = (unsigned)(-(int)t);
      s = (t != 0);
      u = yy.lo - s;
      s = u > yy.lo;
      yy.lo = u - ww.lo;
      s += yy.lo > u;
      u = yy.hi - s;
      s = u > yy.hi;
      yy.hi = u - ww.hi;
      s += yy.hi > u;
      u = xx.lo - s;
      s = u > xx.lo;
      xx.lo = u - zz.lo;
      s += xx.lo > u;
      xx.hi = (xx.hi - zz.hi) - s;
      if (!(xx.hi | xx.lo | yy.hi | yy.lo | t)) {

        if (rndMode == cudaRoundMinInf) {
          return __hiloint2double(0x80000000, xx.lo);
        } else {
          return __hiloint2double(xx.hi, xx.lo);
        }
      }
      if ((int)xx.hi < 0) {



        t = ~t;
        yy.lo = ~yy.lo;
        yy.hi = ~yy.hi;
        xx.lo = ~xx.lo;
        xx.hi = ~xx.hi;
        if (++t == 0) {
          if (++yy.lo == 0) {
            if (++yy.hi == 0) {
              if (++xx.lo == 0) {
              ++xx.hi;
              }
            }
          }
        }
        expo_y ^= 0x80000000;
      }


      while (!(xx.hi & 0x00100000)) {
        xx.hi = (xx.hi << 1) | (xx.lo >> 31);
        xx.lo = (xx.lo << 1) | (yy.hi >> 31);
        yy.hi = (yy.hi << 1) | (yy.lo >> 31);
        yy.lo = (yy.lo << 1);
        expo_x--;
      }
    } else {

      yy.lo = yy.lo + ww.lo;
      s = yy.lo < ww.lo;
      yy.hi = yy.hi + s;
      u = yy.hi < s;
      yy.hi = yy.hi + ww.hi;
      u += yy.hi < ww.hi;
      xx.lo = xx.lo + u;
      s = xx.lo < u;
      xx.lo = xx.lo + zz.lo;
      s += xx.lo < zz.lo;
      xx.hi = xx.hi + zz.hi + s;
      if (xx.hi & 0x00200000) {
        t = t | (yy.lo << 31);
        yy.lo = (yy.lo >> 1) | (yy.hi << 31);
        yy.hi = (yy.hi >> 1) | (xx.lo << 31);
        xx.lo = (xx.lo >> 1) | (xx.hi << 31);
        xx.hi = ((xx.hi & 0x80000000) | (xx.hi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  t = yy.lo | (t != 0);
  t = yy.hi | (t != 0);

  xx.hi |= expo_y;
  if (expo_x <= 0x7FD) {

    xx.hi = xx.hi & ~0x00100000;
    s = xx.lo & 1;
    u = xx.lo;
    if (rndMode == cudaRoundNearest) {
      xx.lo += (t == 0x80000000) ? s : (t >> 31);
    } else if (((rndMode == cudaRoundPosInf) && t && (!expo_y)) ||
               ((rndMode == cudaRoundMinInf) && t && expo_y)) {
      xx.lo += 1;
    }
    xx.hi += (u > xx.lo);
    xx.hi += ((expo_x + 1) << 20);
    return __hiloint2double(xx.hi, xx.lo);
  } else if ((int)expo_x >= 2046) {

    if ((rndMode == cudaRoundNearest) ||
        ((rndMode == cudaRoundPosInf) && (!expo_y)) ||
        ((rndMode == cudaRoundMinInf) && expo_y)) {
      xx.hi = (xx.hi & 0x80000000) | 0x7ff00000;
      xx.lo = 0;
    } else {
      xx.hi = (xx.hi & 0x80000000) | 0x7fefffff;
      xx.lo = 0xffffffff;
    }
    return __hiloint2double(xx.hi, xx.lo);
  }

  expo_x = (unsigned)(-(int)expo_x);
  if (expo_x > 54) {

    if (((rndMode == cudaRoundPosInf) && (!expo_y)) ||
        ((rndMode == cudaRoundMinInf) && expo_y)) {
      return __hiloint2double(xx.hi & 0x80000000, 1);
    } else {
      return __hiloint2double(xx.hi & 0x80000000, 0);
    }
  }
  yy.hi = xx.hi & 0x80000000;
  xx.hi = xx.hi & ~0xffe00000;
  if (expo_x >= 32) {
    t = xx.lo | (t != 0);
    xx.lo = xx.hi;
    xx.hi = 0;
    expo_x -= 32;
  }
  if (expo_x) {
    t = (t >> expo_x) | (xx.lo << (32 - expo_x)) | (t != 0);
    xx.lo = (xx.lo >> expo_x) | (xx.hi << (32 - expo_x));
    xx.hi = (xx.hi >> expo_x);
  }
  expo_x = xx.lo & 1;
  u = xx.lo;
  if (rndMode == cudaRoundNearest) {
    xx.lo += (t == 0x80000000) ? expo_x : (t >> 31);
  } else if (((rndMode == cudaRoundPosInf) && t && (!expo_y)) ||
             ((rndMode == cudaRoundMinInf) && t && expo_y)) {
    xx.lo += 1;
  }
  xx.hi += (u > xx.lo);
  xx.hi |= yy.hi;
  return __hiloint2double(xx.hi, xx.lo);
}

static double __fma_rn(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundNearest);
}

static double __fma_rd(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundMinInf);
}

static double __fma_ru(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundPosInf);
}

static double __fma_rz(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundZero);
}

static double __dadd_rz(double a, double b)
{
  return __fma_rz(a, 1.0, b);
}

static double __dadd_ru(double a, double b)
{
  return __fma_ru(a, 1.0, b);
}

static double __dadd_rd(double a, double b)
{
  return __fma_rd(a, 1.0, b);
}

static double __dmul_rz(double a, double b)
{
  return __fma_rz(a, b, __longlong_as_double(0x8000000000000000ULL));
}

static double __dmul_ru(double a, double b)
{
  return __fma_ru(a, b, __longlong_as_double(0x8000000000000000ULL));
}

static double __dmul_rd(double a, double b)
{
  return __fma_rd(a, b, 0.0);
}

static double __dadd_rn(double a, double b)
{
  return __fma_rn(a, 1.0, b);
}

static double __dmul_rn(double a, double b)
{
  return __fma_rn(a, b, __longlong_as_double(0x8000000000000000ULL));
}
# 4859 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/texture_fetch_functions.h" 1 3
# 1910 "/usr/local/cuda/bin/../include/texture_fetch_functions.h" 3
extern void __cudaTextureFetch(const void *tex, void *index, int integer, void *val);

static int4 __itexfetchi(const void *tex, int4 index)
{
  int4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static uint4 __utexfetchi(const void *tex, int4 index)
{
  uint4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static float4 __ftexfetchi(const void *tex, int4 index)
{
  float4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static int4 __itexfetch(const void *tex, float4 index, int dim)
{
  int4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}

static uint4 __utexfetch(const void *tex, float4 index, int dim)
{
  uint4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}

static float4 __ftexfetch(const void *tex, float4 index, int dim)
{
  float4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}
# 4860 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 945 "/usr/local/cuda/bin/../include/math_functions.h" 2 3


static int __cuda___signbitf(float a)
{
  return (int)((unsigned int)__float_as_int(a) >> 31);
}




static float __cuda_copysignf(float a, float b)
{
  return __int_as_float((__float_as_int(b) & 0x80000000) |
                        (__float_as_int(a) & ~0x80000000));
}
# 969 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) int min(int a, int b); int min(int a, int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umin(unsigned int a, unsigned int b); unsigned int umin(unsigned int a, unsigned int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) int max(int a, int b); int max(int a, int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umax(unsigned int a, unsigned int b); unsigned int umax(unsigned int a, unsigned int b)
{
  return a > b ? a : b;
}
# 1055 "/usr/local/cuda/bin/../include/math_functions.h" 3
static float __internal_nearbyintf(float a)
{
  float fa = fabsf(a);

  if (fa >= 8388608.0f) {
    return a;
  } else {
    volatile float u = 8388608.0f + fa;

    u = u - 8388608.0f;
    return copysignf(u, a);
  }
}

static float __internal_fminf(float a, float b)
{
  volatile union {
    float f;
    unsigned int i;
  } cvta, cvtb;
  int nana, nanb;

  cvta.f = a;
  cvtb.f = b;
  nana = ((cvta.i << 1) > 0xff000000);
  nanb = ((cvtb.i << 1) > 0xff000000);
  if (nana && nanb) return a + b;
  if (nana) return b;
  if (nanb) return a;
  if ((cvta.i | cvtb.i) == 0x80000000) {
    return __int_as_float(0x80000000);
  }
  return a < b ? a : b;
}

static float __internal_fmaxf(float a, float b)
{
  volatile union {
    float f;
    unsigned int i;
  } cvta, cvtb;
  int nana, nanb;

  cvta.f = a;
  cvtb.f = b;
  nana = ((cvta.i << 1) > 0xff000000);
  nanb = ((cvtb.i << 1) > 0xff000000);
  if (nana && nanb) return a + b;
  if (nana) return b;
  if (nanb) return a;
  if ((cvta.f == 0.0f) && (cvtb.f == 0.0f)) {
    cvta.i &= cvtb.i;
    return cvta.f;
  }
  return a > b ? a : b;
}
# 1151 "/usr/local/cuda/bin/../include/math_functions.h" 3
static long int __cuda_labs(long int a)
{
  return labs(a);
}

static float __cuda_ceilf(float a)
{
  return ceilf(a);
}

static float __cuda_floorf(float a)
{
  return floorf(a);
}

static float __cuda_sqrtf(float a)
{
  return sqrtf(a);
}

static float __cuda_rsqrtf(float a)
{
  return 1.0f / sqrtf(a);
}

static float __cuda_truncf(float a)
{
  return truncf(a);
}

static int __cuda_max(int a, int b)
{
  return max(a, b);
}

static int __cuda_min(int a, int b)
{
  return min(a, b);
}

static unsigned int __cuda_umax(unsigned int a, unsigned int b)
{
  return umax(a, b);
}

static unsigned int __cuda_umin(unsigned int a, unsigned int b)
{
  return umin(a, b);
}

static long long int __cuda_llrintf(float a)
{



  return __float2ll_rn(a);

}

static long int __cuda_lrintf(float a)
{




  return (long int)__cuda_llrintf(a);




}

static float __cuda_nearbyintf(float a)
{





  return __internal_nearbyintf(a);

}

static float __cuda_fmaxf(float a, float b)
{





  return __internal_fmaxf(a, b);

}

static float __cuda_fminf(float a, float b)
{





  return __internal_fminf(a, b);

}
# 1272 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda___finitef(float a)
{
  return __cuda_fabsf(a) < __int_as_float(0x7f800000);
}
# 1286 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda___isinff(float a)
{
  return __cuda_fabsf(a) == __int_as_float(0x7f800000);
}

static int __cuda___isnanf(float a)
{
  return !(__cuda_fabsf(a) <= __int_as_float(0x7f800000));
}

static float __cuda_nextafterf(float a, float b)
{
  unsigned int ia;
  unsigned int ib;
  ia = __float_as_int(a);
  ib = __float_as_int(b);




  if (__cuda___isnanf(a) || __cuda___isnanf(b)) return a + b;
  if (__int_as_float (ia | ib) == 0.0f) return __int_as_float(ib);





  if (__int_as_float(ia) == 0.0f) {
    return __cuda_copysignf(__int_as_float(0x00000001), b);
  }

  if ((a < b) && (a < 0.0f)) ia--;
  if ((a < b) && (a > 0.0f)) ia++;
  if ((a > b) && (a < 0.0f)) ia++;
  if ((a > b) && (a > 0.0f)) ia--;
  a = __int_as_float(ia);





  return a;
}

static float __cuda_nanf(const char *tagp)
{

  return __int_as_float(0x7fffffff);
}

static float __internal_fmad(float a, float b, float c)
{
  return a * b + c;
}


static float __internal_atanhf_kernel(float a_1, float a_2)
{
  float a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;
  t = 1.566305595598990E-001f/64.0f;
  t = __internal_fmad (t, a2, 1.995081856004762E-001f/16.0f);
  t = __internal_fmad (t, a2, 3.333382699617026E-001f/4.0f);
  t = t * a2;
  t = __internal_fmad (t, a, a_2);
  t = t + a_1;
  return t;
}




static float __internal_atanf_kernel(float a)
{
  float t4, t0, t1;

  t4 = a * a;
  t0 = -5.674867153f;
  t0 = __internal_fmad (t4, -0.823362947f, t0);
  t0 = __internal_fmad (t0, t4, -6.565555096f);
  t0 = t0 * t4;
  t0 = t0 * a;
  t1 = t4 + 11.33538818f;
  t1 = __internal_fmad (t1, t4, 28.84246826f);
  t1 = __internal_fmad (t1, t4, 19.69667053f);
  t1 = 1.0f / t1;
  a = __internal_fmad (t0, t1, a);
  return a;
}


static float __internal_tan_kernel(float a)
{
  float a2, s, t;

  a2 = a * a;
  t = __internal_fmad (4.114678393115178E-003f, a2, -8.231194034909670E-001f);
  s = a2 - 2.469348886157666E+000f;
  s = 1.0f / s;
  t = t * s;
  t = t * a2;
  t = __internal_fmad (t, a, a);
  return t;
}

static float __internal_accurate_logf(float a)
{
  float t;
  float z;
  float m;
  int ia, e;
  ia = __float_as_int(a);

  if ((ia < 0x00800000) || (ia > 0x7f7fffff)) {
    return __logf(a);
  }

  m = __int_as_float((ia & 0x807fffff) | 0x3f800000);
  e = ((unsigned)ia >> 23) - 127;
  if (m > 1.414213562f) {
    m = m * 0.5f;
    e = e + 1;
  }
  t = m - 1.0f;
  z = m + 1.0f;
  z = __fdividef (t, z);
  z = -t * z;
  z = __internal_atanhf_kernel(t, z);
  z = __internal_fmad ((float)e, 0.693147181f, z);
  return z;
}

static float2 __internal_log_ep(float a)
{
  float2 res;
  int expo;
  float m;
  float log_hi, log_lo;
  float t_hi, t_lo;
  float f, g, u, v, q;



  float r, s, e;

  expo = (__float_as_int(a) >> 23) & 0xff;


  if (expo == 0) {
    a *= 16777216.0f;
    expo = (__float_as_int(a) >> 23) & 0xff;
    expo -= 24;
  }

  expo -= 127;
  m = __int_as_float((__float_as_int(a) & 0x807fffff) | 0x3f800000);
  if (m > 1.414213562f) {
    m = m * 0.5f;
    expo = expo + 1;
  }






  f = m - 1.0f;
  g = m + 1.0f;
  g = 1.0f / g;
  u = 2.0f * f * g;
  v = u * u;
  q = 1.49356810919559350E-001f/64.0f;
  q = __internal_fmad (q, v, 1.99887797540072460E-001f/16.0f);
  q = __internal_fmad (q, v, 3.33333880955515580E-001f/4.0f);
  q = q * v;
  q = q * u;
  log_hi = __int_as_float(__float_as_int(u) & 0xfffff000);
  v = __int_as_float(__float_as_int(f) & 0xfffff000);
  u = 2.0f * (f - log_hi);
  f = f - v;
  u = __internal_fmad (-log_hi, v, u);
  u = __internal_fmad (-log_hi, f, u);
  u = g * u;



  r = log_hi + u;
  s = u - (r - log_hi);
  log_hi = r;
  log_lo = s;

  r = log_hi + q;
  s = ((log_hi - r) + q) + log_lo;
  log_hi = e = r + s;
  log_lo = (r - e) + s;


  t_hi = expo * 0.6931457519f;
  t_lo = expo * 1.4286067653e-6f;


  r = t_hi + log_hi;
  s = (((t_hi - r) + log_hi) + log_lo) + t_lo;
  res.y = e = r + s;
  res.x = (r - e) + s;
  return res;
}

static float __internal_accurate_log2f(float a)
{
  return 1.442695041f * __internal_accurate_logf(a);
}




static float2 __internal_dsmul (float2 x, float2 y)
{
    float2 z;

    volatile float up, vp, u1, u2, v1, v2, mh, ml;



    up = x.y * 4097.0f;
    u1 = (x.y - up) + up;
    u2 = x.y - u1;
    vp = y.y * 4097.0f;
    v1 = (y.y - vp) + vp;
    v2 = y.y - v1;
    mh = __fmul_rn(x.y,y.y);
    ml = (((u1 * v1 - mh) + u1 * v2) + u2 * v1) + u2 * v2;
    ml = (__fmul_rn(x.y,y.x) + __fmul_rn(x.x,y.y)) + ml;
    z.y = up = mh + ml;
    z.x = (mh - up) + ml;
    return z;
}


static unsigned int __cudart_i2opi_f [] = {
  0x3c439041,
  0xdb629599,
  0xf534ddc0,
  0xfc2757d1,
  0x4e441529,
  0xa2f9836e,
};


static float __internal_trig_reduction_kernel(float a, int *quadrant)
{
  float j;
  int q;
  if (__cuda_fabsf(a) > 48039.0f) {

    unsigned int ia = __float_as_int(a);
    unsigned int s = ia & 0x80000000;
    unsigned int result[7];
    unsigned int phi, plo;
    unsigned int hi, lo;
    unsigned int e;
    int idx;
    e = ((ia >> 23) & 0xff) - 128;
    ia = (ia << 8) | 0x80000000;

    idx = 4 - (e >> 5);
    hi = 0;



    for (q = 0; q < 6; q++) {
      plo = __cudart_i2opi_f[q] * ia;
      phi = __umulhi (__cudart_i2opi_f[q], ia);
      lo = hi + plo;
      hi = phi + (lo < plo);
      result[q] = lo;
    }
    result[q] = hi;
    e = e & 31;



    hi = result[idx+2];
    lo = result[idx+1];
    if (e) {
      q = 32 - e;
      hi = (hi << e) + (lo >> q);
      lo = (lo << e) + (result[idx] >> q);
    }
    q = hi >> 30;

    hi = (hi << 2) + (lo >> 30);
    lo = (lo << 2);
    e = (hi + (lo > 0)) > 0x80000000;
    q += e;
    if (s) q = -q;
    if (e) {
      unsigned int t;
      hi = ~hi;
      lo = -(int)lo;
      t = (lo == 0);
      hi += t;
      s = s ^ 0x80000000;
    }
    *quadrant = q;

    e = 0;
    while ((int)hi > 0) {
      hi = (hi << 1) + (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    lo = hi * 0xc90fdaa2;
    hi = __umulhi(hi, 0xc90fdaa2);
    if ((int)hi > 0) {
      hi = (hi << 1) + (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    hi = hi + (lo > 0);
    ia = s | (((e + 126) << 23) + (hi >> 8) + ((hi << 24) >= 0x80000000));
    return __int_as_float(ia);
  }
  q = __float2int_rn (a * 0.636619772f);
  j = (float)q;
  a = __internal_fmad (-j, 1.5703125000000000e+000f, a);
  a = __internal_fmad (-j, 4.8351287841796875e-004f, a);
  a = __internal_fmad (-j, 3.1385570764541626e-007f, a);
  a = __internal_fmad (-j, 6.0771005065061922e-011f, a);
  *quadrant = q;
  return a;
}
# 1630 "/usr/local/cuda/bin/../include/math_functions.h" 3
static float __internal_expf_kernel(float a, float scale)
{
  float j, z;

  j = __cuda_truncf(a * 1.442695041f);
  z = __internal_fmad (j, -0.6931457519f, a);
  z = __internal_fmad (j, -1.4286067653e-6f, z);
  z = z * 1.442695041f;
  z = __cuda_exp2f(z) * __cuda_exp2f(j + scale);
  return z;
}

static float __internal_accurate_expf(float a)
{
  float z;

  z = __internal_expf_kernel(a, 0.0f);
  if (a < -105.0f) z = 0.0f;
  if (a > 105.0f) z = __int_as_float(0x7f800000);
  return z;
}

static float __internal_accurate_exp10f(float a)
{
  float j, z;

  j = __cuda_truncf(a * 3.321928094f);
  z = __internal_fmad (j, -3.0102920532226563e-001f, a);
  z = __internal_fmad (j, -7.9034171557301747e-007f, z);
  z = z * 3.321928094f;
  z = __cuda_exp2f(z) * __cuda_exp2f(j);
  if (a < -46.0f) z = 0.0f;
  if (a > 46.0f) z = __int_as_float(0x7f800000);
  return z;
}

static float __internal_lgammaf_pos(float a)
{
  float sum;
  float s, t;

  if (a == __int_as_float(0x7f800000)) {
    return a;
  }
  if (a >= 3.0f) {
    if (a >= 7.8f) {



      s = 1.0f / a;
      t = s * s;
      sum = 0.77783067e-3f;
      sum = __internal_fmad (sum, t, -0.2777655457e-2f);
      sum = __internal_fmad (sum, t, 0.83333273853e-1f);
      sum = __internal_fmad (sum, s, 0.918938533204672f);
      s = 0.5f * __internal_accurate_logf(a);
      t = a - 0.5f;
      s = s * t;
      t = s - a;
      s = __fadd_rn(s, sum);
      t = t + s;
      return t;
    } else {
      a = a - 3.0f;
      s = -7.488903254816711E+002f;
      s = __internal_fmad (s, a, -1.234974215949363E+004f);
      s = __internal_fmad (s, a, -4.106137688064877E+004f);
      s = __internal_fmad (s, a, -4.831066242492429E+004f);
      s = __internal_fmad (s, a, -1.430333998207429E+005f);
      t = a - 2.592509840117874E+002f;
      t = __internal_fmad (t, a, -1.077717972228532E+004f);
      t = __internal_fmad (t, a, -9.268505031444956E+004f);
      t = __internal_fmad (t, a, -2.063535768623558E+005f);
      t = __fdividef (s, t);
      t = t + a;
      return t;
    }
  } else if (a >= 1.5f) {
    a = a - 2.0f;
    t = 4.959849168282574E-005f;
    t = __internal_fmad (t, a, -2.208948403848352E-004f);
    t = __internal_fmad (t, a, 5.413142447864599E-004f);
    t = __internal_fmad (t, a, -1.204516976842832E-003f);
    t = __internal_fmad (t, a, 2.884251838546602E-003f);
    t = __internal_fmad (t, a, -7.382757963931180E-003f);
    t = __internal_fmad (t, a, 2.058131963026755E-002f);
    t = __internal_fmad (t, a, -6.735248600734503E-002f);
    t = __internal_fmad (t, a, 3.224670187176319E-001f);
    t = __internal_fmad (t, a, 4.227843368636472E-001f);
    t = t * a;
    return t;
  } else if (a >= 0.7f) {
    a = 1.0f - a;
    t = 4.588266515364258E-002f;
    t = __internal_fmad (t, a, 1.037396712740616E-001f);
    t = __internal_fmad (t, a, 1.228036339653591E-001f);
    t = __internal_fmad (t, a, 1.275242157462838E-001f);
    t = __internal_fmad (t, a, 1.432166835245778E-001f);
    t = __internal_fmad (t, a, 1.693435824224152E-001f);
    t = __internal_fmad (t, a, 2.074079329483975E-001f);
    t = __internal_fmad (t, a, 2.705875136435339E-001f);
    t = __internal_fmad (t, a, 4.006854436743395E-001f);
    t = __internal_fmad (t, a, 8.224669796332661E-001f);
    t = __internal_fmad (t, a, 5.772156651487230E-001f);
    t = t * a;
    return t;
  } else {
    t = 3.587515669447039E-003f;
    t = __internal_fmad (t, a, -5.471285428060787E-003f);
    t = __internal_fmad (t, a, -4.462712795343244E-002f);
    t = __internal_fmad (t, a, 1.673177015593242E-001f);
    t = __internal_fmad (t, a, -4.213597883575600E-002f);
    t = __internal_fmad (t, a, -6.558672843439567E-001f);
    t = __internal_fmad (t, a, 5.772153712885004E-001f);
    t = t * a;
    t = __internal_fmad (t, a, a);
    return -__internal_accurate_logf(t);
  }
}


static float __internal_sin_kernel(float x)
{
  float x2, z;

  x2 = x * x;
  z = -1.95152959e-4f;
  z = __internal_fmad (z, x2, 8.33216087e-3f);
  z = __internal_fmad (z, x2, -1.66666546e-1f);
  z = z * x2;
  z = __internal_fmad (z, x, x);
  return z;
}


static float __internal_cos_kernel(float x)
{
  float x2, z;

  x2 = x * x;
  z = 2.44331571e-5f;
  z = __internal_fmad (z, x2, -1.38873163e-3f);
  z = __internal_fmad (z, x2, 4.16666457e-2f);
  z = __internal_fmad (z, x2, -5.00000000e-1f);
  z = __internal_fmad (z, x2, 1.00000000e+0f);
  return z;
}

static float __internal_accurate_sinf(float a)
{
  float z;
  int i;

  if ((__cuda___isinff(a)) || (a == 0.0f)) {
    return __fmul_rn (a, 0.0f);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}







static float __cuda_rintf(float a)
{



  return __cuda_nearbyintf(a);

}

static float __cuda_sinf(float a)
{





  return __internal_accurate_sinf(a);

}

static float __cuda_cosf(float a)
{





  float z;
  int i;

  if (__cuda___isinff(a)) {
    return __fadd_rn (a, -a);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  i++;
  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;

}

static float __cuda_tanf(float a)
{





  float z;
  int i;

  if (__cuda___isinff(a)) {
    return __int_as_float(0x7fffffff);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  z = __internal_tan_kernel(z);
  if (i & 1) {
    z = - (1.0f / z);
  }
  return z;

}

static float __cuda_log2f(float a)
{





  return __internal_accurate_log2f(a);

}

static float __cuda_expf(float a)
{





  return __internal_accurate_expf(a);

}

static float __cuda_exp10f(float a)
{





  return __internal_accurate_exp10f(a);

}

static float __cuda_coshf(float a)
{
  float z;

  a = __cuda_fabsf(a);
  z = __internal_expf_kernel(a, -2.0f);
  z = __internal_fmad (2.0f, z, __fdividef (0.125f, z));
  if (a >= 90.0f) {
    z = __int_as_float(0x7f800000);
  }
  return z;
}

static float __cuda_sinhf(float a)
{
  float s, z;

  s = a;
  a = __cuda_fabsf(a);
  if (a < 1.0f) {
    float a2 = a * a;

    z = 2.816951222e-6f;
    z = __internal_fmad (z, a2, 1.983615978e-4f);
    z = __internal_fmad (z, a2, 8.333350058e-3f);
    z = __internal_fmad (z, a2, 1.666666650e-1f);
    z = z * a2;
    z = __internal_fmad (z, a, a);
  } else {
    z = __internal_expf_kernel(a, -2.0f);
    z = __internal_fmad (2.0f, z, -__fdividef (0.125f, z));
    if (a >= 90.0f) {
      z = __int_as_float(0x7f800000);
    }
  }
  return __cuda_copysignf(z, s);
}

static float __cuda_tanhf(float a)
{
  float s, t;

  t = __cuda_fabsf(a);
  if (t < 0.55f) {
    float z, z2;
    z = t;
    z2 = z * z;
    t = 1.643758066599993e-2f;
    t = __internal_fmad (t, z2, -5.267181327760551e-2f);
    t = __internal_fmad (t, z2, 1.332072505223051e-1f);
    t = __internal_fmad (t, z2, -3.333294663641083e-1f);
    t = t * z2;
    s = __internal_fmad (t, z, z);
  } else {
    s = 1.0f - __fdividef(2.0f,(__internal_expf_kernel(2.0f * t, 0.0f)+1.0f));
    if (t >= 88.0f) {
      s = 1.0f;
    }
  }
  return __cuda_copysignf(s, a);
}

static float __cuda_atan2f(float a, float b)
{



  float t0, t1, t3, fa, fb;



  fb = __cuda_fabsf(b);
  fa = __cuda_fabsf(a);

  if (fa == 0.0f && fb == 0.0f) {
    t3 = __cuda___signbitf(b) ? 3.141592654f : 0;
  } else if ((fa == __int_as_float(0x7f800000)) && (fb == __int_as_float(0x7f800000))) {
    t3 = __cuda___signbitf(b) ? 2.356194490f : 0.785398163f;
  } else {

    if (fb < fa) {
      t0 = fa;
      t1 = fb;
    } else {
      t0 = fb;
      t1 = fa;
    }
    t3 = __internal_accurate_fdividef(t1, t0);
    t3 = __internal_atanf_kernel(t3);

    if (fa > fb) t3 = 1.570796327f - t3;
    if (b < 0.0f) t3 = 3.141592654f - t3;
  }
  t3 = __cuda_copysignf(t3, a);

  return t3;

}

static float __cuda_atanf(float a)
{
  float t0, t1;


  t0 = __cuda_fabsf(a);
  t1 = t0;
  if (t0 > 1.0f) {
    t1 = 1.0f / t1;
  }

  t1 = __internal_atanf_kernel(t1);

  if (t0 > 1.0f) {
    t1 = 1.570796327f - t1;
  }
  return __cuda_copysignf(t1, a);
}


static float __internal_asinf_kernel(float a)
{
  float t2, t3, t4;

  t2 = a * a;
  t3 = -0.501162291f;
  t3 = __internal_fmad (t3, t2, 0.915201485f);
  t3 = t3 * t2;
  t3 = t3 * a;
  t4 = t2 - 5.478654385f;
  t4 = __internal_fmad (t4, t2, 5.491230488f);
  t4 = 1.0f / t4;
  a = __internal_fmad (t3, t4, a);
  return a;
}

static float __cuda_asinf(float a)
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t2 = __internal_fmad (-2.0f, t1, 1.570796327f);
  if (t0 > 0.575f) {
    t1 = t2;
  }
  return __cuda_copysignf(t1, a);
}

static float __cuda_acosf(float a)
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t1 = t0 > 0.575f ? 2.0f * t1 : 1.570796327f - t1;
  if (__cuda___signbitf(a)) {
    t1 = 3.141592654f - t1;
  }
  return t1;
}

static float __cuda_logf(float a)
{





  return __internal_accurate_logf(a);

}

static float __cuda_log10f(float a)
{





  return 0.434294482f * __internal_accurate_logf(a);

}

static float __cuda_log1pf(float a)
{



  float t;




  if (a >= -0.394f && a <= 0.65f) {

    t = a + 2.0f;
    t = __fdividef (a, t);
    t = -a * t;
    t = __internal_atanhf_kernel (a, t);
  } else {
    t = __internal_accurate_logf (1.0f + a);
  }
  return t;

}

static float __cuda_acoshf(float a)
{



  float t;

  t = a - 1.0f;
  if (__cuda_fabsf(t) > 8388608.0f) {

    return 0.693147181f + __internal_accurate_logf(a);
  } else {
    t = t + __cuda_sqrtf(__internal_fmad (a, t, t));
    return __cuda_log1pf(t);
  }

}

static float __cuda_asinhf(float a)
{



  float fa, oofa, t;

  fa = __cuda_fabsf(a);
  if (fa > 8.507059173e37f) {
    t = 0.693147181f + __logf(fa);
  } else {
    oofa = 1.0f / fa;
    t =fa+__fdividef (fa,(oofa+__cuda_sqrtf(__internal_fmad(oofa,oofa,1.0f))));
    t = __cuda_log1pf(t);
  }
  return __cuda_copysignf(t, a);

}

static float __cuda_atanhf(float a)
{



  float fa, t;

  fa = __cuda_fabsf(a);
  t = __fdividef ((2.0f * fa), (1.0f - fa));
  t = 0.5f * __cuda_log1pf(t);
  return __cuda_copysignf(t, a);

}

static float __cuda_expm1f(float a)
{
  float t, z, j, u;

  t = __cuda_rintf (a * 1.442695041f);
  z = __internal_fmad (-t, 0.6931457519f, a);
  z = __internal_fmad (-t, 1.4286067653e-6f, z);

  if (__cuda_fabsf(a) < 0.41f) {
    z = a;
    t = 0.0f;
  }

  j = t;
  if (t == 128.0f) j = j - 1.0f;

  u = 1.38795078474044430E-003f;
  u = __internal_fmad (u, z, 8.38241261853264930E-003f);
  u = __internal_fmad (u, z, 4.16678317762833940E-002f);
  u = __internal_fmad (u, z, 1.66663978874356580E-001f);
  u = __internal_fmad (u, z, 4.99999940395997040E-001f);
  u = u * z;
  u = __internal_fmad (u, z, z);
  if (a == 0.0f) u = a;

  z = __cuda_exp2f (j);
  a = z - 1.0f;
  if (a != 0.0f) u = __internal_fmad (u, z, a);
  if (t == 128.0f) u = u + u;

  if (j > 128.0f) u = __int_as_float(0x7f800000);
  if (j < -25.0f) u = -1.0f;
  return u;
}

static float __cuda_hypotf(float a, float b)
{



  float v, w, t;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);

  if (a > b) {
    v = a;
    w = b;
  } else {
    v = b;
    w = a;
  }
  t = __internal_accurate_fdividef(w, v);
  t = __internal_fmad (t, t, 1.0f);
  t = v * __cuda_sqrtf(t);
  if (v == 0.0f) {
    t = v + w;
  }
  if ((v == __int_as_float(0x7f800000)) || (w == __int_as_float(0x7f800000))) {
    t = __int_as_float(0x7f800000);
  }
  return t;

}

static float __cuda_cbrtf(float a)
{



  float s, t;

  s = __cuda_fabsf(a);
  if ((a == 0.0f) || (s == __int_as_float(0x7f800000))) {
    return a;
  }
  t = __cuda_exp2f(0.333333333f * __log2f(s));
  t = t-(t-(__fdividef(s,(t*t))))*0.333333333f;
  if (__cuda___signbitf(a)) {
    t = -t;
  }
  return t;

}

static float __cuda_erff(float a)
{
  float t, r, q;

  t = __cuda_fabsf(a);
  if (t < 1.0f) {
    t = t * t;
    r = -5.58510127926029810E-004f;
    r = __internal_fmad (r, t, 4.90688891415893070E-003f);
    r = __internal_fmad (r, t, -2.67027980930150640E-002f);
    r = __internal_fmad (r, t, 1.12799056505903940E-001f);
    r = __internal_fmad (r, t, -3.76122956138427440E-001f);
    r = __internal_fmad (r, t, 1.12837911712623450E+000f);
    a = a * r;
  } else if (t <= __int_as_float(0x7f800000)) {



    q = __internal_fmad (t, 0.3275911f, 1.0f);
    q = 1.0f / q;
    r = 1.061405429f;
    r = __internal_fmad (r, q, -1.453152027f);
    r = __internal_fmad (r, q, 1.421413741f);
    r = __internal_fmad (r, q, -0.284496736f);
    r = __internal_fmad (r, q, 0.254829592f);
    r = r * q;
    q = __internal_expf_kernel(-a * a, 0.0f);
    r = __internal_fmad (-q, r, 1.0f);
    if (t >= 5.5f) {
      r = 1.0f;
    }
    a = __int_as_float (__float_as_int(r) | (__float_as_int(a) & 0x80000000));
  }
  return a;
}

static float __cuda_erfinvf (float a)
{
  float fa, t;

  fa = fabsf(a);
  if (fa >= 1.0f) {
    t = __cuda_rsqrtf (__int_as_float (0xffc00000));
    if (fa == 1.0f) {
      t = a * __int_as_float(0x7f800000);
    }
  } else if (fa > 0.9375f) {




    float p, q;

    t = __cuda_log1pf(-fa);
    t = __cuda_rsqrtf(-t);
    p = -1.64441567910e-1f;
    p = __internal_fmad (p, t, 6.80544246825e-1f);
    p = __internal_fmad (p, t, -1.12808139162e+0f);
    p = __internal_fmad (p, t, 6.90969348887e-1f);
    p = __internal_fmad (p, t, 1.38271964963e+0f);
    p = __internal_fmad (p, t, 1.55047000312e-1f);
    q = t + 1.38522814199e+0f;
    q = __internal_fmad (q, t, 1.55024849822e-1f);
    q = q * t;
    t = __fdividef (p, q);
    if (a < 0.0f) t = -t;
  } else if (fa > 0.75f) {




    float p, q;

    t = __internal_fmad (a, a, -0.87890625f);
    p = -7.1986748896e+0f;
    p = __internal_fmad (p, t, +1.3411974175e+1f);
    p = __internal_fmad (p, t, -5.1381573203e+0f);
    p = __internal_fmad (p, t, 4.9633374831e-1f);
    q = t -1.1436535838e+1f;
    q = __internal_fmad (q, t, 1.3568885572e+1f);
    q = __internal_fmad (q, t, -4.1747509256e+0f);
    q = __internal_fmad (q, t, 3.5327242323e-1f);
    p = __fdividef (p, q);
    t = a * p;
  } else {
    float a2;

    a2 = a * a;
    t = 6.1046168794766742E-001f;
    t = __internal_fmad (t, a2, -8.9504882462753121E-001f);
    t = __internal_fmad (t, a2, 7.0224162369928511E-001f);
    t = __internal_fmad (t, a2, -1.9993784895823222E-001f);
    t = __internal_fmad (t, a2, 1.1920613463949599E-001f);
    t = __internal_fmad (t, a2, 8.0131492246997685E-002f);
    t = __internal_fmad (t, a2, 1.2793154958377403E-001f);
    t = __internal_fmad (t, a2, 2.3200529172828793E-001f);
    t = __internal_fmad (t, a2, 8.8622695604694379E-001f);
    t = t * a;
  }
  return t;
}

static float __cuda_erfcf(float a)
{
  if (a <= 0.55f) {
    return 1.0f - __cuda_erff(a);
  } else if (a > 10.0f) {
    return 0.0f;
  } else {
    float p;
    float q;
    float h;
    float l;




    p = 4.014893410762552E-006f;
    p = __internal_fmad (p, a, 5.640401259462436E-001f);
    p = __internal_fmad (p, a, 2.626649872281140E+000f);
    p = __internal_fmad (p, a, 5.486372652389673E+000f);
    p = __internal_fmad (p, a, 5.250714831459401E+000f);
    q = a + 4.651376250488319E+000f;
    q = __internal_fmad (q, a, 1.026302828878470E+001f);
    q = __internal_fmad (q, a, 1.140762166021288E+001f);
    q = __internal_fmad (q, a, 5.251211619089947E+000f);

    h = 1.0f / q;
    q = __internal_fmad (-q * h, h, 2.0f * h);
    p = p * q;

    h = __int_as_float(__float_as_int(a) & 0xfffff000);
    l = __fadd_rn (a, -h);
    q = __fmul_rn (-h, h);
    q = __internal_expf_kernel(q, 0.0f);
    a = a + h;
    l = l * a;
    h = __internal_expf_kernel(-l, 0.0f);
    q = q * h;
    p = p * q;
    return p;
  }
}

static float __cuda_erfcinvf (float a)
{
  float t;
  if (a <= 0.0f) {
    t = __int_as_float(0x7fffffff);
    if (a == 0.0f) {
      t = (1.0f - a) * __int_as_float(0x7f800000);
    }
  }
  else if (a >= 0.0625f) {
    t = __cuda_erfinvf (1.0f - a);
  }
  else {
    float p, q;
    t = __cuda_logf(a);
    t = __cuda_rsqrtf(-t);
    p = -1.64441567910e-1f;
    p = __internal_fmad (p, t, 6.80544246825e-1f);
    p = __internal_fmad (p, t, -1.12808139162e+0f);
    p = __internal_fmad (p, t, 6.90969348887e-1f);
    p = __internal_fmad (p, t, 1.38271964963e+0f);
    p = __internal_fmad (p, t, 1.55047000312e-1f);
    q = t + 1.38522814199e+0f;
    q = __internal_fmad (q, t, 1.55024849822e-1f);
    q = q * t;
    t = __fdividef (p, q);
  }
  return t;
}

static float __cuda_lgammaf(float a)
{
  float t;
  float i;
  int quot;
  t = __internal_lgammaf_pos(__cuda_fabsf(a));
  if (a >= 0.0f) return t;
  a = __cuda_fabsf(a);
  i = __cuda_floorf(a);
  if (a == i) return __int_as_float(0x7f800000);
  if (a < 1e-19f) return -__internal_accurate_logf(a);
  i = __cuda_rintf (2.0f * a);
  quot = (int)i;
  i = __internal_fmad (-i, 0.5f, a);
  i = i * 3.141592654f;
  if (quot & 1) {
    i = __internal_cos_kernel(i);
  } else {
    i = __internal_sin_kernel(i);
  }
  i = __cuda_fabsf(i);
  t = 1.144729886f - __internal_accurate_logf(i * a) - t;
  return t;
}

static float __cuda_ldexpf(float a, int b)
{



  float fa = __cuda_fabsf(a);

  if ((fa == 0.0f) || (fa == __int_as_float(0x7f800000)) || (b == 0)) {
    if (!(fa > 0.0f)) a = a + a;
    return a;
  } else if (__cuda_abs(b) < 126) {
    return a * __cuda_exp2f((float)b);
  } else if (__cuda_abs(b) < 252) {
    int bhalf = b / 2;
    return a * __cuda_exp2f((float)bhalf) * __cuda_exp2f((float)(b - bhalf));
  } else {
    int bquarter = b / 4;
    float t = __cuda_exp2f((float)bquarter);
    return a * t * t * t * __cuda_exp2f((float)(b - 3 * bquarter));
  }

}

static float __cuda_scalbnf(float a, int b)
{




  return __cuda_ldexpf(a, b);

}

static float __cuda_scalblnf(float a, long int b)
{



  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return __cuda_scalbnf(a, t);

}

static float __cuda_frexpf(float a, int *b)
{
  float fa = __cuda_fabsf(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < 1.175494351e-38f) {
    a *= 16777216.0f;
    denorm = 24;
  } else {
    denorm = 0;
  }
  expo = ((__float_as_int(a) >> 23) & 0xff);
  if ((fa == 0.0f) || (expo == 0xff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 126;
    a = __int_as_float(((__float_as_int(a) & 0x807fffff) | 0x3f000000));
  }
  *b = expo;
  return a;
}

static float __cuda_modff(float a, float *b)
{



  float t;
  if (__cuda___finitef(a)) {
    t = __cuda_truncf(a);
    *b = t;
    t = a - t;
    return __cuda_copysignf(t, a);
  } else if (__cuda___isinff(a)) {
    t = 0.0f;
    *b = a;
    return __cuda_copysignf(t, a);
  } else {
    *b = a;
    return a;
  }

}

static float __cuda_fmodf(float a, float b)
{



  float orig_a = a;
  float orig_b = b;
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    return __cuda_rsqrtf (__int_as_float (0xffc00000));
  } else if (a >= b) {


    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }







    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5f;
    }
    return __cuda_copysignf(a, orig_a);
  } else {
    if (!(a > 0.0f)) orig_a = orig_a + orig_a;
    return orig_a;
  }

}

static float __cuda_remainderf(float a, float b)
{

  float twoa = 0.0f;
  unsigned int quot0 = 0;
  float orig_a = a;
  float orig_b = b;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    return __cuda_rsqrtf (__int_as_float (0xffc00000));
  } else if (a >= b) {

    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
# 2638 "/usr/local/cuda/bin/../include/math_functions.h" 3
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5f;
    }
  }


  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
# 2670 "/usr/local/cuda/bin/../include/math_functions.h" 3
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  return a;
}

static float __cuda_remquof(float a, float b, int* quo)
{
  float twoa = 0.0f;
  unsigned int quot = 0;
  unsigned int sign;
  float orig_a = a;
  float orig_b = b;


  sign = 0 - (__cuda___signbitf(a) != __cuda___signbitf(b));
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    *quo = quot;
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    *quo = quot;
    return __cuda_rsqrtf (__int_as_float (0xffc00000));
  } else if (a >= b) {


    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
# 2723 "/usr/local/cuda/bin/../include/math_functions.h" 3
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5f;
    }
  }


  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
# 2758 "/usr/local/cuda/bin/../include/math_functions.h" 3
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  quot = quot & (~((~0)<<3));
  quot = quot ^ sign;
  quot = quot - sign;
  *quo = quot;
  return a;
}

static float __cuda_fmaf(float a, float b, float c)
{
  return __fmaf_rn(a, b, c);
}

static float __internal_accurate_powf(float a, float b)
{
  float2 loga, prod;



  float t;



  loga = __internal_log_ep(a);


  if (__cuda_fabsf(b) > 1.0e34f) b *= 1.220703125e-4f;
  prod.y = b;
  prod.x = 0.0f;
  prod = __internal_dsmul (prod, loga);


  if (__float_as_int(prod.y) == 0x42b17218) {
    prod.y = __int_as_float(__float_as_int(prod.y) - 1);
    prod.x = prod.x + __int_as_float(0x37000000);
  }


  t = __cuda_expf(prod.y);

  if (t != __int_as_float(0x7f800000)) {



    t = __internal_fmad (t, prod.x, t);
  }
  return t;
}

static float __cuda_powif(float a, int b)
{
  unsigned int e = __cuda_abs(b);
  float r = 1.0f;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0f / r : r;
    }
    a = a * a;
  }
}

static double __cuda_powi(double a, int b)
{
  unsigned int e = __cuda_abs(b);
  double r = 1.0;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0 / r : r;
    }
    a = a * a;
  }
}

static float __cuda_powf(float a, float b)
{





  int bIsOddInteger;
  float t;
  if (a == 1.0f || b == 0.0f) {
    return 1.0f;
  }
  if (__cuda___isnanf(a) || __cuda___isnanf(b)) {
    return a + b;
  }
  if (a == __int_as_float(0x7f800000)) {
    return __cuda___signbitf(b) ? 0.0f : __int_as_float(0x7f800000);
  }
  if (__cuda___isinff(b)) {
    if (a == -1.0f) {
      return 1.0f;
    }
    t = (__cuda_fabsf(a) > 1.0f) ? __int_as_float(0x7f800000) : 0.0f;
    if (b < 0.0f) {
      t = 1.0f / t;
    }
    return t;
  }
  bIsOddInteger = (b - (2.0f * floorf(0.5f * b))) == 1.0f;
  if (a == 0.0f) {
    t = bIsOddInteger ? a : 0.0f;
    if (b < 0.0f) {
      t = 1.0f / t;
    }
    return t;
  }
  if (a == -__int_as_float(0x7f800000)) {
    t = - ((b < 0.0f) ? (1.0f / a) : a);
    if (bIsOddInteger) {
      t = __int_as_float(__float_as_int(t) ^ 0x80000000);
    }
    return t;
  }
  if ((a < 0.0f) && (b != __cuda_truncf(b))) {
    return __cuda_rsqrtf(__int_as_float(0xffc00000));
  }
  t = __cuda_fabsf(a);
  t = __internal_accurate_powf(t, b);
  if ((a < 0.0f) && bIsOddInteger) {
    t = __int_as_float(__float_as_int(t) ^ 0x80000000);
  }
  return t;

}


static float __internal_tgammaf_kernel(float a)
{
  float t;
  t = -1.05767296987211380E-003f;
  t = __internal_fmad (t, a, 7.09279059435508670E-003f);
  t = __internal_fmad (t, a, -9.65347121958557050E-003f);
  t = __internal_fmad (t, a, -4.21736613253687960E-002f);
  t = __internal_fmad (t, a, 1.66542401247154280E-001f);
  t = __internal_fmad (t, a, -4.20043267827838460E-002f);
  t = __internal_fmad (t, a, -6.55878234051332940E-001f);
  t = __internal_fmad (t, a, 5.77215696929794240E-001f);
  t = __internal_fmad (t, a, 1.00000000000000000E+000f);
  return t;
}





static float __cuda_tgammaf(float a)
{
  float s, xx, x=a;
  if (x >= 0.0f) {
    if (x > 36.0f) x = 36.0f;
    s = 1.0f;
    xx = x;
    if (x > 34.03f) {
      xx -= 1.0f;
    }
    while (xx > 1.5f) {
      xx = xx - 1.0f;
      s = s * xx;
    }
    if (x >= 0.5f) {
      xx = xx - 1.0f;
    }
    xx = __internal_tgammaf_kernel(xx);
    if (x < 0.5f) {
      xx = xx * x;
    }
    s = __fdividef(s, xx);
    if (x > 34.03f) {

      xx = x - 1.0f;
      s = s * xx;
    }
    return s;
  } else {
    if (x == __cuda_floorf(x)) {
      x = __int_as_float(0x7fffffff);

      return x;

    }
    if (x < -41.1f) x = -41.1f;
    xx = x;
    if (x < -34.03f) {
      xx += 6.0f;
    }
    s = xx;
    while (xx < -0.5f) {
      xx = xx + 1.0f;
      s = s * xx;
    }
    xx = __internal_tgammaf_kernel(xx);
    s = s * xx;
    s = 1.0f / s;
    if (x < -34.03f) {
      xx = x;
      xx *= (x + 1.0f);
      xx *= (x + 2.0f);
      xx *= (x + 3.0f);
      xx *= (x + 4.0f);
      xx *= (x + 5.0f);
      xx = 1.0f / xx;
      s = s * xx;
      if ((a < -42.0f) && !(((int)a)&1)) {
        s = __int_as_float(0x80000000);
      }
    }
    return s;
  }
}

static float __cuda_roundf(float a)
{



  float fa = __cuda_fabsf(a);
  float u = __cuda_copysignf (0.5f, a);
  u = __cuda_truncf (a + u);
  if (fa > 8388608.0f) u = a;
  if (fa < 0.5f) u = __cuda_truncf (a);
  return u;

}

static long long int __internal_llroundf_kernel(float a)
{
  unsigned long long int res, t = 0LL;
  int shift;
  unsigned int ia = __float_as_int(a);
  if ((ia << 1) > 0xff000000) return 0LL;
  if ((int)ia >= 0x5f000000) return 0x7fffffffffffffffLL;
  if (ia >= 0xdf000000) return 0x8000000000000000LL;
  shift = 189 - ((ia >> 23) & 0xff);
  res = ((long long int)(((ia << 8) | 0x80000000) >> 1)) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (t >= 0x8000000000000000LL) {
    res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)(-(long long int)res);
  return (long long int)res;
}

static long long int __cuda_llroundf(float a)
{



  return __internal_llroundf_kernel(a);

}

static long int __cuda_lroundf(float a)
{




  return (long int)__cuda_llroundf(a);
# 3045 "/usr/local/cuda/bin/../include/math_functions.h" 3
}

static float __cuda_fdimf(float a, float b)
{
  float t;
  t = a - b;
  if (a <= b) {
    t = 0.0f;
  }
  return t;
}

static int __cuda_ilogbf(float a)
{
  unsigned int i;
  int expo;
  a = __cuda_fabsf(a);
  if (a <= 1.175494351e-38f) {

    if (a == 0.0f) {
      expo = -((int)((unsigned int)-1 >> 1))-1;
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    if ((i == 0x7f800000)) {
      expo = ((int)((unsigned int)-1 >> 1));
    }
    if ((i > 0x7f800000)) {
      expo = -((int)((unsigned int)-1 >> 1))-1;
    }
  }
  return expo;
}

static float __cuda_logbf(float a)
{



  unsigned int i;
  int expo;
  float res;

  if (__cuda___isnanf(a)) return a + a;

  a = __cuda_fabsf(a);
  if (a <= 1.175494351e-38f) {

    if (a == 0.0f) {
      res = -__int_as_float(0x7f800000);
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
      res = (float)expo;
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    res = (float)expo;
    if ((i >= 0x7f800000)) {

      res = a + a;
    }
  }
  return res;

}

static void __cuda_sincosf(float a, float *sptr, float *cptr)
{





  float t, u, s, c;
  int quadrant;
  if (__cuda___isinff(a)) {
    *sptr = __int_as_float(0x7fffffff);
    *cptr = __int_as_float(0x7fffffff);
    return;
  }
  if (a == 0.0f) {
    *sptr = a;
    *cptr = 1.0f;
    return;
  }
  t = __internal_trig_reduction_kernel(a, &quadrant);
  u = __internal_cos_kernel(t);
  t = __internal_sin_kernel(t);
  if (quadrant & 1) {
    s = u;
    c = t;
  } else {
    s = t;
    c = u;
  }
  if (quadrant & 2) {
    s = -s;
  }
  quadrant++;
  if (quadrant & 2) {
    c = -c;
  }
  *sptr = s;
  *cptr = c;

}
# 3176 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) double rsqrt(double a); double rsqrt(double a)
{
  return 1.0 / sqrt(a);
}

extern __attribute__((__weak__)) float rsqrtf(float a); float rsqrtf(float a)
{
  return (float)rsqrt((double)a);
}
# 3734 "/usr/local/cuda/bin/../include/math_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx3.h" 1 3
# 61 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx3.h" 3
static double __cuda_fabs(double a)
{
  return fabs(a);
}

static double __cuda_fmax(double a, double b)
{

  volatile union __cudart_DoubleUlonglongCvt cvta, cvtb;
  int nana, nanb;

  cvta.d = a;
  cvtb.d = b;
  nana = ((cvta.i << 1) > 0xffe0000000000000ULL);
  nanb = ((cvtb.i << 1) > 0xffe0000000000000ULL);
  if (nana && nanb) return a + b;
  if (nana) return b;
  if (nanb) return a;
  if ((cvta.d == 0.0) && (cvtb.d == 0.0)) {
    cvta.i &= cvtb.i;
    return cvta.d;
  }
  return a > b ? a : b;



}

static double __cuda_fmin(double a, double b)
{

  volatile union __cudart_DoubleUlonglongCvt cvta, cvtb;
  int nana, nanb;

  cvta.d = a;
  cvtb.d = b;
  nana = ((cvta.i << 1) > 0xffe0000000000000ULL);
  nanb = ((cvtb.i << 1) > 0xffe0000000000000ULL);
  if (nana && nanb) return a + b;
  if (nana) return b;
  if (nanb) return a;
  if ((cvta.i | cvtb.i) == 0x8000000000000000ULL) {
    return __longlong_as_double(0x8000000000000000ULL) ;
  }
  return a < b ? a : b;



}

static double __cuda_ceil(double a)
{
  return ceil(a);
}

static double __cuda_floor(double a)
{
  return floor(a);
}

static double __cuda_trunc(double a)
{
  return trunc(a);
}

static double __cuda_nearbyint(double a)
{



  double res = nearbyint(a);





  return res;

}







static double __cuda_rint(double a)
{
  return __cuda_nearbyint(a);
}

static long int __cuda_lrint(double a)
{

  return (long int)__double2ll_rn(a);



}

static long long int __cuda_llrint(double a)
{
  return __double2ll_rn(a);
}

static int __cuda___signbit(double a)
{
  return (int)((unsigned int)__double2hiint(a) >> 31);
}

static int __cuda___finite(double a)
{
  return __cuda_fabs(a) < __longlong_as_double(0x7ff0000000000000ULL);
}

static int __cuda___isinf(double a)
{
  return __cuda_fabs(a) == __longlong_as_double(0x7ff0000000000000ULL);
}

static int __cuda___isnan(double a)
{
  return !(__cuda_fabs(a) <= __longlong_as_double(0x7ff0000000000000ULL));
}

static double __cuda_copysign(double a, double b)
{
  int alo, ahi, bhi;

  bhi = __double2hiint(b);
  alo = __double2loint(a);
  ahi = __double2hiint(a);
  ahi = (bhi & 0x80000000) | (ahi & ~0x80000000);
  return __hiloint2double(ahi, alo);
}


static unsigned long long int __cudart_i2opi_d [] = {
  0x6bfb5fb11f8d5d08ULL,
  0x3d0739f78a5292eaULL,
  0x7527bac7ebe5f17bULL,
  0x4f463f669e5fea2dULL,
  0x6d367ecf27cb09b7ULL,
  0xef2f118b5a0a6d1fULL,
  0x1ff897ffde05980fULL,
  0x9c845f8bbdf9283bULL,
  0x3991d639835339f4ULL,
  0xe99c7026b45f7e41ULL,
  0xe88235f52ebb4484ULL,
  0xfe1deb1cb129a73eULL,
  0x06492eea09d1921cULL,
  0xb7246e3a424dd2e0ULL,
  0xfe5163abdebbc561ULL,
  0xdb6295993c439041ULL,
  0xfc2757d1f534ddc0ULL,
  0xa2f9836e4e441529ULL,
};

static double __internal_trig_reduction_kerneld(double a, int *quadrant)
{
  double j;
  int q;
  if (__cuda_fabs(a) > 2147483648.0) {

    unsigned long long int ia;
    unsigned long long int s;
    unsigned long long int result[5];
    unsigned long long int phi, plo;
    unsigned long long int hi, lo;
    unsigned int e;
    int idx;

    ia = __double_as_longlong(a);
    s = ia & 0x8000000000000000ULL;
    e = (unsigned int)(((ia >> 52) & 0x7ff) - 1024);
    ia = (ia << 11) | 0x8000000000000000ULL;

    idx = 16 - (e >> 6);
    hi = 0;



    for (q = (idx-1); q < min(18,idx+3); q++) {
      plo = __cudart_i2opi_d[q] * ia;
      phi = __umul64hi (__cudart_i2opi_d[q], ia);
      lo = hi + plo;
      hi = phi + (lo < plo);
      result[q-(idx-1)] = lo;
    }
    result[q-(idx-1)] = hi;
    e = e & 63;



    hi = result[3];
    lo = result[2];
    if (e) {
      q = 64 - e;
      hi = (hi << e) | (lo >> q);
      lo = (lo << e) | (result[1] >> q);
    }
    q = (int)(hi >> 62);

    hi = (hi << 2) | (lo >> 62);
    lo = (lo << 2);
    e = (hi | (lo > 0)) > 0x8000000000000000ULL;
    q += e;
    if (s) q = -q;
    if (e) {
      unsigned long long int t;
      hi = ~hi;
      lo = -(long long int)lo;
      t = (lo == 0ULL);
      hi += t;
      s = s ^ 0x8000000000000000ULL;
    }
    *quadrant = q;

    e = 0;
    while ((long long int)hi > 0) {
      hi = (hi << 1) | (lo >> 63);
      lo = (lo << 1);
      e--;
    }
    lo = hi * 0xC90FDAA22168C235ULL;
    hi = __umul64hi (hi, 0xC90FDAA22168C235ULL);
    if ((long long int)hi > 0) {
      hi = (hi << 1) | (lo >> 63);
      lo = (lo << 1);
      e--;
    }
    hi = hi + (lo > 0);
    ia = s | ((((unsigned long long int)(e + 1022)) << 52) + (hi >> 11) +
              ((hi << 53) >= 0x8000000000000000ULL));
    return __longlong_as_double(ia);
  }

  q = __double2int_rn (a * 6.3661977236758138e-1);
  j = (double)q;




  a = __fma_rn (-j, 1.5707963267948966e+000, a);
  a = __fma_rn (-j, 6.1232339957367574e-017, a);
  a = __fma_rn (-j, 8.4784276603688985e-032, a);
  *quadrant = q;
  return a;
}


static double __internal_sin_kerneld(double x)
{
  double x2, z;
  x2 = x * x;
  z = 1.5896230157221844E-010;
  z = __fma_rn (z, x2, -2.5050747762850355E-008);
  z = __fma_rn (z, x2, 2.7557313621385676E-006);
  z = __fma_rn (z, x2, -1.9841269829589539E-004);
  z = __fma_rn (z, x2, 8.3333333333221182E-003);
  z = __fma_rn (z, x2, -1.6666666666666630E-001);
  z = z * x2;
  z = __fma_rn (z, x, x);
  return z;
}


static double __internal_cos_kerneld(double x)
{
  double x2, z;
  x2 = x * x;
  z = -1.136788825395985E-011;
  z = __fma_rn (z, x2, 2.087588480545065E-009);
  z = __fma_rn (z, x2, -2.755731555403950E-007);
  z = __fma_rn (z, x2, 2.480158729365970E-005);
  z = __fma_rn (z, x2, -1.388888888888074E-003);
  z = __fma_rn (z, x2, 4.166666666666664E-002);
  z = __fma_rn (z, x2, -5.000000000000000E-001);
  z = __fma_rn (z, x2, 1.000000000000000E+000);
  return z;
}


static double __internal_tan_kerneld(double x, int i)
{
  double x2, z, q;
  x2 = x * x;
  z = 9.8006287203286300E-006;
  z = __fma_rn (z, x2, -2.4279526494179897E-005);
  z = __fma_rn (z, x2, 4.8644173130937162E-005);
  z = __fma_rn (z, x2, -2.5640012693782273E-005);
  z = __fma_rn (z, x2, 6.7223984330880073E-005);
  z = __fma_rn (z, x2, 8.3559287318211639E-005);
  z = __fma_rn (z, x2, 2.4375039850848564E-004);
  z = __fma_rn (z, x2, 5.8886487754856672E-004);
  z = __fma_rn (z, x2, 1.4560454844672040E-003);
  z = __fma_rn (z, x2, 3.5921008885857180E-003);
  z = __fma_rn (z, x2, 8.8632379218613715E-003);
  z = __fma_rn (z, x2, 2.1869488399337889E-002);
  z = __fma_rn (z, x2, 5.3968253972902704E-002);
  z = __fma_rn (z, x2, 1.3333333333325342E-001);
  z = __fma_rn (z, x2, 3.3333333333333381E-001);
  z = z * x2;
  q = __fma_rn (z, x, x);
  if (i) {
    double s = q - x;
    double w = __fma_rn (z, x, -s);
    z = 1.0 / q;
    z = -z;
    s = __fma_rn (q, z, 1.0);
    q = __fma_rn (z, __fma_rn (z, w, s), z);
  }
  return q;
}

static double __cuda_sqrt(double a)
{
  return sqrt(a);
}

static double __cuda_rsqrt(double a)
{

    return 1.0 / sqrt(a);



}


static double __internal_expm1_kernel (double a)
{
  double t;
  t = 2.08842685477913050E-009;
  t = __fma_rn (t, a, 2.51366409033551950E-008);
  t = __fma_rn (t, a, 2.75574612072447230E-007);
  t = __fma_rn (t, a, 2.75571539284473460E-006);
  t = __fma_rn (t, a, 2.48015869443077950E-005);
  t = __fma_rn (t, a, 1.98412699878799470E-004);
  t = __fma_rn (t, a, 1.38888888892029890E-003);
  t = __fma_rn (t, a, 8.33333333327662860E-003);
  t = __fma_rn (t, a, 4.16666666666656370E-002);
  t = __fma_rn (t, a, 1.66666666666667380E-001);
  t = __fma_rn (t, a, 5.00000000000000000E-001);
  t = t * a;
  t = __fma_rn (t, a, a);
  return t;
}


static double __internal_atanh_kernel (double a_1, double a_2)
{
  double a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;
  t = 7.597322383488143E-002/65536.0;
  t = __fma_rn (t, a2, 6.457518383364042E-002/16384.0);
  t = __fma_rn (t, a2, 7.705685707267146E-002/4096.0);
  t = __fma_rn (t, a2, 9.090417561104036E-002/1024.0);
  t = __fma_rn (t, a2, 1.111112158368149E-001/256.0);
  t = __fma_rn (t, a2, 1.428571416261528E-001/64.0);
  t = __fma_rn (t, a2, 2.000000000069858E-001/16.0);
  t = __fma_rn (t, a2, 3.333333333333198E-001/4.0);
  t = t * a2;
  t = __fma_rn (t, a, a_2);
  t = t + a_1;
  return t;
}

static double __internal_exp2i_kernel(int b)
{
  return __hiloint2double((b + 1023) << 20, 0);
}

static double __internal_half(double a)
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi - 0x00100000, ilo);
}

static double __internal_twice(double a)
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi + 0x00100000, ilo);
}

static double __cuda_sin(double a)
{
  double z;
  int i;
  if (__cuda___isinf(a) || (a == 0.0)) {
    return __dmul_rn(a, 0.0);
  }
  z = __internal_trig_reduction_kerneld(a, &i);

  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

static double __cuda_cos(double a)
{
  double z;
  int i;
  if (__cuda___isinf(a)) {
    return __longlong_as_double(0xfff8000000000000ULL);
  }
  z = __internal_trig_reduction_kerneld(a, &i);

  i++;
  if (i & 1) {
    z = __internal_cos_kerneld(z);
  } else {
    z = __internal_sin_kerneld(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}

static void __cuda_sincos(double a, double *sptr, double *cptr)
{
  double t, u, s, c;
  int i;
  if (__cuda___isinf(a)) {
    *sptr = __longlong_as_double(0xfff8000000000000ULL);
    *cptr = __longlong_as_double(0xfff8000000000000ULL);
    return;
  }
  if (a == 0.0) {
    *sptr = a;
    *cptr = 1.0;
    return;
  }
  t = __internal_trig_reduction_kerneld(a, &i);
  u = __internal_cos_kerneld(t);
  t = __internal_sin_kerneld(t);
  if (i & 1) {
    s = u;
    c = t;
  } else {
    s = t;
    c = u;
  }
  if (i & 2) {
    s = -s;
  }
  i++;
  if (i & 2) {
    c = -c;
  }
  *sptr = s;
  *cptr = c;
}

static double __cuda_tan(double a)
{
  double z;
  int i;
  if (__cuda___isinf(a)) {
    return __longlong_as_double(0xfff8000000000000ULL);
  }
  z = __internal_trig_reduction_kerneld(a, &i);

  z = __internal_tan_kerneld(z, i & 1);
  return z;
}

static double __cuda_log(double a)
{
  double m, f, g, u, v, tmp, q, ulo, log_lo, log_hi;
  int ihi, ilo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);

  if ((a > 0.0) && (a < __longlong_as_double(0x7ff0000000000000ULL))) {
    int e = -1023;

    if ((unsigned)ihi < (unsigned)0x00100000) {
      a = a * 18014398509481984.0;
      e -= 54;
      ihi = __double2hiint(a);
      ilo = __double2loint(a);
    }



    e += ((ihi >> 20) & 0x7ff);
    ihi = (ihi & 0x800fffff) | 0x3ff00000;
    m = __hiloint2double (ihi, ilo);
    if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
      m = __internal_half(m);
      e = e + 1;
    }

    f = m - 1.0;
    g = m + 1.0;
    g = 1.0 / g;
    u = f * g;
    u = u + u;

    v = u * u;
    q = 6.7261411553826339E-2/65536.0;
    q = __fma_rn (q, v, 6.6133829643643394E-2/16384.0);
    q = __fma_rn (q, v, 7.6940931149150890E-2/4096.0);
    q = __fma_rn (q, v, 9.0908745692137444E-2/1024.0);
    q = __fma_rn (q, v, 1.1111111499059706E-1/256.0);
    q = __fma_rn (q, v, 1.4285714283305975E-1/64.0);
    q = __fma_rn (q, v, 2.0000000000007223E-1/16.0);
    q = __fma_rn (q, v, 3.3333333333333326E-1/4.0);
    tmp = __internal_twice (f - u);
    tmp = __fma_rn (-u, f, tmp);
    ulo = g * tmp;

    q = q * v;
    q = q * u;

    log_hi = u;
    log_lo = ulo + q;

    q = __fma_rn ( e, 6.9314718055994529e-1, log_hi);
    tmp = __fma_rn (-e, 6.9314718055994529e-1, q);
    tmp = tmp - log_hi;
    log_hi = q;
    log_lo = log_lo - tmp;
    log_lo = __fma_rn (e, 2.3190468138462996e-17, log_lo);
    return log_hi + log_lo;
  } else {
    if (__cuda___isnan(a)) {
      return a + a;
    }

    if (a == 0) {
      return -__longlong_as_double(0x7ff0000000000000ULL);
    }

    if (a == __longlong_as_double(0x7ff0000000000000ULL)) {
      return a;
    }

    return __longlong_as_double(0xfff8000000000000ULL);
  }
}


static double2 __internal_ddadd_xgty (double2 x, double2 y)
{
  double2 z;

  volatile

  double r, s, e;
  r = x.y + y.y;
  e = x.y - r;
  s = ((e + y.y) + y.x) + x.x;
  z.y = e = r + s;
  z.x = (r - e) + s;
  return z;
}


static double2 __internal_ddmul (double2 x, double2 y)
{

  volatile

  double e;
  double2 t, z;
  t.y = x.y * y.y;
  t.x = __fma_rn (x.y, y.y, -t.y);
  t.x = __fma_rn (x.x, y.x, t.x);
  t.x = __fma_rn (x.y, y.x, t.x);
  t.x = __fma_rn (x.x, y.y, t.x);
  z.y = e = t.y + t.x;
  z.x = (t.y - e) + t.x;
  return z;
}

static double2 __internal_log_ext_prec(double a)
{
  double2 res;
  double2 qq, cc, uu, tt;
  double f, g, u, v, q, ulo, tmp, m;
  int ilo, ihi, expo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  expo = (ihi >> 20) & 0x7ff;

  if (expo == 0) {
    a *= 18014398509481984.0;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }
  expo -= 1023;




  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  m = __hiloint2double (ihi, ilo);
  if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
    m = __internal_half(m);
    expo = expo + 1;
  }





  f = m - 1.0;
  g = m + 1.0;
  g = 1.0 / g;
  u = f * g;
  u = u + u;

  v = u * u;
  q = 6.6253631649203309E-2/65536.0;
  q = __fma_rn (q, v, 6.6250935587260612E-2/16384.0);
  q = __fma_rn (q, v, 7.6935437806732829E-2/4096.0);
  q = __fma_rn (q, v, 9.0908878711093280E-2/1024.0);
  q = __fma_rn (q, v, 1.1111111322892790E-1/256.0);
  q = __fma_rn (q, v, 1.4285714284546502E-1/64.0);
  q = __fma_rn (q, v, 2.0000000000003113E-1/16.0);
  q = q * v;

  tmp = __internal_twice (f - u);
  tmp = __fma_rn (-u, f, tmp);
  ulo = g * tmp;

  qq.y = q;
  qq.x = 0.0;
  uu.y = u;
  uu.x = ulo;
  cc.y = 3.3333333333333331E-1/4.0;
  cc.x = -9.8201492846582465E-18/4.0;
  qq = __internal_ddadd_xgty (cc, qq);

  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  qq = __internal_ddmul(qq, uu);
  uu = __internal_ddadd_xgty (uu, qq);
  u = uu.y;
  ulo = uu.x;

  tt.y = expo * 6.9314718055966296e-001;
  tt.x = expo * 2.8235290563031577e-013;

  res = __internal_ddadd_xgty (tt, uu);
  return res;
}

static double __cuda_log2(double a)
{
  double t;
  t = __cuda_log(a);
  return __fma_rn (t, 1.4426950408889634e+0, t * 2.0355273740931033e-17);
}

static double __cuda_log10(double a)
{
  double t;
  t = __cuda_log(a);
  return __fma_rn (t, 4.3429448190325182e-1, t * 1.09831965021676510e-17);
}

static double __cuda_log1p(double a)
{
  double t;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x3fe55555) || ((int)i < (int)0xbfd99999)) {

    t = a + 2.0;
    t = a / t;
    t = -a * t;
    t = __internal_atanh_kernel(a, t);
    return t;
  }
  return __cuda_log (a + 1.0);
}

static double __internal_exp_kernel(double a, int scale)
{
  double t, fac, z;
  int i;

  t = __cuda_rint (a * 1.4426950408889634e+0);
  i = (int)t;
  z = __fma_rn (t, -6.9314718055994529e-1, a);
  z = __fma_rn (t, -2.3190468138462996e-17, z);
  fac = 2.0;
  if (i <= -1021) {
    i += 55;
    fac = 5.5511151231257827e-17;
  }

  t = __internal_expm1_kernel(z);
  z = __internal_exp2i_kernel(i + scale - 1);
  t = __fma_rn (t, z, z);
  t = t * fac;
  return t;
}

static double __cuda_exp(double a)
{
  double t;
  int i;
  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40862e43) || ((int)i < (int)0xC0874911)) {
    t = __internal_exp_kernel(a, 0);
    return t;
  }
  t = ((unsigned int)i >> 31) ? 0.0 : __longlong_as_double(0x7ff0000000000000ULL);
  if (__cuda___isnan(a)) {
    t = a + a;
  }
  return t;
}

static double __cuda_exp2(double a)
{
  double z;
  double t;
  double fac;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40900000) || ((int)i < (int)0xc090cc00)) {
    t = __cuda_rint (a);
    z = a - t;
    i = (int)t;
    fac = 2.0;
    if (i <= -1021) {
      i += 55;
      fac = 5.5511151231257827e-17;
    }

    z = __fma_rn (z, 6.9314718055994529e-1, z * 2.3190468138462996e-17);
    t = __internal_expm1_kernel(z);
    z = __internal_exp2i_kernel(i - 1);
    t = __fma_rn (t, z, z);
    t = t * fac;
    return t;
  }
  t = ((unsigned int)i >> 31) ? 0.0 : __longlong_as_double(0x7ff0000000000000ULL);
  if (__cuda___isnan(a)) {
    t = a + a;
  }
  return t;
}

static double __cuda_exp10(double a)
{
  double z;
  double t;
  double fac;
  int i;

  i = __double2hiint(a);
  if (((unsigned)i < (unsigned)0x40734414) || ((int)i < (int)0xc07439b8)) {
    t = __cuda_rint (a * 3.3219280948873622e+0);
    i = (int)t;
    z = __fma_rn (t, -3.0102999566398120e-1, a);
    z = __fma_rn (t, -(-2.8037281277851704e-18), z);
    fac = 2.0;
    if (i <= -1021) {
      i += 55;
      fac = 5.5511151231257827e-17;
    }

    z = __fma_rn (z, 2.3025850929940459e+0, z * (-2.1707562233822494e-16));
    t = __internal_expm1_kernel(z);
    z = __internal_exp2i_kernel(i - 1);
    t = __fma_rn (t, z, z);
    t = t * fac;
    return t;
  }
  t = ((unsigned int)i >> 31) ? 0.0 : __longlong_as_double(0x7ff0000000000000ULL);
  if (__cuda___isnan(a)) {
    t = a + a;
  }
  return t;
}

static double __cuda_expm1(double a)
{
  double t, z, u;
  int i, j, k;

  k = __double2hiint(a);
  if (((unsigned)k < (unsigned)0x40862e43) || ((int)k < (int)0xc04a8000)) {
    t = __cuda_rint (a * 1.4426950408889634e+0);
    i = (int)t;
    z = __fma_rn (t, -6.9314718055994529e-1, a);
    z = __fma_rn (t, -2.3190468138462996e-17, z);
    k = k + k;
    if ((unsigned)k < (unsigned)0x7fb3e647) {
      z = a;
      i = 0;
    }
    t = __internal_expm1_kernel(z);
    j = i;
    if (i == 1024) j--;
    u = __internal_exp2i_kernel(j);
    a = u - 1.0;
    t = __fma_rn (t, u, a);
    if (i == 1024) t = t + t;
    if (k == 0) t = z;
    return t;
  }
  t = ((unsigned int)k >> 31) ? -1.0 : __longlong_as_double(0x7ff0000000000000ULL);
  if (__cuda___isnan(a)) {
    t = a + a;
  }
  return t;
}

static double __cuda_cosh(double a)
{
  double z;
  int i;

  z = __cuda_fabs(a);
  i = __double2hiint(z);
  if ((unsigned)i < (unsigned)0x408633cf) {
    z = __internal_exp_kernel(z, -2);
    z = __fma_rn(2.0, z, 0.125 / z);
    return z;
  } else {
    if (z > 0.0) a = __int_as_float(0x7f800000);
    return a + a;
  }
}

static double __cuda_sinh(double a)
{
  double s, z;
  s = a;
  a = __cuda_fabs(a);
  if (a < 1.0) {
    double a2 = a * a;

    z = 1.632386098183803E-010;
    z = __fma_rn (z, a2, 2.504854501385687E-008);
    z = __fma_rn (z, a2, 2.755734274788706E-006);
    z = __fma_rn (z, a2, 1.984126976294102E-004);
    z = __fma_rn (z, a2, 8.333333333452911E-003);
    z = __fma_rn (z, a2, 1.666666666666606E-001);
    z = z * a2;
    z = __fma_rn (z, a, a);
  } else if (a < 2.0) {
    z = __cuda_expm1(a);
    z = __internal_half (z + z / (z + 1.0));
  } else {
    z = __internal_exp_kernel(a, -1);
    z = z + (1.0 / (-4.0 * z));
    if (a >= 7.1047586007394398e+2) {
      z = __longlong_as_double(0x7ff0000000000000ULL);
    }
  }
  z = __cuda_copysign(z, s);
  return z;
}

static double __cuda_tanh(double a)
{
  double t;
  t = __cuda_fabs(a);
  if (t >= 0.55) {
    double s;
    s = 1.0 - 2.0 / (__internal_exp_kernel(2.0 * t, 0) + 1.0);
    if (t > 350.0) {
      s = 1.0;
    }
    a = __cuda_copysign(s, a);
  } else {
    double a2;
    a2 = a * a;
    t = 5.102147717274194E-005;
    t = __fma_rn (t, a2, -2.103023983278533E-004);
    t = __fma_rn (t, a2, 5.791370145050539E-004);
    t = __fma_rn (t, a2, -1.453216755611004E-003);
    t = __fma_rn (t, a2, 3.591719696944118E-003);
    t = __fma_rn (t, a2, -8.863194503940334E-003);
    t = __fma_rn (t, a2, 2.186948597477980E-002);
    t = __fma_rn (t, a2, -5.396825387607743E-002);
    t = __fma_rn (t, a2, 1.333333333316870E-001);
    t = __fma_rn (t, a2, -3.333333333333232E-001);
    t = t * a2;
    t = __fma_rn (t, a, a);
    a = __cuda_copysign(t, a);
  }
  return a;
}

static double __internal_atan_kernel(double a)
{
  double t, a2;
  a2 = a * a;
  t = -2.0258553044438358E-005 ;
  t = __fma_rn (t, a2, 2.2302240345758510E-004);
  t = __fma_rn (t, a2, -1.1640717779930576E-003);
  t = __fma_rn (t, a2, 3.8559749383629918E-003);
  t = __fma_rn (t, a2, -9.1845592187165485E-003);
  t = __fma_rn (t, a2, 1.6978035834597331E-002);
  t = __fma_rn (t, a2, -2.5826796814495994E-002);
  t = __fma_rn (t, a2, 3.4067811082715123E-002);
  t = __fma_rn (t, a2, -4.0926382420509971E-002);
  t = __fma_rn (t, a2, 4.6739496199157994E-002);
  t = __fma_rn (t, a2, -5.2392330054601317E-002);
  t = __fma_rn (t, a2, 5.8773077721790849E-002);
  t = __fma_rn (t, a2, -6.6658603633512573E-002);
  t = __fma_rn (t, a2, 7.6922129305867837E-002);
  t = __fma_rn (t, a2, -9.0909012354005225E-002);
  t = __fma_rn (t, a2, 1.1111110678749424E-001);
  t = __fma_rn (t, a2, -1.4285714271334815E-001);
  t = __fma_rn (t, a2, 1.9999999999755019E-001);
  t = __fma_rn (t, a2, -3.3333333333331860E-001);
  t = t * a2;
  t = __fma_rn (t, a, a);
  return t;
}

static double __cuda_atan2(double a, double b)
{
  double t0, t1, t3;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }


  t3 = __cuda_fabs(b);
  t1 = __cuda_fabs(a);
  if (t3 == 0.0 && t1 == 0.0) {
    t3 = __cuda___signbit(b) ? 3.1415926535897931e+0 : 0;
  } else if (__cuda___isinf(t3) && __cuda___isinf(t1)) {
    t3 = __cuda___signbit(b) ? 2.3561944901923448e+0 : 7.8539816339744828e-1;
  } else {
    t0 = __cuda_fmax (t1, t3);
    t1 = __cuda_fmin (t1, t3);
    t3 = t1 / t0;
    t3 = __internal_atan_kernel(t3);

    if (__cuda_fabs(a) > __cuda_fabs(b)) t3 = 1.5707963267948966e+0 - t3;
    if (b < 0.0) t3 = 3.1415926535897931e+0 - t3;
  }
  t3 = __cuda_copysign(t3, a);
  return t3;
}

static double __cuda_atan(double a)
{
  double t0, t1;

  t0 = __cuda_fabs(a);
  t1 = t0;
  if (t0 > 1.0) {
    t1 = 1.0 / t1;
  }

  t1 = __internal_atan_kernel(t1);

  if (t0 > 1.0) {
    t1 = 1.5707963267948966e+0 - t1;
  }
  return __cuda_copysign (t1, a);
}


static double __internal_asin_kernel(double a, double b)
{
  double r;
  r = 6.259798167646803E-002;
  r = __fma_rn (r, b, -7.620591484676952E-002);
  r = __fma_rn (r, b, 6.686894879337643E-002);
  r = __fma_rn (r, b, -1.787828218369301E-002);
  r = __fma_rn (r, b, 1.745227928732326E-002);
  r = __fma_rn (r, b, 1.000422754245580E-002);
  r = __fma_rn (r, b, 1.418108777515123E-002);
  r = __fma_rn (r, b, 1.733194598980628E-002);
  r = __fma_rn (r, b, 2.237350511593569E-002);
  r = __fma_rn (r, b, 3.038188875134962E-002);
  r = __fma_rn (r, b, 4.464285849810986E-002);
  r = __fma_rn (r, b, 7.499999998342270E-002);
  r = __fma_rn (r, b, 1.666666666667375E-001);
  r = r * b;
  return r;
}

static double __cuda_asin(double a)
{
  double fa, t0, t1;
  int ihi, ahi;
  ahi = __double2hiint(a);
  fa = __cuda_fabs(a);
  ihi = __double2hiint(fa);
  if (ihi < 0x3fe26666) {
    t1 = fa * fa;
    t1 = __internal_asin_kernel (fa, t1);
    t1 = __fma_rn (t1, fa, fa);
    t1 = __cuda_copysign(t1, a);
  } else {
    t1 = __fma_rn (-0.5, fa, 0.5);
    t0 = __cuda_sqrt (t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = -2.0 * t0;
    t1 = __fma_rn (t0, t1, 6.1232339957367660e-17);
    t0 = t0 + 7.8539816339744828e-1;
    t1 = t0 + t1;
    t1 = t1 + 7.8539816339744828e-1;
    if (ahi < 0x3ff00000) {
      t1 = __cuda_copysign(t1, a);
    }
  }
  return t1;
}

static double __cuda_acos(double a)
{
  double t0, t1;
  int ihi, ahi;


  if (__cuda___isnan(a)) {
    return a + a;
  }

  ahi = __double2hiint(a);
  t0 = __cuda_fabs (a);
  ihi = __double2hiint(t0);
  if (ihi < 0x3fe26666) {
    t1 = t0 * t0;
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    if ((unsigned)ahi >= (unsigned)0x80000000) {
      t0 = __fma_rn (1.0, t0, +6.1232339957367660e-17);
      t0 = 1.5707963267948966e+0 + t0;
    } else {
      t0 = __fma_rn (1.0, t0, -6.1232339957367660e-17);
      t0 = 1.5707963267948966e+0 - t0;
    }
  } else {
    t1 = __fma_rn (-0.5, t0, 0.5);
    t0 = __cuda_sqrt(t1);
    t1 = __internal_asin_kernel (t0, t1);
    t0 = __fma_rn (t1, t0, t0);
    t0 = 2.0 * t0;
    if ((unsigned)ahi >= (unsigned)0x80000000) {
      t0 = __fma_rn (1.0, t0, -1.2246467991473532e-16);
      t0 = 3.1415926535897931e+0 - t0;
    }
  }
  return t0;
}

static double __cuda_acosh(double a)
{
  double t;

  if (__cuda___isnan(a)) {
    return a + a;
  }

  t = a - 1.0;
  if (__cuda_fabs(t) > 4503599627370496.0) {

    return 6.9314718055994529e-1 + __cuda_log(a);
  } else {
    t = t + __cuda_sqrt(__fma_rn(a, t, t));
    return __cuda_log1p(t);
  }
}

static double __cuda_asinh(double a)
{
# 1164 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx3.h" 3
  double fa, t;
  fa = __cuda_fabs(a);
  if (fa > 1.0e153) {
    t = 6.9314718055994529e-1 + __cuda_log(fa);
  } else {
    t = fa * fa;
    t = __cuda_log1p (fa + t / (1.0 + __cuda_sqrt(1.0 + t)));
  }

  return __cuda_copysign(t, a);
}

static double __cuda_atanh(double a)
{
  double fa, t;

  if (__cuda___isnan(a)) {
    return a + a;
  }

  fa = __cuda_fabs(a);
  t = (2.0 * fa) / (1.0 - fa);
  t = 0.5 * __cuda_log1p(t);

  if (__cuda___isnan(t)) {
    return t;
  }

  if (__cuda___signbit(a)) {
    t = -t;
  }
  return t;
}

static double __cuda_hypot(double a, double b)
{
  double v, w, t;
  if (__cuda___isinf(a) || __cuda___isinf(b)) {
    return __longlong_as_double(0x7ff0000000000000ULL);
  }
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  v = __cuda_fmax(a, b);
  w = __cuda_fmin(a, b);
  t = w / v;
  t = __fma_rn (t, t, 1.0);
  t = v * __cuda_sqrt(t);
  if (v == 0.0) {
    t = v + w;
  }
  return t;
}

static double __cuda_cbrt(double a)
{
  float s;
  double t, r;
  int ilo, ihi, expo, nexpo, denorm;
  if ((a == 0.0) || !(__cuda___finite(a))) {
    return a + a;
  }
  t = __cuda_fabs(a);
  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  denorm = 0;
  if (expo == 0) {

    t = t * 18014398509481984.0;
    denorm = 18;
    ilo = __double2loint(t);
    ihi = __double2hiint(t);
    expo = ((int)((unsigned int)ihi >> 20) & 0x7ff);
  }

  nexpo = __float2int_rn(0.333333333f * (float)(expo - 1022));
  ihi -= (3 * nexpo) << 20;
  r = __hiloint2double(ihi, ilo);

  s = (float)r;
  t = __cuda_exp2f(0.333333333f * __log2f(s));

  t = t - (t - (r / (t * t))) * 3.3333333333333333e-1;
  t = t - (t - (r / (t * t))) * 3.3333333333333333e-1;

  ilo = __double2loint(t);
  ihi = __double2hiint(t);
  ihi += (nexpo - denorm) << 20;
  t = __hiloint2double(ihi, ilo);
  if (__cuda___signbit(a)) {
     t = -t;
  }
  return t;
}

static double __internal_accurate_pow(double a, double b)
{
  double2 loga;
  double2 prod;
  double t_hi, t_lo;
  double tmp;



  double e;



  loga = __internal_log_ext_prec(a);


  if (__cuda_fabs(b) > 1e304) b *= 1.220703125e-4;

  t_hi = loga.y * b;
  t_lo = __fma_rn (loga.y, b, -t_hi);
  t_lo = __fma_rn (loga.x, b, t_lo);
  prod.y = e = t_hi + t_lo;
  prod.x = (t_hi - e) + t_lo;


  tmp = __cuda_exp(prod.y);

  if (!__cuda___isinf(tmp)) {



    tmp = __fma_rn (tmp, prod.x, tmp);
  }
  return tmp;
}

static double __cuda_pow(double a, double b)
{
  int bIsOddInteger;
  double t;
  if (a == 1.0 || b == 0.0) {
    return 1.0;
  }
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  if (a == __longlong_as_double(0x7ff0000000000000ULL)) {
    return __cuda___signbit(b) ? 0.0 : __longlong_as_double(0x7ff0000000000000ULL);
  }
  if (__cuda___isinf(b)) {
    if (a == -1.0) {
      return 1.0;
    }
    t = __cuda_fabs(a) > 1.0 ? __longlong_as_double(0x7ff0000000000000ULL) : 0.0;
    if (b < 0.0) {
      t = 1.0 / t;
    }
    return t;
  }
  bIsOddInteger = (b - (2.0 * __cuda_floor(0.5 * b))) == 1.0;
  if (a == 0.0) {
    t = bIsOddInteger ? a : 0.0;
    if (b < 0.0) {
      t = 1.0 / t;
    }
    return t;
  }
  if (a == -__longlong_as_double(0x7ff0000000000000ULL)) {
    t = (b < 0.0) ? -1.0/a : -a;
    if (bIsOddInteger) {
      t = __longlong_as_double(__double_as_longlong(t)^0x8000000000000000ULL);
    }
    return t;
  }
  if ((a < 0.0) && (b != __cuda_trunc(b))) {
    return __longlong_as_double(0xfff8000000000000ULL);
  }
  t = __cuda_fabs(a);
  t = __internal_accurate_pow(t, b);
  if ((a < 0.0) && bIsOddInteger) {
    t = __longlong_as_double(__double_as_longlong(t) ^ 0x8000000000000000ULL);
  }
  return t;
}

static double __cuda_erf(double a)
{
  double t, r, q;

  if (__cuda___isnan(a)) {
    return a + a;
  }

  t = __cuda_fabs(a);
  if (t >= 1.0) {
    r = -1.28836351230756500E-019;
    r = __fma_rn (r, t, 1.30597472161093370E-017);
    r = __fma_rn (r, t, -6.33924401259620500E-016);
    r = __fma_rn (r, t, 1.96231865908940140E-014);
    r = __fma_rn (r, t, -4.35272243559990750E-013);
    r = __fma_rn (r, t, 7.37083927929352150E-012);
    r = __fma_rn (r, t, -9.91402142550461630E-011);
    r = __fma_rn (r, t, 1.08817017167760820E-009);
    r = __fma_rn (r, t, -9.93918713097634620E-009);
    r = __fma_rn (r, t, 7.66739923255145500E-008);
    r = __fma_rn (r, t, -5.05440278302806720E-007);
    r = __fma_rn (r, t, 2.87474157099000620E-006);
    r = __fma_rn (r, t, -1.42246725399722510E-005);
    r = __fma_rn (r, t, 6.16994555079419460E-005);
    r = __fma_rn (r, t, -2.36305221938908790E-004);
    r = __fma_rn (r, t, 8.05032844055371070E-004);
    r = __fma_rn (r, t, -2.45833366629108140E-003);
    r = __fma_rn (r, t, 6.78340988296706120E-003);
    r = __fma_rn (r, t, -1.70509103597554640E-002);
    r = __fma_rn (r, t, 3.93322852515666300E-002);
    r = __fma_rn (r, t, -8.37271292613764040E-002);
    r = __fma_rn (r, t, 1.64870423707623280E-001);
    r = __fma_rn (r, t, -2.99729521787681470E-001);
    r = __fma_rn (r, t, 4.99394435612628580E-001);
    r = __fma_rn (r, t, -7.52014596480123030E-001);
    r = __fma_rn (r, t, 9.99933138314926250E-001);
    r = __fma_rn (r, t, -1.12836725321102670E+000);
    r = __fma_rn (r, t, 9.99998988715182450E-001);
    q = __internal_exp_kernel(-t * t, 0);
    r = __fma_rn (r, -q, 1.0);
    if (t >= 6.5) {
      r = 1.0;
    }
    a = __cuda_copysign (r, a);
  } else {
    q = t * t;
    r = -7.77946848895991420E-010;
    r = __fma_rn (r, q, 1.37109803980285950E-008);
    r = __fma_rn (r, q, -1.62063137584932240E-007);
    r = __fma_rn (r, q, 1.64471315712790040E-006);
    r = __fma_rn (r, q, -1.49247123020098620E-005);
    r = __fma_rn (r, q, 1.20552935769006260E-004);
    r = __fma_rn (r, q, -8.54832592931448980E-004);
    r = __fma_rn (r, q, 5.22397760611847340E-003);
    r = __fma_rn (r, q, -2.68661706431114690E-002);
    r = __fma_rn (r, q, 1.12837916709441850E-001);
    r = __fma_rn (r, q, -3.76126389031835210E-001);
    r = __fma_rn (r, q, 1.12837916709551260E+000);
    a = r * a;
  }
  return a;
}

static double __cuda_erfinv(double a)
{
  double fa, t;

  fa = fabs(a);
  if (fa >= 1.0) {
    t = __longlong_as_double(0xfff8000000000000ULL);
    if (fa == 1.0) {
      t = a * __longlong_as_double(0x7ff0000000000000ULL);
    }
  } else if (fa >= 0.9375) {




    double p, q;

    t = __cuda_log1p(-fa);
    t = __cuda_rsqrt(-t);
    p = 2.7834010353747001060e-3;
    p = __fma_rn (p, t, 8.6030097526280260580e-1);
    p = __fma_rn (p, t, 2.1371214997265515515e+0);
    p = __fma_rn (p, t, 3.1598519601132090206e+0);
    p = __fma_rn (p, t, 3.5780402569085996758e+0);
    p = __fma_rn (p, t, 1.5335297523989890804e+0);
    p = __fma_rn (p, t, 3.4839207139657522572e-1);
    p = __fma_rn (p, t, 5.3644861147153648366e-2);
    p = __fma_rn (p, t, 4.3836709877126095665e-3);
    p = __fma_rn (p, t, 1.3858518113496718808e-4);
    p = __fma_rn (p, t, 1.1738352509991666680e-6);
    q = t+ 2.2859981272422905412e+0;
    q = __fma_rn (q, t, 4.3859045256449554654e+0);
    q = __fma_rn (q, t, 4.6632960348736635331e+0);
    q = __fma_rn (q, t, 3.9846608184671757296e+0);
    q = __fma_rn (q, t, 1.6068377709719017609e+0);
    q = __fma_rn (q, t, 3.5609087305900265560e-1);
    q = __fma_rn (q, t, 5.3963550303200816744e-2);
    q = __fma_rn (q, t, 4.3873424022706935023e-3);
    q = __fma_rn (q, t, 1.3858762165532246059e-4);
    q = __fma_rn (q, t, 1.1738313872397777529e-6);
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {




    double p, q;

    t = __fma_rn (a, a, -.87890625);
    p = .21489185007307062000e+0;
    p = __fma_rn (p, t, -.64200071507209448655e+1);
    p = __fma_rn (p, t, .29631331505876308123e+2);
    p = __fma_rn (p, t, -.47644367129787181803e+2);
    p = __fma_rn (p, t, .34810057749357500873e+2);
    p = __fma_rn (p, t, -.12954198980646771502e+2);
    p = __fma_rn (p, t, .25349389220714893917e+1);
    p = __fma_rn (p, t, -.24758242362823355486e+0);
    p = __fma_rn (p, t, .94897362808681080020e-2);
    q = t -.12831383833953226499e+2;
    q = __fma_rn (q, t, .41409991778428888716e+2);
    q = __fma_rn (q, t, -.53715373448862143349e+2);
    q = __fma_rn (q, t, .33880176779595142685e+2);
    q = __fma_rn (q, t, -.11315360624238054876e+2);
    q = __fma_rn (q, t, .20369295047216351160e+1);
    q = __fma_rn (q, t, -.18611650627372178511e+0);
    q = __fma_rn (q, t, .67544512778850945940e-2);
    p = p / q;
    t = a * p;
  } else {




    double p, q;

    t = __fma_rn (a, a, -.5625);
    p = -.23886240104308755900e+2;
    p = __fma_rn (p, t, .45560204272689128170e+3);
    p = __fma_rn (p, t, -.22977467176607144887e+4);
    p = __fma_rn (p, t, .46631433533434331287e+4);
    p = __fma_rn (p, t, -.43799652308386926161e+4);
    p = __fma_rn (p, t, .19007153590528134753e+4);
    p = __fma_rn (p, t, -.30786872642313695280e+3);
    q = t -.83288327901936570000e+2;
    q = __fma_rn (q, t, .92741319160935318800e+3);
    q = __fma_rn (q, t, -.35088976383877264098e+4);
    q = __fma_rn (q, t, .59039348134843665626e+4);
    q = __fma_rn (q, t, -.48481635430048872102e+4);
    q = __fma_rn (q, t, .18997769186453057810e+4);
    q = __fma_rn (q, t, -.28386514725366621129e+3);
    p = p / q;
    t = a * p;
  }
  return t;
}

static double __cuda_erfcinv(double a)
{
  double t;

  if (__cuda___isnan(a)) return a + a;

  if (a <= 0.0) {
    t = __longlong_as_double(0xfff8000000000000ULL);
    if (a == 0.0) {
      t = (1.0 - a) * __longlong_as_double(0x7ff0000000000000ULL);
    }
  }
  else if (a >= 0.0625) {
    t = __cuda_erfinv (1.0 - a);
  }
  else if (a >= 1e-100) {




    double p, q;
    t = __cuda_log(a);
    t = __cuda_rsqrt(-t);
    p = 2.7834010353747001060e-3;
    p = __fma_rn (p, t, 8.6030097526280260580e-1);
    p = __fma_rn (p, t, 2.1371214997265515515e+0);
    p = __fma_rn (p, t, 3.1598519601132090206e+0);
    p = __fma_rn (p, t, 3.5780402569085996758e+0);
    p = __fma_rn (p, t, 1.5335297523989890804e+0);
    p = __fma_rn (p, t, 3.4839207139657522572e-1);
    p = __fma_rn (p, t, 5.3644861147153648366e-2);
    p = __fma_rn (p, t, 4.3836709877126095665e-3);
    p = __fma_rn (p, t, 1.3858518113496718808e-4);
    p = __fma_rn (p, t, 1.1738352509991666680e-6);
    q = t+ 2.2859981272422905412e+0;
    q = __fma_rn (q, t, 4.3859045256449554654e+0);
    q = __fma_rn (q, t, 4.6632960348736635331e+0);
    q = __fma_rn (q, t, 3.9846608184671757296e+0);
    q = __fma_rn (q, t, 1.6068377709719017609e+0);
    q = __fma_rn (q, t, 3.5609087305900265560e-1);
    q = __fma_rn (q, t, 5.3963550303200816744e-2);
    q = __fma_rn (q, t, 4.3873424022706935023e-3);
    q = __fma_rn (q, t, 1.3858762165532246059e-4);
    q = __fma_rn (q, t, 1.1738313872397777529e-6);
    t = p / (q * t);
  }
  else {




    double p, q;
    t = __cuda_log(a);
    t = __cuda_rsqrt(-t);
    p = 6.9952990607058154858e-1;
    p = __fma_rn (p, t, 1.9507620287580568829e+0);
    p = __fma_rn (p, t, 8.2810030904462690216e-1);
    p = __fma_rn (p, t, 1.1279046353630280005e-1);
    p = __fma_rn (p, t, 6.0537914739162189689e-3);
    p = __fma_rn (p, t, 1.3714329569665128933e-4);
    p = __fma_rn (p, t, 1.2964481560643197452e-6);
    p = __fma_rn (p, t, 4.6156006321345332510e-9);
    p = __fma_rn (p, t, 4.5344689563209398450e-12);
    q = t+ 1.5771922386662040546e+0;
    q = __fma_rn (q, t, 2.1238242087454993542e+0);
    q = __fma_rn (q, t, 8.4001814918178042919e-1);
    q = __fma_rn (q, t, 1.1311889334355782065e-1);
    q = __fma_rn (q, t, 6.0574830550097140404e-3);
    q = __fma_rn (q, t, 1.3715891988350205065e-4);
    q = __fma_rn (q, t, 1.2964671850944981713e-6);
    q = __fma_rn (q, t, 4.6156017600933592558e-9);
    q = __fma_rn (q, t, 4.5344687377088206783e-12);
    t = p / (q * t);
  }
  return t;
}

static double __cuda_erfc(double a)
{
  double p, q, h, l;
  int ahi;

  ahi = __double2hiint(a);
  if (ahi < (int)0x3fe80000) {
    return 1.0 - __cuda_erf(a);
  }
  if (a > 27.3) {
    return 0.0;
  }
  if (ahi < (int)0x40140000) {
    p = 5.64189549785304440E-001;
    p = __fma_rn (p, a, 8.17405083437083490E+000);
    p = __fma_rn (p, a, 5.68958722557864720E+001);
    p = __fma_rn (p, a, 2.42568747802647010E+002);
    p = __fma_rn (p, a, 6.80381374390412930E+002);
    p = __fma_rn (p, a, 1.25873132236024590E+003);
    p = __fma_rn (p, a, 1.43925353963809330E+003);
    p = __fma_rn (p, a, 8.15949420587659230E+002);
    q = a+ 1.44881247113239940E+001;
    q = __fma_rn (q, a, 1.01345387970210510E+002);
    q = __fma_rn (q, a, 4.37184684964009650E+002);
    q = __fma_rn (q, a, 1.25588209225251330E+003);
    q = __fma_rn (q, a, 2.43864015012647630E+003);
    q = __fma_rn (q, a, 3.10570469008816280E+003);
    q = __fma_rn (q, a, 2.35995386578953550E+003);
    q = __fma_rn (q, a, 8.15949420692539320E+002);
  } else {
    p = 5.64189583545675280E-001;
    p = __fma_rn (p, a, 2.04728556066513970E+000);
    p = __fma_rn (p, a, 6.75128409662943610E+000);
    p = __fma_rn (p, a, 1.10459345071747900E+001);
    p = __fma_rn (p, a, 1.22570382896313600E+001);
    p = __fma_rn (p, a, 6.01884641114116460E+000);
    q = a+ 3.62871917534986780E+000;
    q = __fma_rn (q, a, 1.24663395327043550E+001);
    q = __fma_rn (q, a, 2.13927672803974790E+001);
    q = __fma_rn (q, a, 2.72082423532866070E+001);
    q = __fma_rn (q, a, 1.86422906830006700E+001);
    q = __fma_rn (q, a, 6.13809834548870550E+000);
  }
  p = p / q;
  h = a * a;
  l = __fma_rn (a, a, -h);
  q = __internal_exp_kernel(-h, 0);
  q = __fma_rn (l, -q, q);
  p = p * q;
  return p;
}


static double __internal_tgamma_kernel(double a)
{
  double t;
  t = -4.42689340712524750E-010;
  t = __fma_rn (t, a, -2.02665918466589540E-007);
  t = __fma_rn (t, a, 1.13812117211195270E-006);
  t = __fma_rn (t, a, -1.25077348166307480E-006);
  t = __fma_rn (t, a, -2.01365017404087710E-005);
  t = __fma_rn (t, a, 1.28050126073544860E-004);
  t = __fma_rn (t, a, -2.15241408115274180E-004);
  t = __fma_rn (t, a, -1.16516754597046040E-003);
  t = __fma_rn (t, a, 7.21894322484663810E-003);
  t = __fma_rn (t, a, -9.62197153268626320E-003);
  t = __fma_rn (t, a, -4.21977345547223940E-002);
  t = __fma_rn (t, a, 1.66538611382503560E-001);
  t = __fma_rn (t, a, -4.20026350341054440E-002);
  t = __fma_rn (t, a, -6.55878071520257120E-001);
  t = __fma_rn (t, a, 5.77215664901532870E-001);
  t = __fma_rn (t, a, 1.00000000000000000E+000);
  return t;
}


static double __internal_stirling_poly(double a)
{
  double x = 1.0 / a;
  double z = 0.0;
  z = __fma_rn (z, x, 8.3949872067208726e-004);
  z = __fma_rn (z, x, -5.1717909082605919e-005);
  z = __fma_rn (z, x, -5.9216643735369393e-004);
  z = __fma_rn (z, x, 6.9728137583658571e-005);
  z = __fma_rn (z, x, 7.8403922172006662e-004);
  z = __fma_rn (z, x, -2.2947209362139917e-004);
  z = __fma_rn (z, x, -2.6813271604938273e-003);
  z = __fma_rn (z, x, 3.4722222222222220e-003);
  z = __fma_rn (z, x, 8.3333333333333329e-002);
  z = __fma_rn (z, x, 1.0000000000000000e+000);
  return z;
}

static double __internal_tgamma_stirling(double a)
{
  if (a < 1.7162437695630274e+002) {

    volatile double t_hi, t_lo, e;



    double2 loga, prod;
    double z = __internal_stirling_poly (a);
    double b = a - 0.5;


    loga = __internal_log_ext_prec(a);


    t_hi = loga.y * b;
    t_lo = __fma_rn (loga.y, b, -t_hi);
    t_lo = __fma_rn (loga.x, b, t_lo);
    prod.y = e = t_hi + t_lo;
    prod.x = (t_hi - e) + t_lo;


    loga.y = -a;
    loga.x = 0.0;
    prod = __internal_ddadd_xgty (prod, loga);


    a = __cuda_exp(prod.y);

    if (!__cuda___isinf(a)) {



      a = __fma_rn (a, prod.x, a);
    }
    a = __fma_rn (a, 2.5066282746310007e+0, a * (-1.8328579980459167e-16));
    return a * z;
  } else {
    return __longlong_as_double(0x7ff0000000000000ULL);
  }
}

static double __cuda_tgamma(double a)
{
  double s, xx, x = a;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  if (__cuda_fabs(x) < 15.0) {




    if (x >= 0.0) {
      s = 1.0;
      xx = x;
      while (xx > 1.5) {
        s = __fma_rn(s, xx, -s);
        xx = xx - 1.0;
      }
      if (x >= 0.5) {
        xx = xx - 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      if (x < 0.5) {
        xx = xx * x;
      }
      s = s / xx;
    } else {
      xx = x;
      s = xx;
      if (x == __cuda_floor(x)) {
        return __longlong_as_double(0xfff8000000000000ULL);
      }
      while (xx < -0.5) {
        s = __fma_rn (s, xx, s);
        xx = xx + 1.0;
      }
      xx = __internal_tgamma_kernel (xx);
      s = s * xx;
      s = 1.0 / s;
    }
    return s;
  } else {
    if (x >= 0.0) {
      return __internal_tgamma_stirling (x);
    } else {
      double t;
      int quot;
      if (x == __cuda_floor(x)) {
        return __longlong_as_double(0xfff8000000000000ULL);
      }
      if (x < -185.0) {
        int negative;
        x = __cuda_floor(x);
        negative = ((x - (2.0 * __cuda_floor(0.5 * x))) == 1.0);
        return negative ? __longlong_as_double(0x8000000000000000ULL) : 0.0;
      }

      xx = __cuda_rint (__internal_twice(x));
      quot = (int)xx;
      xx = __fma_rn (-0.5, xx, x);
      xx = xx * 3.1415926535897931e+0;
      if (quot & 1) {
        xx = __internal_cos_kerneld (xx);
      } else {
        xx = __internal_sin_kerneld (xx);
      }
      if (quot & 2) {
        xx = -xx;
      }
      x = __cuda_fabs (x);
      s = __cuda_exp (-x);
      t = x - 0.5;
      if (x > 140.0) t = __internal_half(t);
      t = __cuda_pow (x, t);
      if (x > 140.0) s = s * t;
      s = s * __internal_stirling_poly (x);
      s = s * x;
      s = s * xx;
      s = 1.0 / s;
      s = __fma_rn (s, 1.2533141373155003e+0, (-9.1642899902295834e-17) * s);
      s = s / t;
      return s;
    }
  }
}

static double __internal_lgamma_pos(double a)
{
  double sum;
  double s, t;

  if (a == __longlong_as_double(0x7ff0000000000000ULL)) {
    return a;
  }
  if (a >= 3.0) {
    if (a >= 8.0) {



      s = 1.0 / a;
      t = s * s;
      sum = -0.1633436431e-2;
      sum = __fma_rn (sum, t, 0.83645878922e-3);
      sum = __fma_rn (sum, t, -0.5951896861197e-3);
      sum = __fma_rn (sum, t, 0.793650576493454e-3);
      sum = __fma_rn (sum, t, -0.277777777735865004e-2);
      sum = __fma_rn (sum, t, 0.833333333333331018375e-1);
      sum = __fma_rn (sum, s, 0.918938533204672);
      s = __internal_half(__cuda_log (a));
      t = a - 0.5;
      s = s * t;
      t = s - a;
      s = s + sum;
      t = t + s;
      return t;
    } else {
      a = a - 3.0;
      s = -4.02412642744125560E+003;
      s = __fma_rn (s, a, -2.97693796998962000E+005);
      s = __fma_rn (s, a, -6.38367087682528790E+006);
      s = __fma_rn (s, a, -5.57807214576539320E+007);
      s = __fma_rn (s, a, -2.24585140671479230E+008);
      s = __fma_rn (s, a, -4.70690608529125090E+008);
      s = __fma_rn (s, a, -7.62587065363263010E+008);
      s = __fma_rn (s, a, -9.71405112477113250E+008);
      t = a -1.02277248359873170E+003;
      t = __fma_rn (t, a, -1.34815350617954480E+005);
      t = __fma_rn (t, a, -4.64321188814343610E+006);
      t = __fma_rn (t, a, -6.48011106025542540E+007);
      t = __fma_rn (t, a, -4.19763847787431360E+008);
      t = __fma_rn (t, a, -1.25629926018000720E+009);
      t = __fma_rn (t, a, -1.40144133846491690E+009);
      t = s / t;
      t = t + a;
      return t;
    }
  } else if (a >= 1.5) {
    a = a - 2.0;
    t = 9.84839283076310610E-009;
    t = __fma_rn (t, a, -6.69743850483466500E-008);
    t = __fma_rn (t, a, 2.16565148880011450E-007);
    t = __fma_rn (t, a, -4.86170275781575260E-007);
    t = __fma_rn (t, a, 9.77962097401114400E-007);
    t = __fma_rn (t, a, -2.03041287574791810E-006);
    t = __fma_rn (t, a, 4.36119725805364580E-006);
    t = __fma_rn (t, a, -9.43829310866446590E-006);
    t = __fma_rn (t, a, 2.05106878496644220E-005);
    t = __fma_rn (t, a, -4.49271383742108440E-005);
    t = __fma_rn (t, a, 9.94570466342226000E-005);
    t = __fma_rn (t, a, -2.23154589559238440E-004);
    t = __fma_rn (t, a, 5.09669559149637430E-004);
    t = __fma_rn (t, a, -1.19275392649162300E-003);
    t = __fma_rn (t, a, 2.89051032936815490E-003);
    t = __fma_rn (t, a, -7.38555102806811700E-003);
    t = __fma_rn (t, a, 2.05808084278121250E-002);
    t = __fma_rn (t, a, -6.73523010532073720E-002);
    t = __fma_rn (t, a, 3.22467033424113040E-001);
    t = __fma_rn (t, a, 4.22784335098467190E-001);
    t = t * a;
    return t;
  } else if (a >= 0.7) {
    a = 1.0 - a;
    t = 1.17786911519331130E-002;
    t = __fma_rn (t, a, 3.89046747413522300E-002);
    t = __fma_rn (t, a, 5.90045711362049900E-002);
    t = __fma_rn (t, a, 6.02143305254344420E-002);
    t = __fma_rn (t, a, 5.61652708964839180E-002);
    t = __fma_rn (t, a, 5.75052755193461370E-002);
    t = __fma_rn (t, a, 6.21061973447320710E-002);
    t = __fma_rn (t, a, 6.67614724532521880E-002);
    t = __fma_rn (t, a, 7.14856037245421020E-002);
    t = __fma_rn (t, a, 7.69311251313347100E-002);
    t = __fma_rn (t, a, 8.33503129714946310E-002);
    t = __fma_rn (t, a, 9.09538288991182800E-002);
    t = __fma_rn (t, a, 1.00099591546322310E-001);
    t = __fma_rn (t, a, 1.11334278141734510E-001);
    t = __fma_rn (t, a, 1.25509666613462880E-001);
    t = __fma_rn (t, a, 1.44049896457704160E-001);
    t = __fma_rn (t, a, 1.69557177031481600E-001);
    t = __fma_rn (t, a, 2.07385551032182120E-001);
    t = __fma_rn (t, a, 2.70580808427600350E-001);
    t = __fma_rn (t, a, 4.00685634386517050E-001);
    t = __fma_rn (t, a, 8.22467033424113540E-001);
    t = __fma_rn (t, a, 5.77215664901532870E-001);
    t = t * a;
    return t;
  } else {
    t= -9.04051686831357990E-008;
    t = __fma_rn (t, a, 7.06814224969349250E-007);
    t = __fma_rn (t, a, -3.80702154637902830E-007);
    t = __fma_rn (t, a, -2.12880892189316100E-005);
    t = __fma_rn (t, a, 1.29108470307156190E-004);
    t = __fma_rn (t, a, -2.15932815215386580E-004);
    t = __fma_rn (t, a, -1.16484324388538480E-003);
    t = __fma_rn (t, a, 7.21883433044470670E-003);
    t = __fma_rn (t, a, -9.62194579514229560E-003);
    t = __fma_rn (t, a, -4.21977386992884450E-002);
    t = __fma_rn (t, a, 1.66538611813682460E-001);
    t = __fma_rn (t, a, -4.20026350606819980E-002);
    t = __fma_rn (t, a, -6.55878071519427450E-001);
    t = __fma_rn (t, a, 5.77215664901523870E-001);
    t = t * a;
    t = __fma_rn (t, a, a);
    return -__cuda_log (t);
  }
}

static double __cuda_lgamma(double a)
{
  double t;
  double i;
  long long int quot;
  if (__cuda___isnan(a)) {
    return a + a;
  }
  t = __internal_lgamma_pos(__cuda_fabs(a));
  if (a >= 0.0) return t;
  a = __cuda_fabs(a);
  i = __cuda_floor(a);
  if (a == i) return __longlong_as_double(0x7ff0000000000000ULL);
  if (a < 1e-19) return -__cuda_log(a);
  i = __cuda_rint (2.0 * a);
  quot = (long long int)i;
  i = __fma_rn (-0.5, i, a);
  i = i * 3.1415926535897931e+0;
  if (quot & 1) {
    i = __internal_cos_kerneld(i);
  } else {
    i = __internal_sin_kerneld(i);
  }
  i = __cuda_fabs(i);
  t = __cuda_log(3.1415926535897931e+0 / (i * a)) - t;
  return t;
}

static double __cuda_ldexp(double a, int b)
{
  double fa = __cuda_fabs (a);
  if ((fa == 0.0) || (fa == __longlong_as_double(0x7ff0000000000000ULL)) || (!(fa <= __longlong_as_double(0x7ff0000000000000ULL)))) {
    return a + a;
  }
  if (b == 0) {
    return a;
  }
  if (b > 2200) b = 2200;
  if (b < -2200) b = -2200;
  if (__cuda_abs (b) < 1022) {
    return a * __internal_exp2i_kernel(b);
  }
  if (__cuda_abs (b) < 2044) {
    int bhalf = b / 2;
    return a * __internal_exp2i_kernel (bhalf) *
           __internal_exp2i_kernel (b - bhalf);
  } else {
    int bquarter = b / 4;
    double t = __internal_exp2i_kernel(bquarter);
    return a * t * t * t *__internal_exp2i_kernel (b - 3 * bquarter);
  }
}

static double __cuda_scalbn(double a, int b)
{

  return __cuda_ldexp(a, b);
}

static double __cuda_scalbln(double a, long int b)
{


  if (b < -2147483648L) b = -2147483648L;
  if (b > 2147483647L) b = 2147483647L;

  return __cuda_scalbn(a, (int)b);
}

static double __cuda_frexp(double a, int *b)
{
  double fa = __cuda_fabs(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < 2.22507385850720140e-308) {
    a *= 18014398509481984.0;
    denorm = 54;
  } else {
    denorm = 0;
  }
  expo = (__double2hiint(a) >> 20) & 0x7ff;
  if ((fa == 0.0) || (expo == 0x7ff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 1022;
    a = __longlong_as_double((__double_as_longlong(a) & 0x800fffffffffffffULL)|
                              0x3fe0000000000000ULL);
  }
  *b = expo;
  return a;
}

static double __cuda_modf(double a, double *b)
{
  double t;
  if (__cuda___finite(a)) {
    t = __cuda_trunc(a);
    *b = t;
    t = a - t;
    return __cuda_copysign(t, a);
  } else if (__cuda___isinf(a)) {
    t = 0.0;
    *b = a;
    return __cuda_copysign(t, a);
  } else {
    *b = a + a;
    return a + a;
  }
}

static double __cuda_fmod(double a, double b)
{
  double orig_a = a;
  double orig_b = b;
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (!((a <= __longlong_as_double(0x7ff0000000000000ULL)) && (b <= __longlong_as_double(0x7ff0000000000000ULL)))) {
      return orig_a + orig_b;
  }
  if (a == __longlong_as_double(0x7ff0000000000000ULL) || b == 0.0) {
    return __longlong_as_double(0xfff8000000000000ULL);
  } else if (a >= b) {
    int bhi = __double2hiint(b);
    int blo = __double2loint(b);
    int ahi = __double2hiint(a);
    double scaled_b = 0.0;
    if (b < 2.22507385850720140e-308) {
      double t = b;
      while ((t < a) && (t < 2.22507385850720140e-308)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= 2.22507385850720140e-308) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5;
    }
    return __cuda_copysign (a, orig_a);
  } else {
    return orig_a;
  }
}

static double __cuda_remainder(double a, double b)
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot0 = 0;
  int bhi;
  int blo;
  int ahi;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    return a + b;
  }
  orig_a = a;
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (a == __longlong_as_double(0x7ff0000000000000ULL) || b == 0.0) {
    return __longlong_as_double(0xfff8000000000000ULL);
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < 2.22507385850720140e-308) {
      double t = b;
      while ((t < a) && (t < 2.22507385850720140e-308)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= 2.22507385850720140e-308) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5;
    }
  }

  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  return a;
}

static double __cuda_remquo(double a, double b, int *c)
{
  double orig_a;
  double twoa = 0.0;
  unsigned int quot = 0;
  unsigned int sign;
  int bhi;
  int blo;
  int ahi;
  if (__cuda___isnan(a) || __cuda___isnan(b)) {
    *c = quot;
    return a + b;
  }
  orig_a = a;
  sign = 0 - (__cuda___signbit(a) != __cuda___signbit(b));
  a = __cuda_fabs(a);
  b = __cuda_fabs(b);
  if (a == __longlong_as_double(0x7ff0000000000000ULL) || b == 0.0) {
    *c = quot;
    return __longlong_as_double(0xfff8000000000000ULL);
  } else if (a >= b) {
    double scaled_b = 0.0;
    bhi = __double2hiint(b);
    blo = __double2loint(b);
    ahi = __double2hiint(a);
    if (b < 2.22507385850720140e-308) {
      double t = b;
      while ((t < a) && (t < 2.22507385850720140e-308)) {
        t = t + t;
      }
      bhi = __double2hiint(t);
      blo = __double2loint(t);
      scaled_b = t;
    }
    if (a >= 2.22507385850720140e-308) {
      scaled_b = __hiloint2double ((bhi & 0x000fffff)|(ahi & 0x7ff00000), blo);
    }
    if (scaled_b > a) {
      scaled_b *= 0.5;
    }
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5;
    }
  }

  twoa = a + a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
  }
  bhi = __double2hiint(a);
  blo = __double2loint(a);
  ahi = __double2hiint(orig_a);
  a = __hiloint2double((ahi & 0x80000000) ^ bhi, blo);
  quot = quot & (~((~0)<<3));
  quot = quot ^ sign;
  quot = quot - sign;
  *c = quot;
  return a;
}

static double __cuda_nextafter(double a, double b)
{
  unsigned long long int ia;
  unsigned long long int ib;
  ia = __double_as_longlong(a);
  ib = __double_as_longlong(b);
  if (__cuda___isnan(a) || __cuda___isnan(b)) return a + b;
  if (((ia | ib) << 1) == 0ULL) return b;
  if ((ia + ia) == 0ULL) {
    return __cuda_copysign (__longlong_as_double(0x0000000000000001ULL), b);
  }
  if ((a < b) && (a < 0.0)) ia--;
  if ((a < b) && (a > 0.0)) ia++;
  if ((a > b) && (a < 0.0)) ia++;
  if ((a > b) && (a > 0.0)) ia--;
  a = __longlong_as_double(ia);
  return a;
}

static double __cuda_nan(const char *s)
{
  unsigned long long i = 0;
  int c;
  int ovfl = 0;
  int invld = 0;
  if (*s == '0') {
    s++;
    if ((*s == 'x') || (*s == 'X')) {
      s++;
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x0fffffffffffffffULL) {
          ovfl = 1;
        }
        c = (((*s) >= 'A') && ((*s) <= 'F')) ? (*s + 'a' - 'A') : (*s);
        if ((c >= 'a') && (c <= 'f')) {
          c = c - 'a' + 10;
          i = i * 16 + c;
        } else if ((c >= '0') && (c <= '9')) {
          c = c - '0';
          i = i * 16 + c;
        } else {
          invld = 1;
        }
        s++;
      }
    } else {
      while (*s == '0') s++;
      while (*s) {
        if (i > 0x1fffffffffffffffULL) {
          ovfl = 1;
        }
        c = *s;
        if ((c >= '0') && (c <= '7')) {
          c = c - '0';
          i = i * 8 + c;
        } else {
          invld = 1;
        }
        s++;
      }
    }
  } else {
    while (*s) {
      c = *s;
      if ((i > 1844674407370955161ULL) ||
          ((i == 1844674407370955161ULL) && (c > '5'))) {
        ovfl = 1;
      }
      if ((c >= '0') && (c <= '9')) {
        c = c - '0';
        i = i * 10 + c;
      } else {
        invld = 1;
      }
      s++;
    }
  }
  if (ovfl) {
    i = ~0ULL;
  }
  if (invld) {
    i = 0ULL;
  }
  i = (i & 0x000fffffffffffffULL) | 0x7ff8000000000000ULL;
  return __longlong_as_double(i);
}

static double __cuda_round(double a)
{
  double fa = __cuda_fabs(a);
  if (fa > 4503599627370496.0) {
    return a;
  } else {
    double u = __cuda_floor(fa + 0.5);
    if (fa < 0.5) u = 0;
    return __cuda_copysign(u, a);
  }
}

static long long int __cuda_llround(double a)
{

  if (a >= 9223372036854775807.0) return 0x7fffffffffffffffLL;
  if (a <= -9223372036854775808.0) return 0x8000000000000000LL;

  return (long long int)(__cuda_round(a));
}

static long int __cuda_lround(double a)
{

  return (long int)(__cuda_llround(a));
# 2326 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx3.h" 3
}

static double __cuda_fdim(double a, double b)
{
  double t;
  t = a - b;
  if (a <= b) {
    t = 0.0;
  }
  return t;
}

static int __cuda_ilogb(double a)
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__cuda___isnan(a)) return -((int)((unsigned int)-1 >> 1))-1;
  if (__cuda___isinf(a)) return ((int)((unsigned int)-1 >> 1));
  if (a == 0.0) return -((int)((unsigned int)-1 >> 1))-1;
  a = __cuda_fabs(a);
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= 2.22507385850720140e-308) {
    return ((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    int expo = -1022;
    while (i < 0x0010000000000000ULL) {
      expo--;
      i = i + i;
    }
    return expo;
  }
}

static double __cuda_logb(double a)
{
  unsigned long long int i;
  unsigned int ihi;
  unsigned int ilo;
  if (__cuda___isnan(a)) return a + a;
  a = __cuda_fabs(a);
  if (a == __longlong_as_double(0x7ff0000000000000ULL)) return a;
  if (a == 0.0) return -__longlong_as_double(0x7ff0000000000000ULL);
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  i = ((unsigned long long int)ihi) << 32 | (unsigned long long int)ilo;
  if (a >= 2.22507385850720140e-308) {
    return (double)((int)((ihi >> 20) & 0x7ff)) - 1023;
  } else {
    int expo = -1022;
    while (i < 0x0010000000000000ULL) {
      expo--;
      i = i + i;
    }
    return (double)expo;
  }
}

static double __cuda_fma(double a, double b, double c)
{
  return __fma_rn(a, b, c);
}
# 3735 "/usr/local/cuda/bin/../include/math_functions.h" 2 3
# 94 "/usr/local/cuda/bin/../include/common_functions.h" 2
# 227 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2
# 6 "/tmp/tmpxft_00006e97_00000000-1_main.cudafe1.stub.c" 2
struct __T20 {int *__par0;float *__par1;int *__par2;int __par3;int __par4;int __par5;int __dummy_field;};
static void __sti____cudaRegisterAll_39_tmpxft_00006e97_00000000_4_main_cpp1_ii_0e7396a7(void) __attribute__((__constructor__));
void __device_stub__Z14PetrinetKernelPiPfS_iii(int *__par0, float *__par1, int *__par2, int __par3, int __par4, int __par5){auto struct __T20 *__T23;
char __[256]; *(char**)&__T23 = __;if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T23->__par0 - (size_t)__T23) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T23->__par1 - (size_t)__T23) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T23->__par2 - (size_t)__T23) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T23->__par3 - (size_t)__T23) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par4, sizeof(__par4), (size_t)&__T23->__par4 - (size_t)__T23) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par5, sizeof(__par5), (size_t)&__T23->__par5 - (size_t)__T23) != cudaSuccess) return;{ volatile static char *__f; __f = ((char *)((void ( *)(int *, float *, int *, int, int, int))PetrinetKernel__entry)); (void)cudaLaunch(((char *)((void ( *)(int *, float *, int *, int, int, int))PetrinetKernel__entry))); };}

}
void PetrinetKernel__entry( int * __cuda_0,float * __cuda_1,int * __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
{__device_stub__Z14PetrinetKernelPiPfS_iii( __cuda_0, __cuda_1, __cuda_2, __cuda_3, __cuda_4, __cuda_5);}
extern "C"{




static void __sti____cudaRegisterAll_39_tmpxft_00006e97_00000000_4_main_cpp1_ii_0e7396a7(void){__cudaFatCubinHandle = __cudaRegisterFatBinary((void*)(&__fatDeviceText)); atexit(__cudaUnregisterBinaryUtil);__cudaRegisterFunction(__cudaFatCubinHandle, (const char*)((void ( *)(int *, float *, int *, int, int, int))PetrinetKernel__entry), (char*)"_Z14PetrinetKernelPiPfS_iii", "_Z14PetrinetKernelPiPfS_iii", (-1), (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);__cudaRegisterSharedVar(__cudaFatCubinHandle, (void**)"mt", 2496, 4, 0);}

}
# 256 "main.cu" 2