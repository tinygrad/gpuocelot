/*!
	\file CudaRuntime.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements the CUDA Runtime API for Ocelot
*/

// C standard library includes
#include <assert.h>

// C++ standard library includes
#include <sstream>
#include <algorithm>

// Ocelot includes
#include <ocelot/cuda/include/__cudaFatFormat.h>
#include <ocelot/cuda/interface/CudaRuntime.h>
#include <ocelot/ir/interface/PTXInstruction.h>

// Hydrazine includes
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

//////////////////////////////////////////////////////////////////////////////////////////////////

const char *cuda::FatBinaryContext::name() const {
	__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)cubin_ptr;
	return binary->ident;
}	

const char *cuda::FatBinaryContext::ptx() const {
	__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)cubin_ptr;
	return binary->ptx->ptx;
}

static executive::ChannelFormatDesc convert(const cudaChannelFormatDesc &cudaDesc) {
	executive::ChannelFormatDesc desc;
	desc.w = cudaDesc.w;
	desc.x = cudaDesc.x;
	desc.y = cudaDesc.y;
	desc.z = cudaDesc.z;
	desc.kind = (executive::ChannelFormatDesc::Kind)(int)cudaDesc.f;
	return desc;
}

static executive::MemcpyKind convert(const cudaMemcpyKind kind) {
	switch (kind) {
		case cudaMemcpyHostToHost:
			return executive::HostToHost;
		case cudaMemcpyDeviceToHost:
			return executive::DeviceToHost;
		case cudaMemcpyHostToDevice:
			return executive::HostToDevice;
		case cudaMemcpyDeviceToDevice:
			return executive::DeviceToDevice;
	}
	return executive::MemcpyKind_invalid;
}

static executive::dim3 convert(dim3 dim) {
	executive::dim3 ed3 = {dim.x, dim.y, dim.z};
	return ed3;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

cuda::HostThreadContext::HostThreadContext(): selectedDevice(0), nextStream(0), nextEvent(0), 
	parameterBlock(0), parameterBlockSize(0) {

}

cuda::CudaContext::CudaContext(): thread(0), context(0) { }
	
//////////////////////////////////////////////////////////////////////////////////////////////////

cuda::CudaRuntime::CudaRuntime() {

}

cuda::CudaRuntime::~CudaRuntime() {
	//
	// free things that need freeing
	//
	
	// thread contexts
	for (HostThreadContextMap::iterator it = threads.begin(); it != threads.end(); ++it) {
		free(it->second.parameterBlock);
	}
	threads.clear();
	
	// textures
	
	// kernels
	
	// fat binaries
	
	// globals
}

//! acquires mutex and locks the runtime
void cuda::CudaRuntime::lock() {
	
}

//! releases mutex
void cuda::CudaRuntime::unlock() {

}


//! sets the last error state for the CudaRuntime object
cudaError_t cuda::CudaRuntime::setLastError(cudaError_t result) {
	HostThreadContext &thread = getHostThreadContext();
	thread.lastError = result;
	return result;
}

//! gets the context of a thread
cuda::HostThreadContext & cuda::CudaRuntime::getHostThreadContext() {
	pthread_t self = pthread_self();
	
	HostThreadContextMap::iterator it = threads.find(self);
	if (it == threads.end()) {
		HostThreadContext thread;
		thread.parameterBlockSize = (1<<13);
		thread.parameterBlock = (unsigned char *)malloc(thread.parameterBlockSize);
		thread.selectedDevice = 0;
		threads[self] = thread;
		it = threads.find(self);
	}
	
	context.selectDevice(it->second.selectedDevice);
	
	return it->second;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/*!
	registers a CUDA fatbinary and returns a handle for referencing the fat binary
*/
void** cuda::CudaRuntime::cudaRegisterFatBinary(void *fatCubin) {
	size_t handle = 0;
	__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;
	
	lock();
	
	//! inserts thread context if not already exists
	getHostThreadContext();
	
	for (FatBinaryVector::const_iterator it = fatBinaries.begin();
		it != fatBinaries.end(); ++it) {
		if (std::string(it->name()) == binary->ident) {
			unlock();
		
			assert(0 && "binary already exists");		
			return 0;
		}	
	}

	// register associated PTX
	if( binary->ptx->ptx != 0 ) {
		std::stringstream ptx( binary->ptx->ptx );
		context.loadModule( binary->ident, false, ptx );
	}
	else {
		assert(0 && "binary contains no PTX");
	}
	
	FatBinaryContext cubinContext;
	cubinContext.cubin_ptr = fatCubin;
	fatBinaries.push_back(cubinContext);
	handle = fatBinaries.size();
	
	unlock();
	
	return (void **)handle;
}

/*!
	unregister a cuda fat binary
*/
void cuda::CudaRuntime::cudaUnregisterFatBinary(void **fatCubinHandle) {
	// do we really care?
}

/*!
	\brief register a CUDA global variable
	\param fatCubinHandle
	\param hostVar
	\param deviceAddress
	\param deviceName
	\param ext
	\param size
	\param constant
	\param global
*/
void cuda::CudaRuntime::cudaRegisterVar(void **fatCubinHandle, char *hostVar, 
	char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global) {

	size_t handle = (size_t)fatCubinHandle - 1;
	lock();
	
	const char *moduleName = fatBinaries[handle].name();
	void *hostVariable = (void *)hostVar;
	
	executive::DeviceAddressSpace addrSpace = (constant ? executive::Device_constant : 
		executive::Device_global);
	context.registerGlobalVariable(moduleName, deviceName, hostVariable, (void *)deviceAddress, 
		size, addrSpace);
	
	unlock();
}

/*!
	\brief registers a CUDA texture reference
	\param fatCubinHandle
	\param hostVar
	\param deviceAddress
	\param deviceName
	\param dim
	\param norm
	\param ext
*/
void cuda::CudaRuntime::cudaRegisterTexture(
	void **fatCubinHandle,
	const struct textureReference *hostVar,
	const void **deviceAddress,
	const char *deviceName,
	int dim,
	int norm,
	int ext) {
	
	size_t handle = (size_t)fatCubinHandle - 1;
	
	lock();
	
	getHostThreadContext();
	
	// register the texture object to the runtime and declare it to the executive
	
	RegisteredTexture texture;
	
	texture.handle = handle;
	texture.pointer = (void *)hostVar;
	texture.module = fatBinaries[handle].name();
	texture.texture = deviceName;
	
	// now tell the executive about it
	context.registerTexture(texture.module, texture.texture, dim, norm);
	
	textures[std::string(texture.texture)] = texture;
	
	unlock();
}

void cuda::CudaRuntime::cudaRegisterShared(
	void **fatCubinHandle,
	void **devicePtr) {
	
	size_t handle = (size_t)fatCubinHandle - 1;
	lock();
	
	const char *moduleName = fatBinaries[handle].name();
	const char *variableName = (const char *)devicePtr;
	
	report("cudaRegisterShared() - module " << moduleName << ", devicePtr: " << std::hex 
		<< devicePtr << std::dec << " named " << variableName);
	
	context.registerGlobalVariable(moduleName, variableName, 0, 0, sizeof(void*), 
		executive::Device_shared);
	
	unlock();
}

void cuda::CudaRuntime::cudaRegisterSharedVar(
	void **fatCubinHandle,
	void **devicePtr,
	size_t size,
	size_t alignment,
	int storage) {

	assert(0 && "unimplemented");
}

void cuda::CudaRuntime::cudaRegisterFunction(
	void **fatCubinHandle,
	const char *hostFun,
	char *deviceFun,
	const char *deviceName,
	int thread_limit,
	uint3 *tid,
	uint3 *bid,
	dim3 *bDim,
	dim3 *gDim,
	int *wSize) {

	size_t handle = (size_t)fatCubinHandle - 1;
	
	lock();

	getHostThreadContext();
		
	void *funcPtr = (void *)hostFun;
	const char *funcName = deviceFun;
	const char *moduleName = fatBinaries[handle].name();
	
	// associate the function pointer with the kernel name and maybe ir::Kernel
	RegisteredKernel kernel;
	kernel.handle = handle;
	kernel.module = moduleName;
	kernel.kernel = funcName;
	kernel.pointer = funcPtr;
	
	kernels[funcPtr] = kernel;
	
	unlock();
}
		
//////////////////////////////////////////////////////////////////////////////////////////////////
//
// memory allocation

cudaError_t cuda::CudaRuntime::cudaMalloc(void **devPtr, size_t size) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
		
	if (context.malloc(devPtr, size)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMallocHost(void **ptr, size_t size) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
	
	if (context.mallocHost(ptr, size)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, 
	size_t height) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
	if (context.mallocPitch(devPtr, pitch, width, height)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMallocArray(struct cudaArray **array, 
	const struct cudaChannelFormatDesc *desc, size_t width, size_t height) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
	if (context.mallocArray(array, convert(*desc), width, height)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFree(void *devPtr) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	report("cudaFree()");
	
	getHostThreadContext();
	if (context.free(devPtr)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFreeHost(void *ptr) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
	if (context.freeHost(ptr)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFreeArray(struct cudaArray *array) {
	cudaError_t result = cudaErrorMemoryAllocation;
	lock();
	getHostThreadContext();
	if (context.freeArray(array)) {
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//
// memory copying

cudaError_t cuda::CudaRuntime::cudaMemcpy(void *dst, const void *src, size_t count, 
	enum cudaMemcpyKind kind) {
	cudaError_t result = cudaSuccess;
	if (kind >= 0 && kind <= 3) {
		lock();
		getHostThreadContext();
		bool success = context.deviceMemcpy(dst, src, count, convert(kind));
		if (!success) {
			result = cudaErrorInvalidDevicePointer;
		}
		unlock();
	}
	else {
		result = cudaErrorInvalidMemcpyDirection;
	}
	return setLastError(result);	
}


cudaError_t cuda::CudaRuntime::cudaMemcpyToSymbol(const char *symbol, const void *src,
	size_t count, size_t offset, enum cudaMemcpyKind kind) {

	cudaError_t result = cudaSuccess;
	lock();
	getHostThreadContext();
	bool success = context.deviceMemcpyToSymbol(symbol, src, count, offset, convert(kind));	
	if (!success) {
		result = cudaErrorInvalidDevicePointer;
	}
	unlock();
	return setLastError(result);	
}

cudaError_t cuda::CudaRuntime::cudaMemcpyAsync(void *dst, const void *src, size_t count, 
	enum cudaMemcpyKind kind, cudaStream_t streamHandle) {
	
	cudaError_t result = cudaErrorNotYetImplemented;
	lock();
	HostThreadContext & thread = getHostThreadContext();
	
	StreamMap::iterator s_it = thread.streams.find(streamHandle);
	if (s_it != thread.streams.end()) {
		if (!context.getDeviceAddressSpace()) {
			//
			// host address space transfers aren't asynchronous
			//
			
			CudaRuntime::cudaMemcpy(dst, src, count, kind);
		}
		else {
			assert(0 && "unimplemented");
		}
	}
	else {
		result = cudaErrorInvalidValue;
	}
	
	unlock();
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// memset
//

cudaError_t cuda::CudaRuntime::cudaMemset(void *devPtr, int value, size_t count) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	getHostThreadContext();
	if (!context.getDeviceAddressSpace()) {
		::memset((void *)devPtr, value, count);
	}
	else {
		assert(0 && "unimplemented");
	}
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, 
	size_t height) {

	cudaError_t result = cudaErrorNotYetImplemented;
	
	assert(0 && "unimplemented");
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// memory allocation
//

cudaError_t cuda::CudaRuntime::cudaGetSymbolAddress(void **devPtr, const char *symbol) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetSymbolSize(size_t *size, const char *symbol) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGetDeviceCount(int *count) {
	cudaError_t result = cudaSuccess;
	lock();
	
	*count = context.getDevices().size();
	
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetDeviceProperties(struct cudaDeviceProp *prop, int dev) {
	cudaError_t result = cudaSuccess;
	
	lock();
	if (dev < (int)context.getDevices().size()) {
		const executive::Device & device = context.getDevices().at(dev);
	
		prop->canMapHostMemory = 1;
		prop->clockRate = device.clockRate;
		prop->computeMode = cudaComputeModeDefault;
		prop->deviceOverlap = device.memcpyOverlap;
		prop->integrated = 0;
		prop->kernelExecTimeoutEnabled = 0;
		prop->major = device.major;
		prop->minor = device.minor;
		prop->maxGridSize[0] = device.maxGridSize[0];
		prop->maxGridSize[1] = device.maxGridSize[1];
		prop->maxGridSize[2] = device.maxGridSize[2];
		prop->maxThreadsDim[0] = device.maxThreadsDim[0];
		prop->maxThreadsDim[1] = device.maxThreadsDim[1];
		prop->maxThreadsDim[2] = device.maxThreadsDim[2];
		prop->maxThreadsPerBlock = device.maxThreadsPerBlock;
		prop->memPitch = device.memPitch;
		prop->multiProcessorCount = device.multiprocessorCount;
		prop->regsPerBlock = device.regsPerBlock;
		prop->sharedMemPerBlock = device.sharedMemPerBlock;
		prop->textureAlignment = device.textureAlign;
		prop->totalConstMem = device.totalConstantMemory;
		prop->totalGlobalMem = device.totalMemory;
		prop->warpSize = device.SIMDWidth;
		
		memcpy(prop->name, device.name.c_str(), device.name.size());
		
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidDevice;
	}
	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetDevice(int device) {
	cudaError_t result = cudaErrorSetOnActiveProcess;
	
	lock();
	if ((int)context.getDevices().size() <= device) {
		result = cudaErrorInvalidDevice;
	}
	else {
		HostThreadContext &thread = getHostThreadContext();
		thread.selectedDevice = device;
		context.selectDevice(device);
		result = cudaSuccess;
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetDevice(int *device) {
	cudaError_t result = cudaSuccess;
	
	lock();
	HostThreadContext &thread = getHostThreadContext();
	*device = thread.selectedDevice;
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetValidDevices(int *device_arr, int len) {
	cudaError_t result = cudaSuccess;
	lock();
	HostThreadContext & thread = getHostThreadContext();
	thread.validDevices.resize(len);
	for (int i = 0 ; i < len; i++) {
		thread.validDevices[i] = device_arr[i];
	}
	unlock();
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetDeviceFlags( int flags ) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

//! binds a texture to a reference and a CUDA memory block
cudaError_t cuda::CudaRuntime::cudaBindTexture(size_t *offset, 
	const struct textureReference *texref, const void *devPtr, 
	const struct cudaChannelFormatDesc *desc, size_t size) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaBindTexture2D(size_t *offset,
	const struct textureReference *texref,const void *devPtr, 
	const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaBindTextureToArray(const struct textureReference *texref, 
	const struct cudaArray *array, const struct cudaChannelFormatDesc *desc) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaUnbindTexture(const struct textureReference *texref) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetTextureAlignmentOffset(size_t *offset, 
	const struct textureReference *texref) {
	cudaError_t result = cudaSuccess;
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetTextureReference(const struct textureReference **texref, 
	const char *symbol) {
	cudaError_t result = cudaErrorInvalidTexture;
	lock();
	
	RegisteredTextureMap::iterator it = textures.find(symbol);
	if (it != textures.end()) {
		*texref = (const struct textureReference *)it->second.pointer;
		result = cudaSuccess;
	}
	
	unlock();
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGetLastError(void) {
	HostThreadContext &thread = getHostThreadContext();
	return thread.lastError;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
	
cudaError_t cuda::CudaRuntime::cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
	cudaStream_t stream) {

	lock();
	report("cudaConfigureCall()");
	
	cudaError_t result = cudaErrorInvalidConfiguration;
	HostThreadContext &thread = getHostThreadContext();
	
	KernelLaunchConfiguration launch(gridDim, blockDim, sharedMem, stream);
	thread.launchConfigurations.push_back(launch);
	result = cudaSuccess;
	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	cudaError_t result = cudaSuccess;
	
	lock();
		
	HostThreadContext &thread = getHostThreadContext();
	
	memcpy(thread.parameterBlock + offset, arg, size);
	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaLaunch(const char *entry) {

	lock();
	
	cudaError_t result = cudaSuccess;
	HostThreadContext &thread = getHostThreadContext();
	
	assert(thread.launchConfigurations.size());
		
	KernelLaunchConfiguration launch(thread.launchConfigurations.back());
	thread.launchConfigurations.pop_back();
	
	// launch via the executive
	//
	// executive needs: kernel grid and block dimensions, shared memory size, parameter data
	//
	
	// void Executive::launch(const std::string & module, const std::string & kernel, dim3 grid, dim3 block,
	// 	size_t sharedMemory, unsigned char *parameterBlock, size_t parameterBlockSize);

	const RegisteredKernel & kernel = kernels[(void *)entry];
	std::string moduleName = kernel.module;
	std::string kernelName = kernel.kernel;
	
	context.launch(moduleName, kernelName, convert(launch.gridDim), convert(launch.blockDim), 
		launch.sharedMemory, thread.parameterBlock, thread.parameterBlockSize);
	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFuncGetAttributes(struct cudaFuncAttributes *attr, 
	const char *func) {
	cudaError_t result = cudaSuccess;
		
	lock();
	
	//
	// go find the kernel and fill out its attributes
	//
	
	attr->constSizeBytes = 0;
	attr->localSizeBytes = 0;
	attr->maxThreadsPerBlock = 0;
	attr->numRegs = 0;
	attr->sharedSizeBytes = 0;
	
	unlock();
	
	assert(0 && "unimplemented");
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// CUDA event creation

cudaError_t cuda::CudaRuntime::cudaEventCreate(cudaEvent_t *event) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	HostThreadContext &thread = getHostThreadContext();
	
	//
	// create an event
	//
	Event createdEvent;
	createdEvent.flags = 0;
	createdEvent.handle = thread.events.size();
	*event = createdEvent.handle;
	thread.events[*event] = createdEvent;
	result = cudaSuccess;
	
	// special behavior for CUDA devices
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
	
	unlock();

	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	HostThreadContext &thread = getHostThreadContext();
	
	//
	// create an event
	//
	Event createdEvent;
	createdEvent.flags = flags;
	createdEvent.handle = thread.events.size();
	*event = createdEvent.handle;
	thread.events[*event] = createdEvent;
	result = cudaSuccess;

	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
		
	unlock();

	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	StreamMap::iterator s_it = thread.streams.find(stream);
	EventMap::iterator e_it = thread.events.find(event);
	if (e_it != thread.events.end() && s_it != thread.streams.end()) {
		e_it->second.timer.start();
		s_it->second.events.push_back(event);
		
		result = cudaSuccess;
	}
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
		
	unlock();	
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventQuery(cudaEvent_t event) {
	cudaError_t result = cudaErrorInvalidValue;

	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	EventMap::iterator e_it = thread.events.find(event);
	if (e_it != thread.events.end()) {
		result = cudaSuccess;
	}
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	unlock();

	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventSynchronize(cudaEvent_t event) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	EventMap::iterator e_it = thread.events.find(event);
	if (e_it != thread.events.end()) {
		result = cudaSuccess;
	}
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventDestroy(cudaEvent_t event) {
	cudaError_t result = cudaErrorInvalidValue;
	
	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	EventMap::iterator e_it = thread.events.find(event);
	if (e_it != thread.events.end()) {
		thread.events.erase(e_it);
		result = cudaSuccess;
	}
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	EventMap::iterator e_start_it = thread.events.find(start);
	EventMap::iterator e_end_it = thread.events.find(end);
	
	if (e_start_it != thread.events.end() && e_end_it != thread.events.end()) {
		e_end_it->second.timer.stop();
		e_start_it->second.timer.stop();
		
		long long end_cycles = (long long)e_end_it->second.timer.cycles();
		long long start_cycles = (long long)e_start_it->second.timer.cycles();
		long long total_cycles = start_cycles - end_cycles;

		e_end_it->second.timer.start();
		e_start_it->second.timer.start();
		
		*ms = (float)((total_cycles + 0.0 ) * 1.0e-9);
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidResourceHandle;
	}
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
	unlock();
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// CUDA streams
//

cudaError_t cuda::CudaRuntime::cudaStreamCreate(cudaStream_t *pStream) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	
	HostThreadContext &thread = getHostThreadContext();
	
	Stream stream;
	stream.handle = (cudaStream_t)thread.streams.size();
	thread.streams[stream.handle] = stream;
	*pStream = stream.handle;
	result = cudaSuccess;
	
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
		
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamDestroy(cudaStream_t stream) {
	cudaError_t result = cudaErrorNotYetImplemented;
	
	lock();
	HostThreadContext &thread = getHostThreadContext();
	StreamMap::iterator s_it = thread.streams.find(stream);

	if (s_it != thread.streams.end()) {
		thread.streams.erase(s_it);
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidResourceHandle;
	}	
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamSynchronize(cudaStream_t stream) {
	cudaError_t result = cudaErrorNotYetImplemented;

	lock();
	HostThreadContext &thread = getHostThreadContext();
	StreamMap::iterator s_it = thread.streams.find(stream);

	if (s_it != thread.streams.end()) {
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidResourceHandle;
	}	
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	
	unlock();

	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamQuery(cudaStream_t stream) {
	cudaError_t result = cudaErrorNotYetImplemented;

	lock();
	HostThreadContext &thread = getHostThreadContext();
	StreamMap::iterator s_it = thread.streams.find(stream);

	if (s_it != thread.streams.end()) {
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidResourceHandle;
	}	
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}	
	
	unlock();
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaThreadExit(void) {
	cudaError_t result = cudaSuccess;
	
	lock();
	getHostThreadContext();
	
	//
	// kill potentially running kernels
	//
	if (context.getSelectedISA() == ir::PTXInstruction::GPU) {
		assert(0 && "unimplemented");
	}
	unlock();
	
	return setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaThreadSynchronize(void) {
	cudaError_t result = cudaSuccess;
	lock();
	getHostThreadContext();
	unlock();
	
	//
	// block on executing threads
	//
	context.synchronize();
	
	return setLastError(result);
}

//////////////////////////////////////////////////////////////////////////////////////////////////