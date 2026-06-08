/*! \file SystemCompatibility.h
	\date Monday August 2, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for hacked code required to assist windows
		compilaiton
*/

#ifndef SYSTEM_COMPATIBILITY_CPP_INCLUDED
#define SYSTEM_COMPATIBILITY_CPP_INCLUDED

// Hydrazine includes
#include <hydrazine/SystemCompatibility.h>

#if __APPLE__
	#include <sys/types.h>
	#include <sys/sysctl.h>
#elif __GNUC__
#if ENABLE_OPENGL
	#include <GL/glx.h>
#endif
	#include <unistd.h>
	#include <sys/sysinfo.h>
	#include <cxxabi.h>
#else
	#error "Unknown system/compiler (APPLE and GNUC are supported)."
#endif

namespace hydrazine
{
	unsigned int getHardwareThreadCount()
	{
	#if __APPLE__
		int nm[2];
	    size_t len = 4;
	    uint32_t count;

	    nm[0] = CTL_HW;
	    nm[1] = HW_AVAILCPU;
	    sysctl(nm, 2, &count, &len, NULL, 0);

	    if(count < 1)
	    {
	        nm[1] = HW_NCPU;
	        sysctl(nm, 2, &count, &len, NULL, 0);
	        if(count < 1)
	        {
		count = 1;
	        }
	    }
	    return count;
	#elif __GNUC__
		return sysconf(_SC_NPROCESSORS_ONLN);
	#endif
	}

	std::string getExecutablePath(const std::string& executableName)
	{
		return executableName;
	}

	long long unsigned int getFreePhysicalMemory()
	{
		#if __APPLE__
			int mib[2];
			uint64_t physical_memory;
			size_t length;
			// Get the Physical memory size
			mib[0] = CTL_HW;
			mib[1] = HW_USERMEM; // HW_MEMSIZE -> physical memory
			length = sizeof(uint64_t);
			sysctl(mib, 2, &physical_memory, &length, NULL, 0);
			return physical_memory;
		#elif __GNUC__
			return get_avphys_pages() * getpagesize();
		#endif
	}

	bool isAnOpenGLContextAvailable()
	{
		#if ENABLE_OPENGL
			#if __APPLE__
				// TODO fill this in
				return false;
			#elif __GNUC__
				GLXContext openglContext = glXGetCurrentContext();
				return (openglContext != 0);
			#endif
		#else
			return false;
		#endif
	}

	bool isMangledCXXString(const std::string& string)
	{
		return string.find("_Z") == 0;
	}

	std::string demangleCXXString(const std::string& string)
	{
		#if __APPLE__
			// TODO fill this in
			return string;
		#elif __GNUC__
			int status = 0;
			std::string name = abi::__cxa_demangle(string.c_str(),
				0, 0, &status);
			if(status < 0)
			{
				name = string;
			}

			return name;
		#endif
	}

}

#endif
