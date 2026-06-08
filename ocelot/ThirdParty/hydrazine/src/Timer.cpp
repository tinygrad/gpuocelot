/*!	\file Timer.cpp
*
*	\brief Source file for the Timer class
*
*	Author: Gregory Diamos
*
*
*/

#ifndef TIMER_CPP_INCLUDED
#define TIMER_CPP_INCLUDED

#include <hydrazine/Timer.h>
#include <sstream>

namespace hydrazine
{
	std::string Timer::toString() const
	{
		std::stringstream stream;

		stream << seconds() << "s (" << cycles() << " ns)";

		return stream.str();
	}
}

#endif
