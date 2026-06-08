
/*!	\file debug.cpp
*
*	\brief Source file for common debug macros
*
*	\author Gregory Diamos
*
*	\date : Wednesday April 29, 2009
*
*/


#ifndef DEBUG_CPP_INCLUDED
#define DEBUG_CPP_INCLUDED

#include <hydrazine/debug.h>
#include <unordered_set>
#include <memory>

namespace hydrazine
{

	/*!
		\brief Global report timer
	*/
	Timer _ReportTimer;

	std::string _debugTime()
	{
		std::stringstream stream;
		stream.setf( std::ios::fixed, std::ios::floatfield );
		stream.precision( 6 );
		stream << _ReportTimer.seconds();
		return stream.str();
	}

	std::string _debugFile( const std::string& file, unsigned int line )
	{
		std::stringstream lineColon;
		lineColon << line << ":";

		std::stringstream stream;

		stream << stripReportPath<'/'>( file ) << ":";
		stream.width( 5 );
		stream.fill( ' ' );
		stream << std::left << lineColon.str();
		return stream.str();
	}

	/*! \brief Global logging infrastructure */
	class LogDatabase
	{
	public:
		LogDatabase();

	public:
		typedef std::unordered_set<std::string> StringSet;

	public:
		bool enableAll;
		StringSet enabledLogs;


	public:
		bool isEnabled(const std::string& logName) const
		{
			return enableAll || (enabledLogs.count(logName) != 0);
		}
	};

	LogDatabase::LogDatabase()
	: enableAll(false)
	{

	}

	static LogDatabase logDatabase;

	void enableAllLogs()
	{
		logDatabase.enableAll = true;
	}

	void enableLog(const std::string& name)
	{
		logDatabase.enabledLogs.insert(name);
	}

    #ifdef __APPLE__
    static NullStream* nullstream = new NullStream;
    #else
	static std::unique_ptr<NullStream> nullstream(new NullStream);
    #endif

	std::ostream& _getStream(const std::string& name)
	{
		if(logDatabase.isEnabled(name))
		{
			std::cout << "(" << _debugTime() << "): " << name << ": ";

			return std::cout;
		}

		return *nullstream;
	}

}
#endif
