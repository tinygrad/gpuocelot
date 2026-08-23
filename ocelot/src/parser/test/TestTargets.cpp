/*! \file TestTargets.cpp
	\brief regression tests for .target parsing and bfi operand typing
*/

#include <sstream>

#include <hydrazine/Test.h>
#include <hydrazine/ArgumentParser.h>
#include <hydrazine/Exception.h>

#include <ocelot/ir/Module.h>

namespace test
{

class TestTargets: public Test
{
public:
	TestTargets()
	{
		name = "TestTargets";
		description = "Parses modern .target shader models, which used to be a";
		description += " lexical error, and bfi with immediate pos/len.";
	}

private:
	/*! \brief parse a module, reporting the parser's own message on failure */
	bool parses(const std::string& ptx, const std::string& what)
	{
		std::stringstream stream(ptx);
		ir::Module module;
		try
		{
			if(!module.load(stream))
			{
				status << what << ": load returned false\n";
				return false;
			}
		}
		catch(const std::exception& e)
		{
			status << what << ": " << e.what() << "\n";
			return false;
		}
		return true;
	}

	std::string kernel(const std::string& target, const std::string& body)
	{
		return ".version 8.0\n.target " + target + "\n.address_size 64\n"
			".visible .entry k()\n{\n" + body + "\tret;\n}\n";
	}

	/*! \brief the lexer used to stop at sm_35, so CUDA 12, which dropped
		sm_35, had no target it could emit */
	bool testShaderModels()
	{
		const char* models[] = { "sm_10", "sm_20", "sm_35", "sm_50", "sm_61",
			"sm_70", "sm_80", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_120" };

		for(auto model : models)
		{
			if(!parses(kernel(model, ""), model)) return false;
		}
		return true;
	}

	/*! \brief pos and len are u32 whatever the instruction type is, but the
		parser types immediates from the instruction, so .b64 made them b64 */
	bool testBfiImmediates()
	{
		const std::string b32 = "\t.reg .b32 %r<4>;\n"
			"\tbfi.b32 %r1, %r2, %r3, 8, 16;\n";
		const std::string b64 = "\t.reg .b64 %rd<4>;\n"
			"\tbfi.b64 %rd1, %rd2, %rd3, 32, 32;\n";

		return parses(kernel("sm_50", b32), "bfi.b32")
			&& parses(kernel("sm_50", b64), "bfi.b64");
	}

public:
	bool doTest()
	{
		return testShaderModels() && testBfiImmediates();
	}
};

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	test::TestTargets test;
	parser.description(test.testDescription());

	parser.parse("-v", test.verbose, false, "Print out info after the test.");
	parser.parse();

	test.test();

	return test.passed();
}
