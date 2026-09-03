/*! \file TestInstructions.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief unit tests for each instruction
*/

#include <sstream>
#include <fstream>

#include <hydrazine/Test.h>

#include <hydrazine/ArgumentParser.h>
#include <hydrazine/Exception.h>
#include <hydrazine/macros.h>
#include <hydrazine/debug.h>

#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>

#include <cmath>
#include <limits>

using namespace std;
using namespace ir;
using namespace executive;

namespace test {

class TestInstructions: public Test {
public:
	int threadCount;
	bool valid;

	EmulatedKernel *kernel = nullptr;
	CooperativeThreadArray* cta = nullptr;
	Module module;
	
	TestInstructions() {
		valid = true;
		name = "TestInstructions";
		kernel = 0;

		status << "Test output:\n";

		threadCount = 32;

		const std::string ptx = "TestInstructions_ptx";

		bool loaded = false;
		
		try {
			loaded = module.loadEmbedded(ptx);
		}
		catch(const hydrazine::Exception& e) {
			status << " error - " << e.what() << "\n";
		}

		if(!loaded) {
			status << "failed to load module '" << ptx << "'\n";
			valid = false;
			return;
		}
		
		IRKernel* rawKernel = module.getKernel("_Z17k_simple_sequencePi");
		if (rawKernel == 0) {
			status << "failed to get kernel\n";
			valid = false;
			return;
		}
		else {
			kernel = new EmulatedKernel(rawKernel, 0);
			kernel->setKernelShape(threadCount, 1, 1);
			kernel->setExternSharedMemorySize(64);
			cta = new CooperativeThreadArray(kernel, ir::Dim3(), false);
		}
	}

	~TestInstructions() {
		delete kernel;
		delete cta;
	}

	/*!
		Constructs a register operand with a given name, type, and register index
	*/
	PTXOperand reg(std::string name, PTXOperand::DataType type, 
		PTXOperand::RegisterType reg) {
		PTXOperand op;
		op.addressMode = PTXOperand::Register;
		op.identifier = name;
		op.type = type;
		op.reg = reg;
		return op;
	}

	PTXOperand sreg(PTXOperand::SpecialRegister reg, PTXOperand::VectorIndex v) {
		PTXOperand op;
		op.addressMode = PTXOperand::Special;
		op.type = PTXOperand::u16;
		op.special = reg;
		op.vIndex = v;
		return op;
	}

	PTXOperand imm_uint(std::string name, PTXOperand::DataType type, PTXU64 imm) {
		PTXOperand op;
		op.addressMode = PTXOperand::Immediate;
		op.identifier = name;
		op.type = type;
		op.imm_uint = imm;
		return op;
	}

	PTXOperand imm_int(std::string name, PTXOperand::DataType type, PTXS64 imm) {
		PTXOperand op;
		op.addressMode = PTXOperand::Immediate;
		op.identifier = name;
		op.type = type;
		op.imm_int = imm;
		return op;
	}

	PTXOperand imm_float(std::string name, PTXOperand::DataType type, PTXF64 imm) {
		PTXOperand op;
		op.addressMode = PTXOperand::Immediate;
		op.identifier = name;
		op.type = type;
		op.imm_float = imm;
		return op;
	}

	/*!
		Tests register getters and setters:
	*/
	bool testRegisterAccessors() {

		using namespace std;
		using namespace ir;
		using namespace executive;

		bool result = true;
		cta->reset();

		for (int j = 0; j < (int)kernel->registerCount(); j++) {
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, j, i*(j+1));
			}
		}
		for (int j = 0; result && j < (int)kernel->registerCount(); j++) {
			for (int i = 0; result && i < threadCount; i++) {
				if (cta->getRegAsU32(i, j) != (ir::PTXU32)(i*(j+1))) {
					result = false;
					status << "set/getRegAsU32 test failed\n";
				}
			}
		}

		for (int i = 0; result && i < threadCount; i ++) {
			// set as float, get as float
			if (result) {
				cta->setRegAsF32(i, 1, 12.5f);
				if (std::fabs(cta->getRegAsF32(i, 1) - 12.5f) > 0.01f) {
					result = false;
					status << "set/getRegAsF32 failed\n";
				}
			}

			// handling types of mixed sizes
			if (result) {
				PTXU64 value = 0xffffffffffffffffULL;
				PTXU64 outval = 0xffffffffffff0000ULL;
				PTXU64 encountered;
				cta->setRegAsU64(i, 3, value);
				cta->setRegAsU16(i, 3, 0);
				encountered = cta->getRegAsU64(i, 3);
				if (encountered != outval) {
					result = false;
					status << "setAsU64, setAsU16 failed: getAsU64 returned 0x" << std::hex;
					status << encountered << ", expected: 0x" << outval << std::dec << " for thread " << i << "\n";
					status << "read the register one more time: 0x" << std::hex << cta->getRegAsU64(i, 3) << dec << "\n";
				}
			}

			// set as float, get as uint
			if (result) {
				cta->setRegAsF32(i, 2, 1.0f);
				if (cta->getRegAsU32(i, 2) != 0x3F800000) {
					result = false;
					status << "setRegAsF32, getRegAsU32 failed: 0x" 
						<< std::hex <<  cta->getRegAsU32(i, 2) << std::dec <<  "\n";
				}
			}
		}
		
		status << "Accessors test passed.\n";
		
		return result;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//
	// Arithmetic instructions
	//
	/////////////////////////////////////////////////////////////////////////////////////////////////


	/*!
		Tests several forms of the abs instruction
	*/
	bool test_Abs() {
		bool result = true;

		PTXInstruction ins;

		cta->reset();

		// s16
		//
		if (result) {
			ins.opcode = PTXInstruction::Abs;
			ins.type = PTXOperand::s16;
			ins.d = reg("r2", PTXOperand::s16, 0);
			ins.a = reg("r1", PTXOperand::s16, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsS16(t, 1, -t);
			}
			cta->eval_Abs(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				if (cta->getRegAsS16(t, 0) != t) {
					result = false;
					status << "abs.s16 failed (thread " << t << "): expected " << t << ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// s32
		//
		if (result) {
			ins.opcode = PTXInstruction::Abs;
			ins.type = PTXOperand::s32;
			ins.d = reg("r2", PTXOperand::s32, 0);
			ins.a = reg("r1", PTXOperand::s32, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsS32(t, 1, -t);
			}
			cta->eval_Abs(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				if (cta->getRegAsS32(t, 0) != t) {
					result = false;
					status << "abs.s32 failed: expected " << t << ", got " << cta->getRegAsS32(t, 0) << "\n";
				}
			}
		}

		// s64
		//
		if (result) {
			ins.opcode = PTXInstruction::Abs;
			ins.type = PTXOperand::s64;
			ins.d = reg("r2", PTXOperand::s64, 0);
			ins.a = reg("r1", PTXOperand::s64, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsS64(t, 1, -t);
			}
			cta->eval_Abs(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				if (cta->getRegAsS64(t, 0) != t) {
					result = false;
					status << "abs.s64 failed: expected " << t << ", got " << cta->getRegAsS64(t, 0) << "\n";
				}
			}
		}

		// f32
		//
		if (result) {
			ins.opcode = PTXInstruction::Abs;
			ins.type = PTXOperand::f32;
			ins.d = reg("r2", PTXOperand::f32, 0);
			ins.a = reg("r1", PTXOperand::f32, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsF32(t, 1, -(float)t * 2.76f);
				cta->setRegAsF32(t, 0, 0);
			}
			cta->eval_Abs(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				if (std::fabs(cta->getRegAsF32(t, 0) - (float)t * 2.76f) > 0.01f) {
					result = false;
					status << "abs.f32 failed: expected " << (float)t * 2.76f << ", got " << cta->getRegAsF32(t, 0) << "\n";
				}
			}
		}

		// f64
		//
		if (result) {
			ins.opcode = PTXInstruction::Abs;
			ins.type = PTXOperand::f64;
			ins.d = reg("r2", PTXOperand::f64, 0);
			ins.a = reg("r1", PTXOperand::f64, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsF64(t, 1, -(double)t * 9.76);
				cta->setRegAsF64(t, 0, 0);
			}
			cta->eval_Abs(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				if (std::fabs(cta->getRegAsF64(t, 0) - (double)t * 9.76) > 0.01f) {
					result = false;
					status << "abs.f64 failed: expected " << t << ", got " << cta->getRegAsF64(t, 0) << "\n";
				}
			}
		}

		status << "Abs test passed.\n";

		return result;
	}

	bool test_Add() {
		bool result = true;

		PTXInstruction ins;

		// f16
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.b = reg("r2", PTXOperand::b16, 1);
			ins.d = reg("r3", PTXOperand::b16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, 0x3e00); // 1.5
				cta->setRegAsU16(i, 1, 0x4000); // 2.0
				cta->setRegAsU16(i, 2, 0);
			}
			if (!ins.valid().empty()) {
				result = false;
				status << "add.f16 rejected\n";
			}
			else {
				cta->eval_Add(cta->getActiveContext(), ins);
				for (int i = 0; i < threadCount; i++) {
					if (cta->getRegAsU16(i, 2) != 0x4300) { // 3.5
						result = false;
						status << "add.f16 incorrect\n";
						break;
					}
				}
			}
		}

		// u16
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 2));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 2) != (i*2+4+i)) {
					result = false;
					status << "add.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 2) != (PTXU32)(i*2+4+i)) {
					result = false;
					status << "add.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 2));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU64(i, 2) != (PTXU64)(i*2+4+i)) {
					result = false;
					status << "add.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS16(i, 2) != (i*2+4+i)) {
					result = false;
					status << "add.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS32(i, 2) != (i*2+4+i)) {
					result = false;
					status << "add.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS64(i, 2) != (i*2+4+i)) {
					result = false;
					status << "add.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 2));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + i));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsF32(i, 2) != (PTXF32)(i*2+4+i)) {
					result = false;
					status << "add.f32 incorrect [" << i << "] - expected: " << (float)(i*2+4+i) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.opcode = PTXInstruction::Add;
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 2));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Add(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF64(i, 2) - (double)(i*2+4+i)) > 0.1) {
					result = false;
					status << "add.f64 incorrect [" << i << "] - expected: " << (PTXF64)(i*2+4+i) 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_AddC() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::AddC;

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = imm_uint("r2", PTXOperand::u32, 0x0fffffffe);
			ins.c = reg("r3", PTXOperand::u32, 6);
			ins.d = reg("r6", PTXOperand::u32, 5);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 5, 0);
				cta->setRegAsU32(i, 6, 1);	// set the carry flag
			}
			cta->eval_AddC(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = (0x0fffffffe + (PTXU32)(i*2) + 1);
				if (cta->getRegAsU32(i, 5) != expected) {
					result = false;
					status << "addc.u32 incorrect\n";
					break;
				}
				// verify carry		
				if (cta->getRegAsU32(i, 6) != 1) {
					result = false;
					status << "addc.u32 failed to set carry bit\n";
					break;
				}		
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = imm_uint("r2", PTXOperand::s32, 0x0fffffffe);
			ins.c = reg("r3", PTXOperand::s32, 6);
			ins.d = reg("r6", PTXOperand::s32, 5);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 5, 0);
				cta->setRegAsU32(i, 6, 1);	// set the carry flag
			}
			cta->eval_AddC(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = (0x0fffffffe + (PTXS32)(i*2) + 1);
				if (cta->getRegAsS32(i, 5) != expected) {
					result = false;
					status << "addc.s32 incorrect\n";
					break;
				}
				// verify carry			
				if (cta->getRegAsS32(i, 6) != 1) {
					result = false;
					status << "addc.s32 failed to set carry bit\n";
					break;
				}				
			}
		}

		return result;
	}

	bool test_Sub() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sub;

		// f16
		//
		if (result) {
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.b = reg("r2", PTXOperand::b16, 1);
			ins.d = reg("r3", PTXOperand::b16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, 0x4000); // 2.0
				cta->setRegAsU16(i, 1, 0x3e00); // 1.5
				cta->setRegAsU16(i, 2, 0);
			}
			if (!ins.valid().empty()) {
				result = false;
				status << "sub.f16 rejected\n";
			}
			else {
				cta->eval_Sub(cta->getActiveContext(), ins);
				for (int i = 0; i < threadCount; i++) {
					if (cta->getRegAsU16(i, 2) != 0x3800) { // 0.5
						result = false;
						status << "sub.f16 incorrect\n";
						break;
					}
				}
			}
		}

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(9 + i * 2));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 2) != (9 + i*2-(4+i))) {
					result = false;
					status << "sub.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(9 + i * 2));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 2) != (PTXU32)(9 + i*2 - (4+i))) {
					result = false;
					status << "sub.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(9 + i * 2));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU64(i, 2) != (PTXU64)(9 + i*2 - (4+i))) {
					result = false;
					status << "sub.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS16(i, 2) != (i*2-(4+i))) {
					result = false;
					status << "sub.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS32(i, 2) != (i*2-(4+i))) {
					result = false;
					status << "sub.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsS64(i, 2) != (i*2-(4+i))) {
					result = false;
					status << "sub.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 2));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + i));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsF32(i, 2) != (PTXF32)(i*2-(4+i))) {
					result = false;
					status << "sub.f32 incorrect [" << i << "] - expected: " << (float)(i*2+4+i) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 2));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Sub(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF64(i, 2) - (double)(i*2-(4+i))) > 0.1) {
					result = false;
					status << "sub.f64 incorrect [" << i << "] - expected: " << (PTXF64)(i*2+4+i) 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_SubC() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::SubC;

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.cc = PTXInstruction::CC;
			ins.a = imm_uint("r2", PTXOperand::u32, 0x0fffffffe);
			ins.b = reg("r1", PTXOperand::u32, 0);
			ins.c = reg("r3", PTXOperand::u32, 6);
			ins.d = reg("r6", PTXOperand::u32, 5);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 5, 0);
				cta->setRegAsU32(i, 6, 1);	// set the carry flag
			}
			cta->eval_SubC(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = (0x0fffffffe - ((PTXU32)(i*2)));
				if (cta->getRegAsU32(i, 5) != expected) {
					result = false;
					status << "subc.u32 incorrect - got " << cta->getRegAsU32(i,5) 
						<< ", expected " << expected << "\n";
					break;
				}
				// verify carry		
				if (cta->getRegAsU32(i, 6) != 1) {
					result = false;
					status << "subc.u32 failed to set borrow bit\n";
					break;
				}		
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.cc = PTXInstruction::CC;
			ins.a = imm_uint("r2", PTXOperand::s32, 0x0fffffffe);
			ins.b = reg("r1", PTXOperand::s32, 0);
			ins.c = reg("r3", PTXOperand::s32, 6);
			ins.d = reg("r6", PTXOperand::s32, 5);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 5, 0);
				cta->setRegAsU32(i, 6, 1);	// set the carry flag
			}
			cta->eval_SubC(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = (0x0fffffffe - ((PTXS32)(i*2)));
				if (cta->getRegAsS32(i, 5) != expected) {
					result = false;
					status << "subc.s32 incorrect\n";
					break;
				}
				// verify carry			
				if (cta->getRegAsS32(i, 6) != 1) {
					result = false;
					status << "subc.s32 failed to set borrow bit\n";
					break;
				}				
			}
		}

		return result;
	}

	/*!
		Sum of absolute differences

		d = c + ((a<b) ? b-a : a-b);
	*/
	bool test_Sad() {
		bool result = true;
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sad;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.c = reg("r3", PTXOperand::u16, 2);
			ins.d = reg("r4", PTXOperand::u16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 2));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 2);
				cta->setRegAsU16(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 a = (i * 2), b = (4 + i), c = 2;
				PTXU16 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsU16(i, 2) != expected) {
					result = false;
					status << "sad.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.c = reg("r3", PTXOperand::u32, 2);
			ins.d = reg("r4", PTXOperand::u32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 2);
				cta->setRegAsU32(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 a = (i * 2), b = (4 + i), c = 2;
				PTXU32 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsU32(i, 2) != expected) {
					result = false;
					status << "sad.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.c = reg("r3", PTXOperand::u64, 2);
			ins.d = reg("r4", PTXOperand::u64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 2));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 2);
				cta->setRegAsU64(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 a = (i * 2), b = (4 + i), c = 2;
				PTXU64 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsU64(i, 2) != expected) {
					result = false;
					status << "sad.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.c = reg("r3", PTXOperand::s16, 2);
			ins.d = reg("r4", PTXOperand::s16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 2);
				cta->setRegAsS16(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 a = (i * 2), b = (4 + i), c = 2;
				PTXS16 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "sad.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.c = reg("r3", PTXOperand::s32, 2);
			ins.d = reg("r4", PTXOperand::s32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 2);
				cta->setRegAsS32(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (i * 2), b = (4 + i), c = 2;
				PTXS32 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "sad.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.c = reg("r3", PTXOperand::u64, 2);
			ins.d = reg("r4", PTXOperand::u64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 2);
				cta->setRegAsS64(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 a = (i * 2), b = (4 + i), c = 2;
				PTXS64 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "sad.s64 incorrect\n";
					break;
				}
			}
		}

		return result;
	}

#define argmin(a, b) ((a) > (b) ? (b) : (a))
#define argmax(a, b) ((b) > (a) ? (b) : (a))

	bool test_Min() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Min;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 2));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = argmin(i*2, 4+i);
				if (cta->getRegAsU16(i, 2) != expected) {
					result = false;
					status << "min.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = argmin(i*2, 4+i);
				if (cta->getRegAsU32(i, 2) != expected) {
					result = false;
					status << "min.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 2));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = argmin(i*2, 4+i);
				if (cta->getRegAsU64(i, 2) != expected) {
					result = false;
					status << "min.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = argmin(i*2, 4+i);
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "min.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = argmin(i*2, 4+i);
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "min.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = argmin(i*2, 4+i);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "min.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 2));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + i));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = argmin(i*2, 4+i);
				if (cta->getRegAsF32(i, 2) != expected) {
					result = false;
					status << "min.f32 incorrect [" << i << "] - expected: " << (float)(i*2+4+i) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 2));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 expected = argmin(i*2, 4+i);
				if (std::fabs(cta->getRegAsF64(i, 2) - expected) > 0.1) {
					result = false;
					status << "min.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}


	bool test_Max() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Max;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 2));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = argmax(i*2, 4+i);
				if (cta->getRegAsU16(i, 2) != expected) {
					result = false;
					status << "max.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 2));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = argmax(i*2, 4+i);
				if (cta->getRegAsU32(i, 2) != expected) {
					result = false;
					status << "max.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 2));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = argmax(i*2, 4+i);
				if (cta->getRegAsU64(i, 2) != expected) {
					result = false;
					status << "max.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = argmax(i*2, 4+i);
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "max.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = argmax(i*2, 4+i);
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "max.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = argmax(i*2, 4+i);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "max.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 2));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + i));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = argmax(i*2, 4+i);
				if (cta->getRegAsF32(i, 2) != expected) {
					result = false;
					status << "max.f32 incorrect [" << i << "] - expected: " << (float)(i*2+4+i) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 2));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 expected = argmax(i*2, 4+i);
				if (std::fabs(cta->getRegAsF64(i, 2) - expected) > 0.1) {
					result = false;
					status << "max.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Neg() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Neg;

		// bf16
		//
		if (result) {
			ins.type = PTXOperand::bf16;
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.d = reg("r3", PTXOperand::b16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, 0x3fc0); // 1.5
				cta->setRegAsU16(i, 2, 0);
			}
			if (!ins.valid().empty()) {
				result = false;
				status << "neg.bf16 rejected\n";
			}
			else {
				cta->eval_Neg(cta->getActiveContext(), ins);
				for (int i = 0; i < threadCount; i++) {
					if (cta->getRegAsU16(i, 2) != 0xbfc0) { // -1.5
						result = false;
						status << "neg.bf16 incorrect\n";
						break;
					}
				}
			}
		}

		// f16
		//
		if (result) {
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.d = reg("r3", PTXOperand::b16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, 0x3e00); // 1.5
				cta->setRegAsU16(i, 2, 0);
			}
			if (!ins.valid().empty()) {
				result = false;
				status << "neg.f16 rejected\n";
			}
			else {
				cta->eval_Neg(cta->getActiveContext(), ins);
				for (int i = 0; i < threadCount; i++) {
					if (cta->getRegAsU16(i, 2) != 0xbe00) { // -1.5
						result = false;
						status << "neg.f16 incorrect\n";
						break;
					}
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 2));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = -(i*2);
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "neg.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 2));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = -(i*2);
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "neg.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 2));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = -(i*2);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "neg.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 2));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = -(i*2);
				if (cta->getRegAsF32(i, 2) != expected) {
					result = false;
					status << "neg.f32 incorrect [" << i << "] - expected: " << (float)(i*2+4+i) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 2));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 expected = -(i*2);
				if (std::fabs(cta->getRegAsF64(i, 2) - expected) > 0.1) {
					result = false;
					status << "neg.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}


	bool test_Rem() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Rem;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 8 + 8));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsU16(i, 2) != expected) {
					result = false;
					status << "rem.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 8 + 8));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsU32(i, 2) != expected) {
					result = false;
					status << "rem.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 8 + 8));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsU64(i, 2) != expected) {
					result = false;
					status << "rem.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 8 + 8));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "rem.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 8 + 8));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "rem.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 8 + 8));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = ((i * 8 + 8) % (4 + i));
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "rem.s64 incorrect\n";
					break;
				}
			}
		}

		return result;
	}


	bool test_Div() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Div;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i * 8 + 8));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsU16(i, 2) != expected) {
					result = false;
					status << "div.u16 incorrect\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.d = reg("r3", PTXOperand::u32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i * 8 + 8));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsU32(i, 2) != expected) {
					result = false;
					status << "div.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.d = reg("r3", PTXOperand::u64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i * 8 + 8));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsU64(i, 2) != expected) {
					result = false;
					status << "div.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.d = reg("r3", PTXOperand::s16, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i * 8 + 8));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsS16(i, 2) != expected) {
					result = false;
					status << "div.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.d = reg("r3", PTXOperand::s32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i * 8 + 8));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsS32(i, 2) != expected) {
					result = false;
					status << "div.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.d = reg("r3", PTXOperand::s64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i * 8 + 8));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = ((i * 8 + 8) / (4 + i));
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "div.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i * 8 + 8));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + i));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = ((PTXF32)(i * 8 + 8) / (PTXF32)(4 + i));
				if (std::fabs(cta->getRegAsF32(i, 2) - expected) > 0.1f) {
					result = false;
					status << "div.f32 incorrect [" << i << "] - expected: " << (float)expected 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 8 + 8));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = ((PTXF64)(i * 8 + 8) / (PTXF64)(4 + i));
				if (std::fabs(cta->getRegAsF64(i, 2) - expected) > 0.1) {
					result = false;
					status << "div.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Mad() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mad;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.modifier = (PTXInstruction::lo & (~PTXInstruction::hi));
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.c = reg("r3", PTXOperand::u16, 2);
			ins.d = reg("r4", PTXOperand::u16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i + 1));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + 2*i));
				cta->setRegAsU16(i, 2, (PTXU16)i);
				cta->setRegAsU16(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = ((i + 1) * (4 + 2*i) + i);
				if (cta->getRegAsU16(i, 3) != expected) {
					result = false;
					status << "mad.u16 incorrect - expected " << expected 
						<< ", got " << cta->getRegAsU16(i, 3) << "\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.c = reg("r3", PTXOperand::u32, 2);
			ins.d = reg("r4", PTXOperand::u32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i + 1));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + 2*i));
				cta->setRegAsU32(i, 2, (PTXU32)i);
				cta->setRegAsU32(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = ((i + 1) * (4 + 2*i) + i);
				if (cta->getRegAsU32(i, 3) != expected) {
					result = false;
					status << "mad.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.c = reg("r3", PTXOperand::u64, 2);
			ins.d = reg("r4", PTXOperand::u64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i + 1));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + 2*i));
				cta->setRegAsU64(i, 2, (PTXU64)i);
				cta->setRegAsU64(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64  expected = ((i + 1) * (4 + 2*i) + i);
				if (cta->getRegAsU64(i, 3) != expected) {
					result = false;
					status << "mad.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.c = reg("r3", PTXOperand::s16, 2);
			ins.d = reg("r4", PTXOperand::s16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i - 1));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + 2*i));
				cta->setRegAsS16(i, 2, (PTXS16)i);
				cta->setRegAsS16(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = (i - 1) * (4 + 2*i) + (i);
				if (cta->getRegAsS16(i, 3) != expected) {
					result = false;
					status << "mad.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.c = reg("r3", PTXOperand::s32, 2);
			ins.d = reg("r4", PTXOperand::s32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i - 1));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + 2*i));
				cta->setRegAsS32(i, 2, (PTXS32)i);
				cta->setRegAsS32(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = (i - 1) * (4 + 2*i) + (i);
				if (cta->getRegAsS32(i, 3) != expected) {
					result = false;
					status << "mad.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.c = reg("r3", PTXOperand::s64, 2);
			ins.d = reg("r4", PTXOperand::s64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i - 1));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + 2*i));
				cta->setRegAsS64(i, 2, (PTXS64)i);
				cta->setRegAsS64(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = (i - 1) * (4 + 2*i) + (i);
				if (cta->getRegAsS64(i, 3) != expected) {
					result = false;
					status << "mad.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.c = reg("r3", PTXOperand::f32, 2);
			ins.d = reg("r4", PTXOperand::f32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i - 1));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + 2*i));
				cta->setRegAsF32(i, 2, (PTXF32)i);
				cta->setRegAsF32(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = (PTXF32)(i - 1) * (PTXF32)(4 + 2*i) + (PTXF32)(i);
				if (std::fabs(cta->getRegAsF32(i, 3) - expected) > 0.1f) {
					result = false;
					status << "mad.f32 incorrect [" << i << "] - expected: " << (float)expected 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.c = reg("r3", PTXOperand::f64, 2);
			ins.d = reg("r4", PTXOperand::f64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i - 1));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + 2*i));
				cta->setRegAsF64(i, 2, (PTXF64)i);
				cta->setRegAsF64(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = (PTXF64)(i - 1) * (PTXF64)(4 + 2*i) + (PTXF64)(i);
				if (std::fabs(cta->getRegAsF64(i, 3) - expected) > 0.1) {
					result = false;
					status << "mad.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Mul() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mul;

		// f16
		//
		if (result) {
			ins.type = PTXOperand::f16;
			ins.modifier = 0;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.b = reg("r2", PTXOperand::b16, 1);
			ins.c = reg("r3", PTXOperand::b16, 2);
			ins.d = reg("r4", PTXOperand::b16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, 0x3e00); // 1.5
				cta->setRegAsU16(i, 1, 0x4000); // 2.0
				cta->setRegAsU16(i, 2, 0);
				cta->setRegAsU16(i, 3, 0);
			}
			std::string error = ins.valid();
			if (!error.empty()) {
				result = false;
				status << "mul.f16 rejected: " << error << "\n";
			}
			else {
				cta->eval_Mul(cta->getActiveContext(), ins);
				for (int i = 0; i < threadCount; i++) {
					if (cta->getRegAsU16(i, 3) != 0x4200) { // 3.0
						result = false;
						status << "mul.f16 incorrect\n";
						break;
					}
				}
			}
		}

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.modifier = (PTXInstruction::lo & (~PTXInstruction::hi));
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.c = reg("r3", PTXOperand::u16, 2);
			ins.d = reg("r4", PTXOperand::u16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, (PTXU16)(i + 1));
				cta->setRegAsU16(i, 1, (PTXU16)(4 + 2*i));
				cta->setRegAsU16(i, 2, (PTXU16)i);
				cta->setRegAsU16(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = ((i + 1) * (4 + 2*i));
				if (cta->getRegAsU16(i, 3) != expected) {
					result = false;
					status << "mul.u16 incorrect - expected " << expected 
						<< ", got " << cta->getRegAsU16(i, 3) << "\n";
					break;
				}
			}
		}

		// u32
		//
		if (result) {
			ins.type = PTXOperand::u32;
			ins.a = reg("r1", PTXOperand::u32, 0);
			ins.b = reg("r2", PTXOperand::u32, 1);
			ins.c = reg("r3", PTXOperand::u32, 2);
			ins.d = reg("r4", PTXOperand::u32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU32)(i + 1));
				cta->setRegAsU32(i, 1, (PTXU32)(4 + 2*i));
				cta->setRegAsU32(i, 2, (PTXU32)i);
				cta->setRegAsU32(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = ((i + 1) * (4 + 2*i));
				if (cta->getRegAsU32(i, 3) != expected) {
					result = false;
					status << "mul.u32 incorrect\n";
					break;
				}
			}
		}

		// u64
		//
		if (result) {
			ins.type = PTXOperand::u64;
			ins.a = reg("r1", PTXOperand::u64, 0);
			ins.b = reg("r2", PTXOperand::u64, 1);
			ins.c = reg("r3", PTXOperand::u64, 2);
			ins.d = reg("r4", PTXOperand::u64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i + 1));
				cta->setRegAsU64(i, 1, (PTXU64)(4 + 2*i));
				cta->setRegAsU64(i, 2, (PTXU64)i);
				cta->setRegAsU64(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64  expected = ((i + 1) * (4 + 2*i));
				if (cta->getRegAsU64(i, 3) != expected) {
					result = false;
					status << "mul.u64 incorrect\n";
					break;
				}
			}
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.b = reg("r2", PTXOperand::s16, 1);
			ins.c = reg("r3", PTXOperand::s16, 2);
			ins.d = reg("r4", PTXOperand::s16, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, (PTXS16)(i - 1));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + 2*i));
				cta->setRegAsS16(i, 2, (PTXS16)i);
				cta->setRegAsS16(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = (i - 1) * (4 + 2*i);
				if (cta->getRegAsS16(i, 3) != expected) {
					result = false;
					status << "mul.s16 incorrect\n";
					break;
				}
			}
		}

		// s32
		//
		if (result) {
			ins.type = PTXOperand::s32;
			ins.a = reg("r1", PTXOperand::s32, 0);
			ins.b = reg("r2", PTXOperand::s32, 1);
			ins.c = reg("r3", PTXOperand::s32, 2);
			ins.d = reg("r4", PTXOperand::s32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, (PTXS32)(i - 1));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + 2*i));
				cta->setRegAsS32(i, 2, (PTXS32)i);
				cta->setRegAsS32(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = (i - 1) * (4 + 2*i);
				if (cta->getRegAsS32(i, 3) != expected) {
					result = false;
					status << "mul.s32 incorrect\n";
					break;
				}
			}
		}

		// s64
		//
		if (result) {
			ins.type = PTXOperand::s64;
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.c = reg("r3", PTXOperand::s64, 2);
			ins.d = reg("r4", PTXOperand::s64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(i - 1));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + 2*i));
				cta->setRegAsS64(i, 2, (PTXS64)i);
				cta->setRegAsS64(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = (i - 1) * (4 + 2*i);
				if (cta->getRegAsS64(i, 3) != expected) {
					result = false;
					status << "mul.s64 incorrect\n";
					break;
				}
			}
		}

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.c = reg("r3", PTXOperand::f32, 2);
			ins.d = reg("r4", PTXOperand::f32, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(i - 1));
				cta->setRegAsF32(i, 1, (PTXF32)(4 + 2*i));
				cta->setRegAsF32(i, 2, (PTXF32)i);
				cta->setRegAsF32(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = (PTXF32)(i - 1) * (PTXF32)(4 + 2*i);
				if (std::fabs(cta->getRegAsF32(i, 3) - expected) > 0.1f) {
					result = false;
					status << "mul.f32 incorrect [" << i << "] - expected: " << (float)expected 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.c = reg("r3", PTXOperand::f64, 2);
			ins.d = reg("r4", PTXOperand::f64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i - 1));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + 2*i));
				cta->setRegAsF64(i, 2, (PTXF64)i);
				cta->setRegAsF64(i, 3, 0);
			}
			cta->eval_Mul(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 expected = (PTXF64)(i - 1) * (PTXF64)(4 + 2*i);
				if (std::fabs(cta->getRegAsF64(i, 3) - expected) > 0.1) {
					result = false;
					status << "mul.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//
	// Floating-point instructions
	//
	/////////////////////////////////////////////////////////////////////////////////////////////////


	bool test_Rcp() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Rcp;

		double freq = 2.0f / (double)threadCount;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(0.1f + (float)i * freq));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Rcp(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - 1.0f/(PTXF32)(0.1f + (float)i * freq)) > 0.1f) {
					result = false;
					status << "rcp.f32 incorrect [" << i << "] - expected: " 
						<< 1.0f/(PTXF32)(0.1f + (float)i * freq) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(0.1f + (double)i * freq));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Rcp(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF64(i, 2) - 1.0/(0.1 + (double)i * freq)) > 0.1f) {
					result = false;
					status << "rcp.f64 incorrect [" << i << "] - expected: " << 1.0/(0.1 + (double)i * freq)
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Cos() {
		bool result = true;

		PTXInstruction ins;

		// f32
		//
		if (result) {
			float freq = 2 * 3.14159f / (float)threadCount;

			ins.opcode = PTXInstruction::Cos;
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)((float)i * freq));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Cos(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - (PTXF32)cos((float)i * freq)) > 0.1f) {
					result = false;
					status << "cos.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)cos((float)i * freq) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}
		return result;
	}

	bool test_Sin() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sin;

		// f32
		//
		if (result) {
			float freq = 2 * 3.14159f / (float)threadCount;

			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)((float)i * freq));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Sin(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - (PTXF32)sin((float)i * freq)) > 0.1f) {
					result = false;
					status << "sin.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)sin((float)i * freq) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}
	
	bool test_CopySign() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fma;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				int as = (!(i & 0x01) ? -1 : 1);
				int bs = (!(i % 0x02) ? 1 : -1);
				cta->setRegAsF32(i, 0, (PTXF32)((float)(as * i) / (float)threadCount * 4.2f));
				cta->setRegAsF32(i, 1, (PTXF32)((float)(bs * i) / (float)threadCount * 2.7f));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Fma(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 got = cta->getRegAsF32(i, 2);
				
				PTXF32 a = cta->getRegAsF32(i, 0);
				PTXF32 b = cta->getRegAsF32(i, 1);
				
				PTXF32 exp = b;
				if (a < 0) {
					exp = -std::fabs(b);
				}
				else {
					exp = std::fabs(b);
				}
					
				if (std::fabs(got - exp) > 0.1f) {
					result = false;
					status << "fma.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)exp
						<< ", got " << got << "\n";
					break;
				}
			}
		}
		
		// f64
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				int as = (!(i & 0x01) ? -1 : 1);
				int bs = (!(i % 0x02) ? 1 : -1);
				cta->setRegAsF64(i, 0, (PTXF64)((double)(as * i) / (double)threadCount * 1.2));
				cta->setRegAsF64(i, 1, (PTXF64)((double)(bs * i) / (double)threadCount * 7.7));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Fma(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 got = cta->getRegAsF64(i, 2);
				
				PTXF64 a = cta->getRegAsF64(i, 0);
				PTXF64 b = cta->getRegAsF64(i, 1);
				
				PTXF64 exp = b;
				if (a < 0) {
					exp = -std::fabs(b);
				}
				else {
					exp = std::fabs(b);
				}
					
				if (std::fabs(got - exp) > 0.1) {
					result = false;
					status << "fma.f64 incorrect [" << i << "] - expected: " 
						<< exp
						<< ", got " << got << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Ex2() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ex2;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)((float)i / (float)threadCount * 4.0f));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Ex2(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - (PTXF32)exp2((float)i / (float)threadCount * 4.0f)) > 0.1f) {
					result = false;
					status << "ex2.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)exp2((float)i / (float)threadCount * 4.0f) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Fma() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fma;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.c = reg("r4", PTXOperand::f32, 3);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)((float)i / (float)threadCount * 4.0f));
				cta->setRegAsF32(i, 1, (PTXF32)((float)i / (float)threadCount * 2.0f));
				cta->setRegAsF32(i, 3, (PTXF32)((float)i / (float)threadCount * 0.5f));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Fma(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 got = cta->getRegAsF32(i, 2);
				PTXF32 exp = (float)i / (float)threadCount * 4.0f * (float)i / (float)threadCount * 2.0f +
					(float)i / (float)threadCount * 0.5f;
					
				if (std::fabs(got - exp) > 0.1f) {
					result = false;
					status << "fma.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)exp
						<< ", got " << got << "\n";
					break;
				}
			}
		}
		
		// f64
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.c = reg("r4", PTXOperand::f64, 3);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF32)((double)i / (double)threadCount * 4.5));
				cta->setRegAsF64(i, 1, (PTXF32)((double)i / (double)threadCount * 2.25));
				cta->setRegAsF64(i, 3, (PTXF32)((double)i / (double)threadCount * 0.55));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Fma(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 got = cta->getRegAsF32(i, 2);
				PTXF32 exp = (double)i / (double)threadCount * 4.5 * (double)i / (double)threadCount * 2.25 +
					(double)i / (double)threadCount *  0.55;
					
				if (std::fabs(got - exp) > 0.1) {
					result = false;
					status << "fma.f64 incorrect [" << i << "] - expected: " 
						<< exp
						<< ", got " << got << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Bf16Fma() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fma;
		ins.type = PTXOperand::bf16;
		ins.modifier = PTXInstruction::rn;
		ins.a = reg("r1", PTXOperand::b16, 0);
		ins.b = reg("r2", PTXOperand::b16, 1);
		ins.c = reg("r4", PTXOperand::b16, 3);
		ins.d = reg("r3", PTXOperand::b16, 2);

		// 1.0 * 1.0078125 + 0.00390625 is halfway between
		// 0x3f81 and 0x3f82, so round to the even result 0x3f82.
		for (int i = 0; i < threadCount; i++) {
			cta->setRegAsU16(i, 0, 0x3f80);
			cta->setRegAsU16(i, 1, 0x3f81);
			cta->setRegAsU16(i, 3, 0x3b80);
			cta->setRegAsU16(i, 2, 0);
		}

		cta->eval_Fma(cta->getActiveContext(), ins);
		for (int i = 0; i < threadCount; i++) {
			if (cta->getRegAsU16(i, 2) != 0x3f82) {
				status << "fma.rn.bf16 incorrect [" << i << "]\n";
				return false;
			}
		}

		return true;
	}

	bool test_F16Fma() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fma;
		ins.type = PTXOperand::f16;
		ins.modifier = PTXInstruction::rn;
		ins.a = reg("a", PTXOperand::b16, 0);
		ins.b = reg("b", PTXOperand::b16, 1);
		ins.c = reg("c", PTXOperand::b16, 3);
		ins.d = reg("d", PTXOperand::b16, 2);

		// 1.0 * (1.0 + 2^-10) - 2^-11 is halfway between
		// 1.0 and the next half value, so round to the even result 1.0.
		for (int i = 0; i < threadCount; i++) {
			cta->setRegAsU16(i, 0, 0x3c00);
			cta->setRegAsU16(i, 1, 0x3c01);
			cta->setRegAsU16(i, 3, 0x9000);
			cta->setRegAsU16(i, 2, 0);
		}

		cta->eval_Fma(cta->getActiveContext(), ins);
		for (int i = 0; i < threadCount; i++) {
			if (cta->getRegAsU16(i, 2) != 0x3c00) {
				status << "fma.rn.f16 incorrect [" << i << "]\n";
				return false;
			}
		}

		return true;
	}

	bool test_Mma() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mma;
		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn;

		auto vector = [this](PTXOperand::DataType type,
			PTXOperand::DataType elementType, PTXOperand::Vec vec,
			int firstRegister, int count) {
			PTXOperand operand;
			operand.addressMode = PTXOperand::Register;
			operand.type = type;
			operand.vec = vec;
			for(int i = 0; i < count; ++i) {
				operand.array.push_back(reg("mma", elementType,
					(PTXOperand::RegisterType)(firstRegister + i)));
			}
			return operand;
		};

		ins.d = vector(PTXOperand::f32, PTXOperand::f32,
			PTXOperand::v4, 0, 4);
		ins.c = vector(PTXOperand::f32, PTXOperand::f32,
			PTXOperand::v4, 6, 4);
		ins.a = vector(PTXOperand::f16, PTXOperand::b32,
			PTXOperand::v4, 0, 4);
		ins.b = vector(PTXOperand::f16, PTXOperand::b32,
			PTXOperand::v2, 4, 2);

		const PTXU16 f16Values[17] = {
			0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500,
			0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980,
			0x4a00, 0x4a80, 0x4b00, 0x4b80, 0x4c00
		};
		cta->reset();
		for(int thread = 0; thread < threadCount; ++thread) {
			const int lane = thread & 31;
			const int groupID = lane >> 2;
			const int threadInGroup = lane & 3;
			for(int regIndex = 0; regIndex < 4; ++regIndex) {
				PTXU32 packed = 0;
				for(int half = 0; half < 2; ++half) {
					const int i = regIndex * 2 + half;
					const int row = (i < 2 || (i >= 4 && i < 6))
						? groupID : groupID + 8;
					const int value = row + 1;
					packed |= (PTXU32)f16Values[value] << (16 * half);
				}
				cta->setRegAsB32(thread, regIndex, packed);
			}
			for(int regIndex = 0; regIndex < 2; ++regIndex) {
				PTXU32 packed = 0;
				for(int half = 0; half < 2; ++half) {
					const int col = groupID;
					packed |= (PTXU32)f16Values[col + 1] << (16 * half);
				}
				cta->setRegAsB32(thread, 4 + regIndex, packed);
			}
			for(int i = 0; i < 4; ++i) {
				const int row = groupID + (i >= 2 ? 8 : 0);
				const int col = threadInGroup * 2 + (i & 1);
				cta->setRegAsF32(thread, 6 + i, (PTXF32)(100 * row + col));
			}
		}

		cta->eval_Mma(cta->getActiveContext(), ins);
		for(int thread = 0; thread < threadCount; ++thread) {
			const int lane = thread & 31;
			const int groupID = lane >> 2;
			const int threadInGroup = lane & 3;
			for(int regIndex = 0; regIndex < 4; ++regIndex) {
				const int row = groupID + (regIndex >= 2 ? 8 : 0);
				const int col = threadInGroup * 2 + (regIndex & 1);
				const PTXF32 expected = 16.0f * (row + 1) * (col + 1)
					+ (PTXF32)(100 * row + col);
				if(std::fabs(cta->getRegAsF32(thread, regIndex) - expected) > 0.001f) {
					status << "mma.m16n8k16.f16 incorrect ["
						<< thread << "]\n";
					return false;
				}
			}
		}

		ins.a.type = PTXOperand::bf16;
		ins.b.type = PTXOperand::bf16;
		const PTXU32 bf16One = 0x3f803f80u;
		const PTXU32 bf16Two = 0x40004000u;
		cta->reset();
		for(int thread = 0; thread < threadCount; ++thread) {
			for(int regIndex = 0; regIndex < 4; ++regIndex) {
				cta->setRegAsB32(thread, regIndex, bf16One);
			}
			for(int regIndex = 4; regIndex < 6; ++regIndex) {
				cta->setRegAsB32(thread, regIndex, bf16Two);
			}
			for(int regIndex = 6; regIndex < 10; ++regIndex) {
				cta->setRegAsF32(thread, regIndex, 3.0f);
			}
		}

		cta->eval_Mma(cta->getActiveContext(), ins);
		for(int thread = 0; thread < threadCount; ++thread) {
			for(int regIndex = 0; regIndex < 4; ++regIndex) {
				if(std::fabs(cta->getRegAsF32(thread, regIndex) - 35.0f) > 0.001f) {
					status << "mma.m16n8k16.bf16 incorrect ["
						<< thread << "]\n";
					return false;
				}
			}
		}

		// MMA is warp-collective: a partially active warp must not execute it.
		cta->getActiveContext().active[0] = false;
		bool rejectedPartialWarp = false;
		try {
			cta->eval_Mma(cta->getActiveContext(), ins);
		}
		catch (RuntimeException &) {
			rejectedPartialWarp = true;
		}
		cta->getActiveContext().active[0] = true;
		if (!rejectedPartialWarp) {
			status << "mma.m16n8k16 accepted a partial warp\n";
			return false;
		}

		return true;
	}

	bool test_Lg2() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Lg2;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(0.5f + (float)i / (float)threadCount * 4.0f));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Lg2(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - (PTXF32)log2(0.5f + (float)i / (float)threadCount * 4.0f)) > 0.1f) {
					result = false;
					status << "lg2.f32 incorrect [" << i 
						<< "] - log2(" << (0.5f + (float)i / (float)threadCount * 4.0f) << ") - expected: " 
						<< (PTXF32)log2(0.5f + (float)i / (float)threadCount * 4.0f) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}
		return result;
	}

	bool test_Sqrt() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sqrt;

		double freq = 2.0f / (double)threadCount;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(0.1f + (float)i * freq));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Sqrt(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - (PTXF32)sqrt(0.1f + (float)i * freq)) > 0.1f) {
					result = false;
					status << "sqrt.f32 incorrect [" << i << "] - expected: " 
						<< (PTXF32)sqrt(0.1f + (float)i * freq) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(0.1f + (double)i * freq));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Sqrt(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF64(i, 2) - sqrt(0.1 + (double)i * freq)) > 0.1f) {
					result = false;
					status << "sqrt.f64 incorrect [" << i << "] - expected: " << sqrt(0.1 + (double)i * freq)
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_Rsqrt() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Rsqrt;

		double freq = 2.0f / (double)threadCount;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.d = reg("r3", PTXOperand::f32, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0, (PTXF32)(0.1f + (float)i * freq));
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Rsqrt(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF32(i, 2) - 1.0f/(PTXF32)sqrt(0.1f + (float)i * freq)) > 0.1f) {
					result = false;
					status << "rsqrt.f32 incorrect [" << i << "] - expected: " 
						<< 1.0f/(PTXF32)sqrt(0.1f + (float)i * freq) 
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		// f64
		//
		if (result) {
			ins.type = PTXOperand::f64;
			ins.a = reg("r1", PTXOperand::f64, 0);
			ins.d = reg("r3", PTXOperand::f64, 2);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(0.1f + (double)i * freq));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Rsqrt(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (std::fabs(cta->getRegAsF64(i, 2) - 1.0/sqrt(0.1 + (double)i * freq)) > 0.1f) {
					result = false;
					status << "rsqrt.f64 incorrect [" << i << "] - expected: " << 1.0/sqrt(0.1 + (double)i * freq)
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//
	// logical and shift instructions
	//
	/////////////////////////////////////////////////////////////////////////////////////////////////

/*!
		Tests several forms of the and instruction
	*/
	bool test_And() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::And;

		cta->reset();

		// b16
		//
		if (result) {
			ins.type = PTXOperand::b16;
			ins.d = reg("r3", PTXOperand::b16, 0);
			ins.a = reg("r1", PTXOperand::b16, 1);
			ins.b = reg("r2", PTXOperand::b16, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB16(t, 1, t);
				cta->setRegAsB16(t, 2, 3*t);
			}
			cta->eval_And(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB16 expected = (t) & (3*t);
				if (cta->getRegAsB16(t, 0) != expected) {
					result = false;
					status << "and.b16 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b32
		//
		if (result) {
			ins.type = PTXOperand::b32;
			ins.d = reg("r3", PTXOperand::b32, 0);
			ins.a = reg("r1", PTXOperand::b32, 1);
			ins.b = reg("r2", PTXOperand::b32, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB32(t, 1, t);
				cta->setRegAsB32(t, 2, 3*t);
			}
			cta->eval_And(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB32 expected = (t) & (3*t);
				if (cta->getRegAsB32(t, 0) != expected) {
					result = false;
					status << "and.b32 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b64
		//
		if (result) {
			ins.type = PTXOperand::b64;
			ins.d = reg("r3", PTXOperand::b64, 0);
			ins.a = reg("r1", PTXOperand::b64, 1);
			ins.b = reg("r2", PTXOperand::b64, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB64(t, 1, t);
				cta->setRegAsB64(t, 2, 3*t);
			}
			cta->eval_And(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB64 expected = (t) & (3*t);
				if (cta->getRegAsB64(t, 0) != expected) {
					result = false;
					status << "and.b64 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}
		return result;
	}

	/*!
		Tests several forms of the and instruction
	*/
	bool test_Or() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Or;

		cta->reset();

		// b16
		//
		if (result) {
			ins.type = PTXOperand::b16;
			ins.d = reg("r3", PTXOperand::b16, 0);
			ins.a = reg("r1", PTXOperand::b16, 1);
			ins.b = reg("r2", PTXOperand::b16, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB16(t, 1, t);
				cta->setRegAsB16(t, 2, 3*t);
			}
			cta->eval_Or(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB16 expected = (t) | (3*t);
				if (cta->getRegAsB16(t, 0) != expected) {
					result = false;
					status << "or.b16 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b32
		//
		if (result) {
			ins.type = PTXOperand::b32;
			ins.d = reg("r3", PTXOperand::b32, 0);
			ins.a = reg("r1", PTXOperand::b32, 1);
			ins.b = reg("r2", PTXOperand::b32, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB32(t, 1, t);
				cta->setRegAsB32(t, 2, 3*t);
			}
			cta->eval_Or(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB32 expected = (t) | (3*t);
				if (cta->getRegAsB32(t, 0) != expected) {
					result = false;
					status << "or.b32 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b64
		//
		if (result) {
			ins.type = PTXOperand::b64;
			ins.d = reg("r3", PTXOperand::b64, 0);
			ins.a = reg("r1", PTXOperand::b64, 1);
			ins.b = reg("r2", PTXOperand::b64, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB64(t, 1, t);
				cta->setRegAsB64(t, 2, 3*t);
			}
			cta->eval_Or(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB64 expected = (t) | (3*t);
				if (cta->getRegAsB64(t, 0) != expected) {
					result = false;
					status << "or.b64 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}
		return result;
	}

	/*!
		Tests several forms of the and instruction
	*/
	bool test_Xor() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Xor;

		cta->reset();

		// b16
		//
		if (result) {
			ins.type = PTXOperand::b16;
			ins.d = reg("r3", PTXOperand::b16, 0);
			ins.a = reg("r1", PTXOperand::b16, 1);
			ins.b = reg("r2", PTXOperand::b16, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB16(t, 1, t);
				cta->setRegAsB16(t, 2, 3*t);
			}
			cta->eval_Xor(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB16 expected = ((t) ^ (3*t));
				if (cta->getRegAsB16(t, 0) != expected) {
					result = false;
					status << "xor.b16 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b32
		//
		if (result) {
			ins.type = PTXOperand::b32;
			ins.d = reg("r3", PTXOperand::b32, 0);
			ins.a = reg("r1", PTXOperand::b32, 1);
			ins.b = reg("r2", PTXOperand::b32, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB32(t, 1, t);
				cta->setRegAsB32(t, 2, 3*t);
			}
			cta->eval_Xor(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB32 expected = ((t) ^ (3*t));
				if (cta->getRegAsB32(t, 0) != expected) {
					result = false;
					status << "xor.b32 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS32(t, 0) << "\n";
				}
			}
		}

		// b64
		//
		if (result) {
			ins.type = PTXOperand::b64;
			ins.d = reg("r3", PTXOperand::b64, 0);
			ins.a = reg("r1", PTXOperand::b64, 1);
			ins.b = reg("r2", PTXOperand::b64, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB64(t, 1, t);
				cta->setRegAsB64(t, 2, 3*t);
			}
			cta->eval_Xor(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB64 expected = ((t) ^ (3*t));
				PTXB64 got = cta->getRegAsB64(t, 0);
				if (got != expected) {
					result = false;
					status << "xor.b64 failed (thread " << t << "): expected " << expected 
						<< ", got " << got << "\n";
				}
			}
		}
		return result;
	}

	/*!
		Tests several forms of the and instruction
	*/
	bool test_Not() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Not;

		cta->reset();

		// b16
		//
		if (result) {
			ins.type = PTXOperand::b16;
			ins.d = reg("r3", PTXOperand::b16, 0);
			ins.a = reg("r1", PTXOperand::b16, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB16(t, 1, t);
				cta->setRegAsB16(t, 0, 0);
			}
			cta->eval_Not(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB16 expected = (~t);
				if (cta->getRegAsB16(t, 0) != expected) {
					result = false;
					status << "xor.b16 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b32
		//
		if (result) {
			ins.type = PTXOperand::b32;
			ins.d = reg("r3", PTXOperand::b32, 0);
			ins.a = reg("r1", PTXOperand::b32, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB32(t, 1, t);
				cta->setRegAsB32(t, 0, 0);
			}
			cta->eval_Not(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB32 expected = (~t);
				if (cta->getRegAsB32(t, 0) != expected) {
					result = false;
					status << "xor.b32 failed (thread " << t << "): expected " << expected 
						<< ", got " << cta->getRegAsS16(t, 0) << "\n";
				}
			}
		}

		// b64
		//
		if (result) {
			ins.type = PTXOperand::b64;
			ins.d = reg("r3", PTXOperand::b64, 0);
			ins.a = reg("r1", PTXOperand::b64, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsB64(t, 1, t);
				cta->setRegAsB64(t, 0, 0);
			}
			cta->eval_Not(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				PTXB64 expected = (~t);
				PTXB64 got = cta->getRegAsB64(t, 0);
				if (got != expected) {
					result = false;
					status << "xor.b64 failed (thread " << t << "): expected " << expected 
						<< ", got " << got << "\n";
				}
			}
		}
		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// Load, store

	/*!
		d = a;           // named variable a


		d = *a;          // register
		d = *(a+immOff); // register-plus-offset
		d = *(immAddr);  // immediate address
	*/
	bool test_Ld_global() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;

		cta->reset();

		//
		// Global memory
		//

		ins.addressSpace = PTXInstruction::Global;

		// register indirect
		if (result) {
			PTXU32 source[2] = { 0xaa551376 };
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != 0xaa551376) {
					result = false;
					status << "ld.u32.global failed - [" << i << "] - expected 0xaa551376, got " 
						<< cta->getRegAsU32(i, 5) << "\n";
				}
			}
		}

		// register indirect with offset
		if (result) {
			PTXU32 source[4] = { 0xaa551376, 0x75320011, 0x9988aaff, 0x00};
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.a.offset = sizeof(PTXU32);
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 got = cta->getRegAsU32(i, 5);
				if (got != 0x75320011) {
					result = false;
					status << "ld.u32.global failed - [" << i << "] - expected 0x75320011, got 0x" << hex
						<< got << dec << "\n";
				}
			}
		}

		// immediate
		if (result) {
			PTXU32 source[4] = { 0xaa551376, 0x75320011, 0x99b8aafd, 0x00};
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a.type = PTXOperand::u64;
			ins.a.addressMode = PTXOperand::Immediate;
			ins.a.offset = 0;
			ins.a.imm_uint = (unsigned long)&source[2];
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 got = cta->getRegAsU32(i, 5);
				if (got != 0x99b8aafd) {
					result = false;
					status << "ld.u32.global failed - [" << i << "] - expected 0x99b8aafd, got 0x" << hex
						<< got << dec << "\n";
				}
			}
		}
		return result;
	}

	bool test_Ld_shared() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;

		cta->reset();

		//
		// Shared memory
		//

		PTXU32 *shared = (PTXU32 *)cta->functionCallStack.sharedMemoryPointer();

		ins.addressSpace = PTXInstruction::Shared;

		shared[0] = 0x55aa3377;		
		shared[1] = 0x98765431;
		shared[2] = 0x21003acd;
		shared[3] = 0x10081983;
		shared[4] = 0x05311984;

		// register indirect
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i % 4) * sizeof(PTXU32));
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != shared[i % 4]) {
					result = false;
					status << "ld.u32.shared [reg] failed - [" << i << "] - expected 0x" << hex << shared[i % 4] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		// register indirect + offset
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.a.offset = sizeof(PTXU32);
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i % 4) * sizeof(PTXU32));
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != shared[1 + (i % 4)]) {
					result = false;
					status << "ld.u32.shared [reg+offset] failed - [" << i << "] - expected 0x" << hex << shared[1 + (i % 4)] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		// immediate
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Immediate;
			ins.a.offset = 0;
			ins.a.imm_uint = (unsigned long)(2*sizeof(PTXU32));
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, 0);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != shared[2]) {
					result = false;
					status << "ld.u32.shared [imm] failed - [" << i << "] - expected 0x" << hex << shared[2] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		return result;
	}

	bool test_Ld_param() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;

		cta->reset();

		//
		// Parameter memory
		//

		// we only need this to set values, the instruction itself sees it as just another address space
		PTXU32 *space = (PTXU32 *)cta->kernel->ArgumentMemory;

		ins.addressSpace = PTXInstruction::Param;

		space[0] = 0x55aa3377;		
		space[1] = 0x98765431;
		space[2] = 0x21003acd;
		space[3] = 0x10081983;
		space[4] = 0x05311984;

		// register indirect
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i % 4) * sizeof(PTXU32));
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != space[i % 4]) {
					result = false;
					status << "ld.u32.param [reg] failed - [" << i << "] - expected 0x" << hex << space[i % 4] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		// register indirect + offset
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.a.offset = sizeof(PTXU32);
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)(i % 4) * sizeof(PTXU32));
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != space[1 + (i % 4)]) {
					result = false;
					status << "ld.u32.param [reg+offset] failed - [" << i << "] - expected 0x" << hex << space[1 + (i % 4)] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		// immediate
		if (result) {
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Immediate;
			ins.a.offset = 0;
			ins.a.imm_uint = (unsigned long)(2*sizeof(PTXU32));
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, 0);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU32(i, 5) != space[2]) {
					result = false;
					status << "ld.u32.param [imm] failed - [" << i << "] - expected 0x" << hex << space[2] 
						<< ", got 0x" << cta->getRegAsU32(i, 5) << dec << "\n";
				}
			}
		}

		return result;
	}

	bool test_Ld_global_vec() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;

		cta->reset();

		//
		// Global memory
		//

		ins.addressSpace = PTXInstruction::Global;

		/*
		// print out PTX*** sizes to be sure
		status << "u16: " << sizeof(PTXU16) << ", u32: " << sizeof(PTXU32) << ", u64: " << sizeof(PTXU64) << "\n";
		status << "s16: " << sizeof(PTXS16) << ", s32: " << sizeof(PTXS32) << ", s64: " << sizeof(PTXS64) << "\n";
		status << "        f32: " << sizeof(PTXF32) << ", f64: " << sizeof(PTXF64) << "\n";
		*/

		if (result) {
			PTXU32 source[2] __attribute__((aligned(2*sizeof(PTXU32)))) 
				= { 0x0aa551376, 0x091834321 };
			ins.d = reg("rd", PTXOperand::u32, 1);
			ins.d.vec = PTXOperand::v2;
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.a.offset = 0;
			ins.d.array.resize( 2 );
			ins.d.array[0] = reg("rd[0]", PTXOperand::u32, 1);
			ins.d.array[1] = reg("rd[1]", PTXOperand::u32, 2);
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 1, 0);
				cta->setRegAsU32(i, 2, 0);
				cta->setRegAsU32(i, 3, 0);
				cta->setRegAsU32(i, 4, 0);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 d0 = cta->getRegAsU32(i, 1);
				PTXU32 d1 = cta->getRegAsU32(i, 2);
				if (d0 != 0xaa551376 || d1 != 0x91834321) {
					result = false;
					status << "ld.u32.global.v2 failed - [" << i << 
						"] - expected { 0xaa551376, 0x91834321 }, got { 0x" << hex 
						<< d0 << ", 0x" << d1 << dec << "}\n";
				}
			}
		}		

		if (result) {
			PTXU32 source[4] __attribute__((aligned(4*sizeof(PTXU32)))) 
				= { 0x0aa551376, 0x091834321, 0x9f995432, 0x12345678 };
			ins.d = reg("rd", PTXOperand::u32, 1);
			ins.d.vec = PTXOperand::v4;
			ins.d.array.resize( 4 );
			ins.d.array[0] = reg("rd[0]", PTXOperand::u32, 1);
			ins.d.array[1] = reg("rd[1]", PTXOperand::u32, 2);
			ins.d.array[2] = reg("rd[2]", PTXOperand::u32, 3);
			ins.d.array[3] = reg("rd[3]", PTXOperand::u32, 4);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.a.offset = 0;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 1, 0);
				cta->setRegAsU32(i, 2, 0);
				cta->setRegAsU32(i, 3, 0);
				cta->setRegAsU32(i, 4, 0);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				for (int j = 0; j < 4; j++) {
					if (cta->getRegAsU32(i, 1+j) != source[j]) {
						status << "ld.u32.global.v4 failed\n";
						result = false; 
						break;
					}
				}
			}
		}


		return result;
	}

	bool test_Ld() {
		bool result = true;
		
		// scalar loads
		result = (result && test_Ld_global() && test_Ld_shared());

		// vector loads
		result = (result && test_Ld_global_vec());

		return result;
	}

	/*!
		Store to global memory
	*/
	bool test_St_global() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::St;

		cta->reset();

		//
		// Global memory
		//

		ins.addressSpace = PTXInstruction::Global;

		// register indirect
		if (result) {
			PTXU32 source[64] = { 0 };
			ins.d = reg("ra", PTXOperand::u64, 5);
			ins.a = reg("rd", PTXOperand::u32, 0);
			ins.d.addressMode = PTXOperand::Indirect;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 5, (PTXU64)&source[i]);
				cta->setRegAsU32(i, 0, i);
			}

			cta->eval_St(cta->getActiveContext(), ins);
			for (PTXU32 i = 0; i < (PTXU32)threadCount; i++) {
				if (source[i] != i) {
					result = false;
					status << "st.u32.global [reg] failed\n";
				}
			}
		}

		// register indirect + offset
		if (result) {
			PTXU32 source[65] = { 0 };
			ins.d = reg("ra", PTXOperand::u64, 5);
			ins.a = reg("rd", PTXOperand::u32, 0);
			ins.d.addressMode = PTXOperand::Indirect;
			ins.d.offset = sizeof(PTXU32);
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 5, (PTXU64)&source[i]);
				cta->setRegAsU32(i, 0, i);
			}

			cta->eval_St(cta->getActiveContext(), ins);
			for (PTXU32 i = 0; i < (PTXU32)threadCount; i++) {
				if (source[i+1] != i) {
					result = false;
					status << "st.u32.global [reg+off] failed. Expected " << (i+1) << ", got " << source[i+1] << "\n";
				}
			}
		}

		// register indirect + offset
		if (result) {
			PTXU32 source[65] = { 0 };
			ins.d = reg("ra", PTXOperand::u64, 5);
			ins.a = reg("rd", PTXOperand::u32, 0);
			ins.d.addressMode = PTXOperand::Immediate;
			ins.d.offset = 0;
			ins.d.imm_uint = (PTXU64)&source[0];
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, i);
			}

			cta->eval_St(cta->getActiveContext(), ins);

			if (source[0] != (PTXU32)threadCount - 1) {
				result = false;
				status << "st.u32.global [imm] failed\n";
			}
		}

		return result;
	}

	/*!
		Store to global memory
	*/
	bool test_St_vec() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::St;

		cta->reset();

		//
		// Global memory
		//

		ins.addressSpace = PTXInstruction::Global;

		// register indirect
		if (result) {
			PTXU32 block[128] __attribute__((aligned(4*sizeof(PTXU32)))) = {0};

			ins.a = reg("rval", PTXOperand::u32, 1);
			ins.a.array.resize( 4 );
			ins.a.array[0] = reg("rval[0]", PTXOperand::u32, 1);
			ins.a.array[1] = reg("rval[1]", PTXOperand::u32, 2);
			ins.a.array[2] = reg("rval[2]", PTXOperand::u32, 3);
			ins.a.array[3] = reg("rval[3]", PTXOperand::u32, 4);
			ins.a.vec = PTXOperand::v4;

			ins.d = reg("raddr", PTXOperand::u64, 0);
			ins.d.addressMode = PTXOperand::Indirect;
			ins.d.offset = 0;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, (PTXU64)&block[i*4]);
				cta->setRegAsU32(i, 1, i);
				cta->setRegAsU32(i, 2, i*2);
				cta->setRegAsU32(i, 3, i*3);
				cta->setRegAsU32(i, 4, i*4);
			}

			cta->eval_St(cta->getActiveContext(), ins);
			for (PTXU32 i = 0; i < (PTXU32)threadCount; i++) {
				for (PTXU32 j = 0; j < 4; j++) {
					if (block[i*4+j] != i * (j+1)) {
						result = false;
						status << "st.u32.global.v4 [reg] failed\n";
					}
				}
			}
		}

		return result;
	}

	bool test_St() {
		bool result = true;
		
		// scalar stores
		result = (result && test_St_global());

		// vector stores
		result = (result && test_St_vec());

		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// mov, cvt
	
	bool test_Mov() {
		bool result = true;

		/*
			mov.f32 d,a;
			mov.u16 u,v;
			mov.f32 k,0.1;
			mov.u32 ptr, A;       // move address of A into ptr
			mov.u32 ptr, A[5];    // move address of A[5] into ptr
			mov.b32 addr, myFunc; // get address of myFunc
		*/

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mov;

		cta->reset();

		// from register

		// from special register tidX
		if (result) {
			ins.d = reg("r6", PTXOperand::u16, 0);
			ins.a = sreg(PTXOperand::tid, PTXOperand::ix);
			ins.type = PTXOperand::u16;
			cta->eval_Mov(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 0) != (PTXU16)i) {
					result = false;
					status << "mov.u32 r6, tidX failed\n";
				}
			}
		}

		// from special register tidY
		if (result) {
			ins.d = reg("r6", PTXOperand::u16, 0);
			ins.a = sreg(PTXOperand::tid, PTXOperand::iy);
			ins.type = PTXOperand::u16;
			cta->eval_Mov(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 0) != 0) {
					result = false;
					status << "mov.u32 r6, tidY failed\n";
				}
			}
		}

		// from special register ntidX
		if (result) {
			ins.d = reg("r6", PTXOperand::u16, 0);
			ins.a = sreg(PTXOperand::ntid, PTXOperand::ix);
			ins.type = PTXOperand::u16;
			cta->eval_Mov(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 0) != threadCount) {
					result = false;
					status << "mov.u32 r6, ntidX failed\n";
				}
			}
		}

		// from special register ntidY
		if (result) {
			ins.d = reg("r6", PTXOperand::u16, 0);
			ins.a = sreg(PTXOperand::ntid, PTXOperand::iy);
			ins.type = PTXOperand::u16;
			cta->eval_Mov(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (cta->getRegAsU16(i, 0) != 1) {
					result = false;
					status << "mov.u32 r6, ntidY failed\n";
				}
			}
		}

		// pack a 16-bit immediate and register into a 32-bit destination
		if (result) {
			ins.d = reg("f10", PTXOperand::f32, 0);
			ins.a = PTXOperand();
			ins.a.addressMode = PTXOperand::Register;
			ins.a.type = PTXOperand::s16;
			ins.a.vec = PTXOperand::v2;
			ins.a.array.push_back(imm_uint("0", PTXOperand::s16, 0));
			ins.a.array.push_back(reg("rs1", PTXOperand::s16, 1));
			ins.type = PTXOperand::b32;

			for (int i = 0; i < threadCount; ++i) {
				cta->setRegAsB16(i, 1, 0x3f80);
			}

			cta->eval_Mov(cta->getActiveContext(), ins);

			for (int i = 0; i < threadCount; ++i) {
				if (cta->getRegAsB32(i, 0) != 0x3f800000) {
					result = false;
					status << "mov.b32 f10, {0, rs1} failed\n";
					break;
				}
			}
		}

		// pack two 16-bit immediates into a 32-bit destination
		if (result) {
			ins.a = PTXOperand();
			ins.a.addressMode = PTXOperand::Register;
			ins.a.type = PTXOperand::b16;
			ins.a.vec = PTXOperand::v2;
			ins.a.array.push_back(imm_uint("5", PTXOperand::b16, 5));
			ins.a.array.push_back(imm_uint("3", PTXOperand::b16, 3));

			cta->eval_Mov(cta->getActiveContext(), ins);

			for (int i = 0; i < threadCount; ++i) {
				if (cta->getRegAsB32(i, 0) != 0x00030005) {
					result = false;
					status << "mov.b32 f10, {5, 3} failed\n";
					break;
				}
			}
		}

		// from label
	
		return result;
	}

	bool test_Cvt() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Cvt;

		cta->reset();

		// cvt.rn.bf16.f32
		ins.type = PTXOperand::bf16;
		ins.modifier = PTXInstruction::rn;
		ins.d = reg("d", PTXOperand::b16, 0);
		ins.a = reg("a", PTXOperand::f32, 1);

		const PTXU32 input[] = {
			0x3f800000, // exact
			0x3f807fff, // below halfway
			0x3f808000, // halfway, upper even
			0x3f808001, // above halfway
			0x3f818000, // halfway, upper odd
			0x00000000, // +0
			0x80000000, // -0
			0x7f800000, // +infinity
			0x7fc00000  // NaN
		};
		const PTXU16 expected[] = {
			0x3f80,
			0x3f80,
			0x3f80,
			0x3f81,
			0x3f82,
			0x0000,
			0x8000,
			0x7f80,
			0x7fff
		};
		const int cases = sizeof(input) / sizeof(input[0]);

		for (int i = 0; i < threadCount; ++i) {
			cta->setRegAsU32(i, 1, input[i % cases]);
			cta->setRegAsU16(i, 0, 0);
		}

		cta->eval_Cvt(cta->getActiveContext(), ins);

		for (int i = 0; i < threadCount; ++i) {
			PTXU16 got = cta->getRegAsU16(i, 0);
			if (got != expected[i % cases]) {
				status << "cvt.rn.bf16.f32 failed (thread " << i
					<< "): expected 0x" << hex << expected[i % cases]
					<< ", got 0x" << got << dec << "\n";
				result = false;
				break;
			}
		}

		if (result) {
			ins.modifier = PTXInstruction::rz;
			try {
				cta->eval_Cvt(cta->getActiveContext(), ins);
				status << "cvt.rz.bf16.f32 should not be implemented\n";
				result = false;
			} catch (RuntimeException &) {
				// Expected: only .rn is implemented.
			}
		}

		if (result) {
			// cvt.f32.bf16
			ins.type = PTXOperand::f32;
			ins.modifier = 0;
			ins.d = reg("d", PTXOperand::f32, 0);
			ins.a = reg("a", PTXOperand::bf16, 1);

			cta->setRegAsU16(0, 1, 0xc020);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsU32(0, 0) != 0xc0200000) {
				status << "cvt.f32.bf16 failed\n";
				result = false;
			}
		}

		if (result) {
			// cvt.rn.f16.f32
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::f32, 1);

			cta->setRegAsF32(0, 1, 2049.0f);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsU16(0, 0) != 0x6800) {
				status << "cvt.rn.f16.f32 failed\n";
				result = false;
			}
		}

		if (result) {
			// cvt.f32.f16
			ins.type = PTXOperand::f32;
			ins.modifier = 0;
			ins.d = reg("d", PTXOperand::f32, 0);
			ins.a = reg("a", PTXOperand::f16, 1);

			cta->setRegAsU16(0, 1, 0x3c00);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsU32(0, 0) != 0x3f800000) {
				status << "cvt.f32.f16 failed\n";
				result = false;
			}
		}

		if (result) {
			// cvt.rzi.s32.f16
			ins.type = PTXOperand::s32;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::s32, 0);
			ins.a = reg("a", PTXOperand::f16, 1);

			cta->setRegAsU16(0, 1, 0x3e00);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsS32(0, 0) != 1) {
				status << "cvt.rzi.s32.f16 failed\n";
				result = false;
			}
		}

		if (result) {
			// cvt.rn.f16.s64
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::s64, 1);

			cta->setRegAsS64(0, 1, 2049);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsU16(0, 0) != 0x6800) {
				status << "cvt.rn.f16.s64 failed\n";
				result = false;
			}
		}

		if (result) {
			// cvt.rn.f16.f64
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::f64, 1);

			cta->setRegAsF64(0, 1, 1.0);
			cta->eval_Cvt(cta->getActiveContext(), ins);

			if (cta->getRegAsU16(0, 0) != 0x3c00) {
				status << "cvt.rn.f16.f64 failed\n";
				result = false;
			}
		}
	
		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// set, setp, selp, slct

	bool test_Set() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Set;

		cta->reset();

		if (result) {
			// set.u32.s32.ge
			// 
			ins.type = PTXOperand::u32;
			ins.d = reg("r", PTXOperand::u32, 3);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.comparisonOperator = PTXInstruction::Ge;
			ins.booleanOperator = PTXInstruction::BoolNop;
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 3, 0x55555555);
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
				cta->setRegAsPredicate(i, 0, (i % 5));
			}
			cta->eval_Set(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i), b = (i);
				bool r_t = (a >= b);
				PTXU32 r_d = (r_t ? 0xFFFFFFFF : 0x00);
				if (r_d != cta->getRegAsU32(i, 3)) {
					status << "[set.u32.s32.ge test] " << ins.toString() << "; failed on thread " << i << "\n";
					result = false; break;
				}
			}
		}
		if (result) {
			// set.f32.s32.ge.and
			// 
			ins.type = PTXOperand::f32;
			ins.d = reg("r", PTXOperand::f32, 3);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.c = reg("c", PTXOperand::pred, 0);
			ins.comparisonOperator = PTXInstruction::Ge;
			ins.booleanOperator = PTXInstruction::BoolAnd;
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 3, 0x55555555);
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
				cta->setRegAsPredicate(i, 0, (i % 3));
			}
			cta->eval_Set(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i), b = (i);
				bool r_t = (a >= b) && (bool)(i % 3);
				PTXF32 r_d = (r_t ? 1.0f : 0.0f);
				if (r_d != cta->getRegAsF32(i, 3)) {
					status << "[set.f32.s32.and test] " << ins.toString() << "; failed on thread " << i << "\n";
					result = false; break;
				}
			}
		}

		if (result) {
			// set.u32.f32.lt.or
			// 
			ins.type = PTXOperand::u32;
			ins.d = reg("r", PTXOperand::u32, 3);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			ins.c = reg("c", PTXOperand::pred, 0);
			ins.comparisonOperator = PTXInstruction::Lt;
			ins.booleanOperator = PTXInstruction::BoolOr;
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 3, 0x55555555);
				cta->setRegAsF32(i, 1, (PTXF32)(4 - i));
				cta->setRegAsF32(i, 2, (PTXF32)i);
				cta->setRegAsPredicate(i, 0, (i % 5));
			}
			cta->eval_Set(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i), b = (i);
				bool r_t = (a < b) || (bool)(i % 5);
				PTXU32 r_d = (r_t ? 0xFFFFFFFF : 0x00);
				if (r_d != cta->getRegAsU32(i, 3)) {
					status << "[set.u32.f32.lt.or test] " << ins.toString() 
						<< "; failed on thread " << i << "\n";
					result = false; break;
				}
			}
		}

		if (result) {
			// set.eq.f16.f16.and
			ins = PTXInstruction();
			ins.opcode = PTXInstruction::Set;
			ins.type = PTXOperand::f16;
			ins.d = reg("d", PTXOperand::b16, 3);
			ins.a = reg("a", PTXOperand::f16, 1);
			ins.b = reg("b", PTXOperand::f16, 2);
			ins.c = reg("c", PTXOperand::pred, 0);
			ins.comparisonOperator = PTXInstruction::Eq;
			ins.booleanOperator = PTXInstruction::BoolAnd;

			cta->setRegAsU16(0, 1, 0x3c00); // 1.0
			cta->setRegAsU16(0, 2, 0x3c00); // 1.0
			cta->setRegAsPredicate(0, 0, true);
			cta->eval_Set(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 3) != 0x3c00) {
				status << "[set.eq.f16.f16.and test] failed\n";
				result = false;
			}

			cta->setRegAsPredicate(0, 0, false);
			cta->eval_Set(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 3) != 0x0000) {
				status << "[set.eq.f16.f16.and false predicate test] failed\n";
				result = false;
			}
		}

		return result;
	}

	bool test_SetP() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::SetP;

		cta->reset();

		if (result) {
			// setp.s32.lt p|q, a, b; // p = (a < b); q = !(a < b);
			//
			ins.type = PTXOperand::s32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.comparisonOperator = PTXInstruction::Lt;

			if (ins.toString() != "setp.lt.s32 p|q, a, b") {
				status << "test_SetP - lt instruction printed as: " << ins.toString() << "\n";
				result = false;
			}
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (((4-i) < i && cta->getRegAsPredicate(i, 3) && !cta->getRegAsPredicate(i, 4))) {
					// good
				}
				else if ( !((4-i)<i) && !cta->getRegAsPredicate(i, 3) && cta->getRegAsPredicate(i, 4)) {
					// good
				}
				else {
					status << "[s32 Lt test] " << ins.toString() << " failed - thread " << i 
						<< ", (" << cta->getRegAsS32(i, 1) << " < " << cta->getRegAsS32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}

		if (result) {
			// greater than or equal operator
			//
			ins.type = PTXOperand::s32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.comparisonOperator = PTXInstruction::Ge;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				if (((4-i) >= i && cta->getRegAsPredicate(i, 3) && !cta->getRegAsPredicate(i, 4))) {
					// good
				}
				else if ( !((4-i) >= i) && !cta->getRegAsPredicate(i, 3) && cta->getRegAsPredicate(i, 4)) {
					// good
				}
				else {
					status << "[s32 Ge test] " << ins.toString() << "; failed - thread " << i 
						<< ", (" << cta->getRegAsS32(i, 1) << " >= " << cta->getRegAsS32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}

		if (result) {
			// floating-point operators
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			ins.comparisonOperator = PTXInstruction::Gt;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 1, 2.25f * i);
				cta->setRegAsF32(i, 2, 3.14f - 1.1f * (float)i);
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				float a = 2.25f * i;
				float b = 3.14f - 1.1f * (float)i;
				bool r_p = (a > b), r_q = !(a > b);

				if (cta->getRegAsPredicate(i, 3) == r_p && cta->getRegAsPredicate(i, 4) == r_q) {
					// good
				}
				else {
					status << "[f32 Gt test] " << ins.toString() << "; failed - thread " << i 
						<< ", (" << cta->getRegAsF32(i, 1) << " > " << cta->getRegAsF32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}
		if (result) {
			// floating-point operators
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			ins.comparisonOperator = PTXInstruction::Le;

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 1, 2.25f * i);
				cta->setRegAsF32(i, 2, 3.14f - 1.1f * (float)i);
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				float a = 2.25f * i;
				float b = 3.14f - 1.1f * (float)i;
				bool r_p = (a <= b), r_q = !(a <= b);

				if (cta->getRegAsPredicate(i, 3) == r_p && cta->getRegAsPredicate(i, 4) == r_q) {
					// good
				}
				else {
					status << "[f32 Le test] " << ins.toString() << "; failed - thread " << i 
						<< ", (" << cta->getRegAsF32(i, 1) << " <= " << cta->getRegAsF32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}

		if (result) {
			// setp.s32.and.lt p|q, a, b; // p = (a < b); q = !(a < b);
			//
			ins.type = PTXOperand::s32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.c = reg("c", PTXOperand::pred, 0);
			ins.comparisonOperator = PTXInstruction::Lt;
			ins.booleanOperator = PTXInstruction::BoolAnd;
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
				cta->setRegAsPredicate(i, 0, (i % 2));
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i);
				PTXS32 b = (i);
				bool r_c = (bool)(i % 2);
				bool r_p = (a < b) && r_c, r_q = !(a < b) && r_c; 


				if (r_p == cta->getRegAsPredicate(i, 3) && r_q == cta->getRegAsPredicate(i, 4)) {
					// good
				}
				else {
					status << "[s32 Lt+And test] " << ins.toString() << " failed - thread " << i 
						<< ", (" << cta->getRegAsS32(i, 1) << " < " << cta->getRegAsS32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}
		if (result) {
			// setp.s32.or.ge p|q, a, b; // p = (a < b); q = !(a < b);
			//
			ins.type = PTXOperand::s32;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.c = reg("c", PTXOperand::pred, 0);
			ins.comparisonOperator = PTXInstruction::Ge;
			ins.booleanOperator = PTXInstruction::BoolOr;
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
				cta->setRegAsPredicate(i, 0, (i % 5));
			}
			cta->eval_SetP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i);
				PTXS32 b = (i);
				bool r_c = (bool)(i % 5);
				bool r_p = (a >= b) || r_c, r_q = !(a >= b) || r_c; 

				if (r_p == cta->getRegAsPredicate(i, 3) && r_q == cta->getRegAsPredicate(i, 4)) {
					// good
				}
				else {
					status << "[s32 Ge+Or test] " << ins.toString() << " failed - thread " << i 
						<< ", (" << cta->getRegAsS32(i, 1) << " >= " << cta->getRegAsS32(i, 2) << ") ?? - "
						<< " p = " << cta->getRegAsPredicate(i, 3) << ", q = " << cta->getRegAsPredicate(i, 4) << "\n";
					result = false;
					break;
				}
			}
		}

		if (result) {
			// Unordered floating-point comparisons are true when an input is NaN.
			ins = PTXInstruction();
			ins.opcode = PTXInstruction::SetP;
			ins.type = PTXOperand::f64;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::f64, 1);
			ins.b = reg("b", PTXOperand::f64, 2);
			ins.comparisonOperator = PTXInstruction::Equ;

			cta->setRegAsF64(0, 1,
				std::numeric_limits<PTXF64>::quiet_NaN());
			cta->setRegAsF64(0, 2, 1.0);
			cta->eval_SetP(cta->getActiveContext(), ins);

			if (!cta->getRegAsPredicate(0, 3) ||
				cta->getRegAsPredicate(0, 4)) {
				status << "[f64 Equ NaN test] " << ins.toString()
					<< " failed\n";
				result = false;
			}
		}

		if (result) {
			// Half inputs are widened exactly, with FTZ applied before widening.
			ins = PTXInstruction();
			ins.opcode = PTXInstruction::SetP;
			ins.type = PTXOperand::f16;
			ins.d = reg("p", PTXOperand::pred, 3);
			ins.pq = reg("q", PTXOperand::pred, 4);
			ins.a = reg("a", PTXOperand::b16, 1);
			ins.b = reg("b", PTXOperand::b16, 2);
			ins.comparisonOperator = PTXInstruction::Lt;

			cta->setRegAsU16(0, 1, 0x3c00); // 1.0
			cta->setRegAsU16(0, 2, 0x4000); // 2.0
			cta->eval_SetP(cta->getActiveContext(), ins);
			const bool normal = cta->getRegAsPredicate(0, 3) &&
				!cta->getRegAsPredicate(0, 4);

			ins.comparisonOperator = PTXInstruction::Eq;
			cta->setRegAsU16(0, 1, 0x0001); // minimum half subnormal
			cta->setRegAsU16(0, 2, 0x0000);
			cta->eval_SetP(cta->getActiveContext(), ins);
			const bool preserved = !cta->getRegAsPredicate(0, 3) &&
				cta->getRegAsPredicate(0, 4);

			ins.modifier = PTXInstruction::ftz;
			cta->eval_SetP(cta->getActiveContext(), ins);
			const bool flushed = cta->getRegAsPredicate(0, 3) &&
				!cta->getRegAsPredicate(0, 4);

			if (!normal || !preserved || !flushed) {
				status << "[f16 widening/FTZ test] failed\n";
				result = false;
			}
		}

		return result;
	}

	/*!
		selp.type d, a, b, c;

		.type = { .b16, .b32, .b64,
				      .u16, .u32, .u64,
				      .s16, .s32, .s64,
				            .f32, .f64 };

	*/
	bool test_SelP() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::SelP;

		cta->reset();

		if (result) {
			// selp.s32 r4, a, b, p
			//
			ins.type = PTXOperand::s32;
			ins.d = reg("r4", PTXOperand::s32, 3);
			ins.a = reg("a", PTXOperand::s32, 1);
			ins.b = reg("b", PTXOperand::s32, 2);
			ins.c = reg("p", PTXOperand::pred, 0);
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 1, 4 - i);
				cta->setRegAsS32(i, 2, i);
				cta->setRegAsPredicate(i, 0, (i % 3));
			}
			cta->eval_SelP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (4 - i);
				PTXS32 b = (i);
				bool r_c = (bool)(i % 3);
				PTXS32 r_d = (r_c ? a : b);

				if (r_d == cta->getRegAsS32(i, 3)) {
					// good
				}
				else {
					status << "[s32 selp test] " << ins.toString() << "; thread " << i << " failed: ";
					status << " a = " << cta->getRegAsS32(i, 1) 
						<< ", b = " << cta->getRegAsS32(i, 2) 
						<< ", d = " << cta->getRegAsS32(i, 3) << ", expected d = " << r_d << "\n";
					result = false;
					break;
				}
			}
		}

		if (result) {
			// selp.f32 r4, a, b, p
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("r4", PTXOperand::f32, 3);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			ins.c = reg("p", PTXOperand::pred, 0);
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 1, (PTXF32)(4 - i));
				cta->setRegAsF32(i, 2, (PTXF32)i);
				cta->setRegAsPredicate(i, 0, (i % 3));
			}
			cta->eval_SelP(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 a = (PTXF32)(4 - i);
				PTXF32 b = (PTXF32)(i);
				bool r_c = (bool)(i % 3);
				PTXF32 r_d = (r_c ? a : b);

				if (r_d == cta->getRegAsF32(i, 3)) {
					// good
				}
				else {
					status << "[s32 selp test] " << ins.toString() << "; thread " << i << " failed: ";
					status << " a = " << cta->getRegAsF32(i, 1) 
						<< ", b = " << cta->getRegAsF32(i, 2) 
						<< ", d = " << cta->getRegAsF32(i, 3) << ", expected d = " << r_d << "\n";
					result = false;
					break;
				}
			}
		}
	
		return result;
	}

	bool test_SlCt() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::SlCt;

		cta->reset();

		if (result) {
			// slct.f32.f32 r, a, b, c
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("r", PTXOperand::f32, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			ins.c = reg("c", PTXOperand::f32, 3);
	
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 1, (PTXF32)(4 - i) * 2.0f);
				cta->setRegAsF32(i, 2, (PTXF32)i * 3.0f);
				cta->setRegAsF32(i, 3, (PTXF32)(4 - i));
				cta->setRegAsF32(i, 0, (PTXF32)0);

			}
			cta->eval_SlCt(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 r_a = (PTXF32)(4 - i) * 2.0f;
				PTXF32 r_b = (PTXF32)i * 3.0f;
				PTXF32 r_c = (PTXF32)(4 - i);

				PTXF32 d = cta->getRegAsF32(i, 0);
				if (r_c >= 0 && r_a != d) {
					status << "[slct.f32.f32 test] " << ins.toString() << "; failed on thread " << i << "\n";
					result = false;
					break;
				}
				else if (r_c < 0 && r_b != d) {
					status << "[slct.f32.f32 test] " << ins.toString() << "; failed on thread " << i << "\n";
					result = false;
					break;
				}
			}
		}
	
		return result;
	}
	
	bool test_TestP() {
		bool result = false;
		/*
		PTXInstruction ins;
		ins.opcode = PTXInstruction::TestP;

		cta->reset();

		// f32
		//
		if (result) {
			// testp.op.type p, a
			//
			//	op: .finite, .infinite, .number, .notanumber, .normal, .subnormal
			//	type: .f32, .f64
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("p", PTXOperand::pred, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
	
			ir::PTXInstruction::FloatingPointMode floatModes[] = {
				ir::PTXInstruction::Finite,
				ir::PTXInstruction::Infinite,
				ir::PTXInstruction::Number,
				ir::PTXInstruction::NotANumber,
				ir::PTXInstruction::Normal,
				ir::PTXInstruction::SubNormal,
				ir::PTXInstruction::FloatingPointMode_Invalid
			};
			
			PTXF32 floatValues[] = {
				-1, 0, 1, FLT_EPSILON, -FLT_EPSILON, 0
			};
			
			for (int mode = 0; floatModes[mode] != ir::PTXInstruction::FloatingPointMode_Invalid; mode++) {
				ins.opcode = PTXInstruction::TestP;
				ins.floatingPointMode = floatModes[mode];
				ins.d = reg("p", PTXOperand::pred, 0);
				ins.a = reg("a", PTXOperand::f32, 1);
				
				
				
			}
			
		}
		
		// f64
		//
		if (result) {
			// testp.op.type p, a
			//
			//	op: .finite, .infinite, .number, .notanumber, .normal, .subnormal
			//	type: .f32, .f64
			//
			ins.type = PTXOperand::f32;
			ins.d = reg("p", PTXOperand::pred, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
	
			
		}
		*/
		return result;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////

	bool test_Pred_Add() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Add;

		cta->reset();

		ins.a = reg("a", PTXOperand::s32, 0);
		ins.b = reg("b", PTXOperand::s32, 1);
		ins.d = reg("d", PTXOperand::s32, 2);

		for (int i = 0; i < threadCount; i++) {
			cta->getActiveContext().active[i] = ((i % 2) ? false : true);
			cta->setRegAsS32(i, 0, i);
			cta->setRegAsS32(i, 1, 2*i);
			cta->setRegAsS32(i, 2, -1);
		}
		cta->eval_Add(cta->getActiveContext(), ins);

		for (int i = 0; i < threadCount; i++) {
			PTXS32 got = cta->getRegAsS32(i, ins.d.reg);
			if (i % 2) {
				if (got != -1) {
					status << "test_Pred_Add - error on thread " << i << ": expected -1, got " <<
						got << "\n";
				}
			}
			else {
				PTXS32 e = i + 2*i;
				if (got != e) {
					status << "test_Pred_Add - error on thread " << i << ": expected "
						 << e << ", got " << got << "\n";
				}
			}
		}

		return result;
	}

	bool test_Pred_Ld() {

		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;

		cta->reset();

		//
		// Global memory
		//

		ins.addressSpace = PTXInstruction::Global;

		// register indirect
		if (result) {
			PTXU32 source[2] = { 0xaa551376 };
			ins.d = reg("rd", PTXOperand::u32, 5);
			ins.a = reg("ra", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.type = PTXOperand::u32;

			for (int i = 0; i < threadCount; i++) {
				cta->getActiveContext().active[i] = ((i % 2) ? false : true);
				cta->setRegAsU64(i, 0, (PTXU64)&source[0]);
				cta->setRegAsU32(i, 5, 0);
			}

			cta->eval_Ld(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 got = cta->getRegAsU32(i, 5);
				if (i % 2) {
					if (got != 0) {
						status << "test_Pred_ld - ld.u32.global failed - [" << i 
							<< "] - expected 0, got " << got << "\n";
					}
				}
				else {
					if (got != 0xaa551376) {
						result = false;
						status << "test_Pred_ld - ld.u32.global failed - [" << i 
							<< "] - expected 0xaa551376, got " << got << "\n";
					}
				}
			}
		}		

		return result;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////

	/*!
		Test driver
	*/
	bool doTest() {
	
		if(!valid) {
			return false;
		}
	
		bool result = testRegisterAccessors();
		bool prolix = true;

		try {
			// ld, store instructions
			result = (result && test_Ld());
			result = (result && test_St());
			if (prolix && result) {
				status << "pass: load and store instructions\n";
			}

			// mov instruction
			result = (result && test_Mov());

			// cvt instruction
			result = (result && test_Cvt());
	
			// arithmetic instructions
			result = (result && test_Abs());
			result = (result && test_Add());
			result = (result && test_Sub());
			result = (result && test_Div());
			result = (result && test_Neg());
			result = (result && test_Rem());
			result = (result && test_Min());
			result = (result && test_Max());
			if (prolix && result) {
				status << "pass: arithmetic instructions\n";
			}

			// difficult arithmetic instructions
			result = (result && test_Mad());
			result = (result && test_Mul());
			result = (result && test_AddC());
			result = (result && test_SubC());
			if (prolix && result) {
				status << "pass: exotic arithmetic instructions\n";
			}

			// floating-point instructions
			result = (result && test_Cos());
			result = (result && test_Sin());
			result = (result && test_Ex2());
			result = (result && test_F16Fma());
			result = (result && test_Bf16Fma());
			result = (result && test_Mma());
			result = (result && test_Lg2());
			result = (result && test_Sqrt());
			result = (result && test_Rsqrt());
			result = (result && test_Rcp());
			if (prolix && result) {
				status << "pass: floating-point instructions\n";
			}

			// logical and shift instructions
			result = (result && test_And());
			result = (result && test_Or());
			result = (result && test_Xor());
			result = (result && test_Not());
			if (prolix && result) {
				status << "pass: logical instructions\n";
			}

			// predicate and comparison operators
			result = (result && test_Set());
			result = (result && test_SetP());
			result = (result && test_SelP());
			result = (result && test_SlCt());
			if (prolix && result) {
				status << "pass: comparison instructions\n";
			}

			// test predication of various instructions
			result = (result && test_Pred_Add());
			result = (result && test_Pred_Ld());
			if (prolix && result) {
				status << "pass: predicated Add and Ld isntructions\n";
			}


			// if you made it here, the instruction-level tests have succeeded
		}
		catch (RuntimeException &exp) {
			status << "unhandled exception: " << exp.message 
				<< "\non instruction " << exp.instruction.toString() << "\n\n";
			result = false;
		}

		return result;
	}

private:

};

}

int main(int argc, char **argv) {
	using namespace std;
	using namespace ir;
	using namespace test;

	hydrazine::ArgumentParser parser( argc, argv );
	test::TestInstructions test;

	parser.description( test.testDescription() );

	parser.parse( "-s", test.seed, 0,
		"Set the random seed, 0 implies seed with time." );
	parser.parse( "-v", test.verbose, false, "Print out info after the test." );
	parser.parse();

	test.test();

	return test.passed();
}
