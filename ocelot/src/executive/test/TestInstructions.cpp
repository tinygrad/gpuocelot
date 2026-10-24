/*! \file TestInstructions.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief unit tests for each instruction
*/

#include <sstream>
#include <fstream>

#include <hydrazine/Test.h>

#include <hydrazine/ArgumentParser.h>
#include <hydrazine/Casts.h>
#include <hydrazine/Exception.h>
#include <hydrazine/FloatingPoint.h>
#include <hydrazine/macros.h>
#include <hydrazine/debug.h>

#include <ocelot/ir/Module.h>
#include <ocelot/analysis/DataflowGraph.h>
#include <ocelot/ir/ControlFlowGraph.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>

#include <cmath>
#include <limits>

using namespace std;
using namespace ir;
using namespace executive;

namespace test {

class ConstMemoryTestKernel : public EmulatedKernel {
public:
	ConstMemoryTestKernel(ir::IRKernel* kernel)
		: EmulatedKernel(kernel, 0, false) {}
	void setConstMemory(unsigned int size) {
		delete[] ConstMemory;
		ConstMemory = new char[size];
		_constMemorySize = size;
	}
};

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

		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_abs() { .reg .b16 d16, a16; "
			<< ".reg .b32 d32, a32; abs.f16 d16, a16; "
			<< "abs.ftz.f16x2 d32, a32; abs.bf16 d16, a16; "
			<< "abs.bf16x2 d32, a32; ret; }\n";
		try { Module parsed; parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "half abs parse failed: " << error.what() << "\n";
			return false;
		}
		auto abs16 = [&](PTXOperand::DataType type, int modifier,
			PTXU16 a, PTXU16 expected) {
			ins.opcode = PTXInstruction::Abs; ins.type = type; ins.modifier = modifier;
			ins.a = reg("a", PTXOperand::b16, 1); ins.d = reg("d", PTXOperand::b16, 0);
			cta->setRegAsU16(0, 1, a); cta->eval_Abs(cta->getActiveContext(), ins);
			return cta->getRegAsU16(0, 0) == expected;
		};
		result = result && abs16(PTXOperand::f16, 0, 0xbe00, 0x3e00);
		result = result && abs16(PTXOperand::f16, 0, 0x8001, 0x0001);
		result = result && abs16(PTXOperand::f16, PTXInstruction::ftz, 0x8001, 0);
		result = result && abs16(PTXOperand::bf16, 0, 0xbfc0, 0x3fc0);
		result = result && abs16(PTXOperand::bf16, 0, 0x8001, 0x0001);
		auto absPacked = [&](PTXOperand::DataType type, int modifier,
			PTXU32 a, PTXU32 expected) {
			ins.type = type; ins.modifier = modifier;
			ins.a = reg("a", PTXOperand::b32, 1); ins.d = reg("d", PTXOperand::b32, 0);
			cta->setRegAsU32(0, 1, a); cta->eval_Abs(cta->getActiveContext(), ins);
			return cta->getRegAsU32(0, 0) == expected;
		};
		result = result && absPacked(PTXOperand::f16x2, 0, 0xbe003e00, 0x3e003e00);
		result = result && absPacked(PTXOperand::f16x2, 0, 0x80018001, 0x00010001);
		result = result && absPacked(PTXOperand::f16x2, PTXInstruction::ftz, 0x80018001, 0);
		result = result && absPacked(PTXOperand::bf16x2, 0, 0xbfc03fc0, 0x3fc03fc0);
		result = result && absPacked(PTXOperand::bf16x2, 0, 0x80010000, 0x00010000);
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
		if (result) {
			ins.modifier = 0; // PTX defaults to round-to-nearest-even.
			cta->setRegAsU16(0, 0, 0x3c00); // 1.0
			cta->setRegAsU16(0, 1, 0x1000); // half an f16 ULP at 1.0
			const int previous = hydrazine::fegetround();
			hydrazine::fesetround(FE_UPWARD);
			cta->eval_Add(cta->getActiveContext(), ins);
			const bool roundedNearest = cta->getRegAsU16(0, 2) == 0x3c00;
			const bool restored = hydrazine::fegetround() == FE_UPWARD;
			hydrazine::fesetround(previous);
			if (!roundedNearest || !restored) {
				status << "add.f16 default rounding failed\n";
				result = false;
			}
		}
		ins.modifier = PTXInstruction::rz;
		if (ins.valid().empty()) {
			status << "add.rz.f16 accepted\n";
			result = false;
		}
		ins.modifier = PTXInstruction::ftz | PTXInstruction::sat;
		if (!ins.valid().empty()) {
			status << "add.ftz.sat.f16 rejected\n";
			result = false;
		}

		// f16x2
		//
		if (result) {
			ins.type = PTXOperand::f16x2;
			ins.modifier = 0;
			ins.a = reg("r1", PTXOperand::b32, 0);
			ins.b = reg("r2", PTXOperand::b32, 1);
			ins.d = reg("r3", PTXOperand::b32, 2);
			if (!ins.valid().empty()) result = false;
			ins.modifier = PTXInstruction::rp;
			if (ins.valid().empty()) {
				status << "add.rp.f16x2 accepted\n";
				result = false;
			}
			cta->reset();
			auto packedAdd = [&](int modifier, PTXU32 a, PTXU32 b,
				PTXU32 expected, bool alias) {
				ins.modifier = modifier;
				ins.d.reg = alias ? 0 : 2;
				cta->setRegAsU32(0, 0, a);
				cta->setRegAsU32(0, 1, b);
				cta->setRegAsU32(0, 2, 0xdeadbeef);
				cta->eval_Add(cta->getActiveContext(), ins);
				return cta->getRegAsU32(0, ins.d.reg) == expected;
			};
			result = result && packedAdd(0, 0x40003c00, 0x3c004000, 0x42004200, false);
			result = result && packedAdd(0, 0x3c013c00, 0x10001000, 0x3c023c00, false);
			result = result && packedAdd(PTXInstruction::rn, 0x3c013c00,
				0x10001000, 0x3c023c00, false);
			result = result && packedAdd(PTXInstruction::ftz, 0x80010001,
				0x80000000, 0x80000000, false);
			result = result && packedAdd(PTXInstruction::sat, 0x7e004000,
				0x00000000, 0x00003c00, false);
			result = result && packedAdd(0, 0x40003c00, 0x3c004000,
				0x42004200, true);
			if (result) {
				const int previous = hydrazine::fegetround();
				hydrazine::fesetround(FE_UPWARD);
				const bool defaultRn = packedAdd(0, 0x3c013c00,
					0x10001000, 0x3c023c00, false);
				const bool restoredUpward = hydrazine::fegetround() == FE_UPWARD;
				hydrazine::fesetround(FE_DOWNWARD);
				const bool cancellation = packedAdd(0, 0xbc003c00,
					0x3c00bc00, 0, false);
				const bool restoredDownward = hydrazine::fegetround() == FE_DOWNWARD;
				hydrazine::fesetround(previous);
				if (!defaultRn || !cancellation || !restoredUpward || !restoredDownward) result = false;
			}
			ins.modifier = 0;
			ins.d.reg = 2;
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsU32(0, 2, 0xcafebabe);
			cta->eval_Add(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 2) != 0xcafebabe) result = false;
			ins.pg.condition = PTXOperand::PT;
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
		if (result) {
			ins.type = PTXOperand::f16;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.b = reg("r2", PTXOperand::b16, 1);
			ins.d = reg("r3", PTXOperand::b16, 2);
			ins.modifier = 0; // PTX defaults to round-to-nearest-even.
			cta->setRegAsU16(0, 0, 0x3c00); // 1.0
			cta->setRegAsU16(0, 1, 0x9000); // -(half an f16 ULP at 1.0)
			const int previous = hydrazine::fegetround();
			hydrazine::fesetround(FE_UPWARD);
			cta->eval_Sub(cta->getActiveContext(), ins);
			const bool roundedNearest = cta->getRegAsU16(0, 2) == 0x3c00;
			const bool restored = hydrazine::fegetround() == FE_UPWARD;
			hydrazine::fesetround(previous);
			if (!roundedNearest || !restored) {
				status << "sub.f16 default rounding failed\n";
				result = false;
			}
		}
		ins.modifier = PTXInstruction::rz;
		if (ins.valid().empty()) {
			status << "sub.rz.f16 accepted\n";
			result = false;
		}
		ins.modifier = PTXInstruction::ftz | PTXInstruction::sat;
		if (!ins.valid().empty()) {
			status << "sub.ftz.sat.f16 rejected\n";
			result = false;
		}

		// f16x2
		//
		if (result) {
			ins.type = PTXOperand::f16x2;
			ins.modifier = 0;
			ins.a = reg("r1", PTXOperand::b32, 0);
			ins.b = reg("r2", PTXOperand::b32, 1);
			ins.d = reg("r3", PTXOperand::b32, 2);
			if (!ins.valid().empty()) result = false;
			ins.modifier = PTXInstruction::rp;
			if (ins.valid().empty()) {
				status << "sub.rp.f16x2 accepted\n";
				result = false;
			}
			cta->reset();
			auto packedSub = [&](int modifier, PTXU32 a, PTXU32 b,
				PTXU32 expected, bool alias) {
				ins.modifier = modifier;
				ins.d.reg = alias ? 0 : 2;
				cta->setRegAsU32(0, 0, a);
				cta->setRegAsU32(0, 1, b);
				cta->setRegAsU32(0, 2, 0xdeadbeef);
				cta->eval_Sub(cta->getActiveContext(), ins);
				return cta->getRegAsU32(0, ins.d.reg) == expected;
			};
			result = result && packedSub(0, 0x3e004000, 0x40003e00, 0xb8003800, false);
			result = result && packedSub(0, 0x3c013c00, 0x90009000, 0x3c023c00, false);
			result = result && packedSub(PTXInstruction::rn, 0x3c013c00,
				0x90009000, 0x3c023c00, false);
			result = result && packedSub(PTXInstruction::ftz, 0x80010001,
				0, 0x80000000, false);
			result = result && packedSub(PTXInstruction::ftz, 0x84010401,
				0x84000400, 0x80000000, false);
			result = result && packedSub(PTXInstruction::sat, 0x40000000,
				0x003c00, 0x3c000000, false);
			result = result && packedSub(PTXInstruction::sat, 0x00007e00,
				0, 0, false);
			result = result && packedSub(0, 0x3e004000, 0x40003e00,
				0xb8003800, true);
			if (result) {
				const int previous = hydrazine::fegetround();
				hydrazine::fesetround(FE_UPWARD);
				const bool defaultRn = packedSub(0, 0x3c013c00,
					0x90009000, 0x3c023c00, false);
				const bool restoredUpward = hydrazine::fegetround() == FE_UPWARD;
				hydrazine::fesetround(FE_DOWNWARD);
				const bool cancellation = packedSub(0, 0xbc003c00,
					0xbc003c00, 0, false);
				const bool restoredDownward = hydrazine::fegetround() == FE_DOWNWARD;
				hydrazine::fesetround(previous);
				if (!defaultRn || !cancellation || !restoredUpward || !restoredDownward)
					result = false;
			}
			ins.modifier = 0;
			ins.d.reg = 2;
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsU32(0, 2, 0xcafebabe);
			cta->eval_Sub(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 2) != 0xcafebabe) result = false;
			ins.pg.condition = PTXOperand::PT;
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

	bool test_AddSubRounding() {
		PTXInstruction ins;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.b = reg("r2", PTXOperand::f32, 1);
		ins.d = reg("r3", PTXOperand::f32, 2);
		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz, PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 add32[][2] = {{0x3f800000, 0xbf800000},
			{0x3f800000, 0xbf800000}, {0x3f800000, 0xbf800001},
			{0x3f800001, 0xbf800000}};
		const PTXU32 sub32[] = {0x3f800000, 0x3f7fffff, 0x3f7fffff, 0x3f800000};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			ins.opcode = PTXInstruction::Add;
			cta->setRegAsF32(0, 0, 1.0f);
			cta->setRegAsF32(0, 1, std::ldexp(1.0f, -24));
			cta->setRegAsF32(1, 0, -1.0f);
			cta->setRegAsF32(1, 1, -std::ldexp(1.0f, -24));
			cta->eval_Add(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != add32[i][0]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != add32[i][1]) {
				status << "add.f32 rounding failed\n";
				return false;
			}
			ins.opcode = PTXInstruction::Sub;
			cta->setRegAsF32(0, 0, 1.0f);
			cta->setRegAsF32(0, 1, std::ldexp(1.0f, -25));
			cta->eval_Sub(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != sub32[i]) {
				status << "sub.f32 rounding failed\n";
				return false;
			}
		}
		ins.type = PTXOperand::f64;
		ins.a.type = ins.b.type = ins.d.type = PTXOperand::f64;
		const PTXU64 expected64[][3] = {
			{0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3ff0000000000000ull},
			{0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3fefffffffffffffull},
			{0x3ff0000000000000ull, 0xbff0000000000001ull, 0x3fefffffffffffffull},
			{0x3ff0000000000001ull, 0xbff0000000000000ull, 0x3ff0000000000000ull}};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			ins.opcode = PTXInstruction::Add;
			cta->setRegAsF64(0, 0, 1.0);
			cta->setRegAsF64(0, 1, std::ldexp(1.0, -53));
			cta->setRegAsF64(1, 0, -1.0);
			cta->setRegAsF64(1, 1, -std::ldexp(1.0, -53));
			cta->eval_Add(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2)) != expected64[i][0]
				|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2)) != expected64[i][1]) {
				status << "add.f64 rounding failed\n";
				return false;
			}
			ins.opcode = PTXInstruction::Sub;
			cta->setRegAsF64(0, 1, std::ldexp(1.0, -54));
			cta->eval_Sub(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2)) != expected64[i][2]) {
				status << "sub.f64 rounding failed\n";
				return false;
			}
		}
		const int previous = hydrazine::fegetround();
		hydrazine::fesetround(FE_UPWARD);
		ins.modifier = 0;
		ins.opcode = PTXInstruction::Add;
		cta->setRegAsF64(0, 0, 1.0);
		cta->setRegAsF64(0, 1, std::ldexp(1.0, -53));
		cta->eval_Add(cta->getActiveContext(), ins);
		const bool defaultRn = hydrazine::bit_cast<PTXU64>(
			cta->getRegAsF64(0, 2)) == 0x3ff0000000000000ull;
		const bool restored = hydrazine::fegetround() == FE_UPWARD;
		hydrazine::fesetround(previous);
		if (!defaultRn || !restored) {
			status << "default rounding or rounding-mode restoration failed\n";
			return false;
		}
		return true;
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
				if (cta->getRegAsU16(i, 3) != expected) {
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
				if (cta->getRegAsU32(i, 3) != expected) {
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
				if (cta->getRegAsU64(i, 3) != expected) {
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
				cta->setRegAsS16(i, 0, (PTXS16)(-i * 2));
				cta->setRegAsS16(i, 1, (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 2);
				cta->setRegAsS16(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 a = (-i * 2), b = (4 + i), c = 2;
				PTXS16 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS16(i, 3) != expected) {
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
				cta->setRegAsS32(i, 0, (PTXS32)(-i * 2));
				cta->setRegAsS32(i, 1, (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 2);
				cta->setRegAsS32(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 a = (-i * 2), b = (4 + i), c = 2;
				PTXS32 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS32(i, 3) != expected) {
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
			ins.a = reg("r1", PTXOperand::s64, 0);
			ins.b = reg("r2", PTXOperand::s64, 1);
			ins.c = reg("r3", PTXOperand::s64, 2);
			ins.d = reg("r4", PTXOperand::s64, 3);

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, (PTXS64)(-i * 2));
				cta->setRegAsS64(i, 1, (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 2);
				cta->setRegAsS64(i, 3, 0);
			}
			cta->eval_Sad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 a = (-i * 2), b = (4 + i), c = 2;
				PTXS64 expected = c + ((a < b) ? b-a : a-b);
				if (cta->getRegAsS64(i, 3) != expected) {
					result = false;
					status << "sad.s64 incorrect\n";
					break;
				}
			}
		}

		return result;
	}

	bool test_LdLu() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_ld_lu() {\n"
			<< "  .reg .f32 d;\n"
			<< "  .reg .u64 a;\n"
			<< "  ld.lu.f32 d, [a];\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse ld.lu example: " << error.what() << "\n";
			return false;
		}
		bool foundLu = false;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns || ptxIns->opcode != PTXInstruction::Ld) continue;
					if (ptxIns->cacheOperation == PTXInstruction::Lu) foundLu = true;
				}
		if (!foundLu) {
			result = false;
			status << "ld.lu did not parse with cacheOperation == Lu\n";
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ld;
		ins.type = PTXOperand::f32;
		ins.addressSpace = PTXInstruction::Global;
		ins.cacheOperation = PTXInstruction::Lu;
		ins.volatility = PTXInstruction::Volatile;
		ins.a = reg("a", PTXOperand::u64, 0);
		ins.a.addressMode = PTXOperand::Indirect;
		ins.d = reg("d", PTXOperand::f32, 1);
		if (ins.valid().empty()) {
			result = false;
			status << "ld.lu combined with .volatile was incorrectly accepted\n";
		}

		return result;
	}

	// suld/sust only support .cop = {.ca, .cg, .cs, .cv} per PTX ISA; the
	// grammar's cacheOperation production is shared with ld, which also
	// accepts .nc and .lu, so opcode-specific validation must reject those
	// two on surface ops.
	bool test_SuldSustCacheOperator() {
		bool result = true;
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Suld;
		ins.formatMode = PTXInstruction::Unformatted;
		ins.type = PTXOperand::b32;
		if (!ins.valid().empty()) {
			result = false;
			status << "suld with default cache operator was incorrectly rejected\n";
		}
		ins.cacheOperation = PTXInstruction::Cv;
		if (!ins.valid().empty()) {
			result = false;
			status << "suld.cv was incorrectly rejected\n";
		}
		ins.cacheOperation = PTXInstruction::Nc;
		if (ins.valid().empty()) {
			result = false;
			status << "suld.nc was incorrectly accepted\n";
		}
		ins.cacheOperation = PTXInstruction::Lu;
		if (ins.valid().empty()) {
			result = false;
			status << "suld.lu was incorrectly accepted\n";
		}

		ins.opcode = PTXInstruction::Sust;
		ins.cacheOperation = PTXInstruction::Cs;
		if (!ins.valid().empty()) {
			result = false;
			status << "sust.cs was incorrectly rejected\n";
		}
		ins.cacheOperation = PTXInstruction::Nc;
		if (ins.valid().empty()) {
			result = false;
			status << "sust.nc was incorrectly accepted\n";
		}
		ins.cacheOperation = PTXInstruction::Lu;
		if (ins.valid().empty()) {
			result = false;
			status << "sust.lu was incorrectly accepted\n";
		}

		return result;
	}

	// st only supports .cop = {.wb, .cg, .cs, .wt}; it previously shared
	// ld's grammar production, which wrongly accepted .ca/.cv/.nc/.lu and
	// couldn't parse .wb/.wt at all (tokens declared but never wired into
	// any rule). st now has its own storeCacheOperation production.
	bool test_StCacheOperator() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_st_cop() {\n"
			<< "  .reg .f32 d;\n"
			<< "  .reg .u64 a;\n"
			<< "  st.wb.f32 [a], d;\n"
			<< "  st.wt.f32 [a], d;\n"
			<< "  st.cg.f32 [a], d;\n"
			<< "  st.cs.f32 [a], d;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse st.wb/.wt/.cg/.cs example: " << error.what() << "\n";
			return false;
		}
		unsigned int found = 0;
		const PTXInstruction::CacheOperation expected[] = {
			PTXInstruction::Wb, PTXInstruction::Wt,
			PTXInstruction::Cg, PTXInstruction::Cs};
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns || ptxIns->opcode != PTXInstruction::St) continue;
					if (found < 4 && ptxIns->cacheOperation == expected[found]) ++found;
				}
		if (found != 4) {
			status << "st.wb/.wt/.cg/.cs did not all parse with the expected cache operator\n";
			return false;
		}

		std::stringstream illegal;
		illegal << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_st_cop_illegal() {\n"
			<< "  .reg .f32 d;\n"
			<< "  .reg .u64 a;\n"
			<< "  st.ca.f32 [a], d;\n"
			<< "  ret;\n}\n";
		Module illegalModule;
		try {
			illegalModule.load(illegal);
			status << "st.ca was incorrectly accepted by the parser\n";
			return false;
		}
		catch (const std::exception&) {
			// expected: .ca is not a legal st cache operator
		}

		return true;
	}

	// Bison only attaches a rule's trailing { action } to its LAST
	// alternative, not all of them (rule : A | B | C { action } only runs
	// action for C). Several grammar productions shared one action across
	// multiple alternatives, so only the last-listed modifier actually got
	// recorded; every other modifier silently kept the instruction's
	// previous/default field value. This checks the first-listed
	// (previously-broken) alternative of each affected production parses
	// with the correct field, not just the last one.
	bool test_GrammarSharedActionFix() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_shared_action() {\n"
			<< "  .reg .f32 d;\n"
			<< "  .reg .u64 a;\n"
			<< "  .reg .pred p, q;\n"
			<< "  ld.cg.f32 d, [a];\n"
			<< "  lop3.and.b32 d|p, d, d, d, 0x3f, q;\n"
			<< "  lop3.or.b32 d|p, d, d, d, 0x3f, q;\n"
			<< "  bar.arrive 0;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse shared-action regression example: "
				<< error.what() << "\n";
			return false;
		}
		bool ldCg = false, lop3And = false, lop3Or = false, barArrive = false;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns) continue;
					if (ptxIns->opcode == PTXInstruction::Ld
						&& ptxIns->cacheOperation == PTXInstruction::Cg) ldCg = true;
					if (ptxIns->opcode == PTXInstruction::Lop3
						&& ptxIns->booleanOperator == PTXInstruction::BoolAnd) lop3And = true;
					if (ptxIns->opcode == PTXInstruction::Lop3
						&& ptxIns->booleanOperator == PTXInstruction::BoolOr) lop3Or = true;
					if (ptxIns->opcode == PTXInstruction::Bar
						&& ptxIns->barrierOperation == PTXInstruction::BarArrive) barArrive = true;
				}
		// require BOTH lop3 forms to be seen with their distinct operator,
		// not just one, so a coincidental match against uninitialized
		// memory can't make this pass for the wrong reason
		if (!ldCg || !lop3And || !lop3Or || !barArrive) {
			status << "shared-action regression: ld.cg=" << ldCg
				<< " lop3.and=" << lop3And << " lop3.or=" << lop3Or
				<< " bar.arrive=" << barArrive << "\n";
			return false;
		}

		return true;
	}

	// stringToOpcode() had no entries for "prefetch"/"prefetchu" at all,
	// so both silently fell through to Nop and failed to parse with
	// "NOP is not a valid instruction" -- unrelated to the shared-action
	// grammar bug, found while regression-testing it.
	bool test_PrefetchOpcode() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_prefetch() {\n"
			<< "  .reg .u64 a;\n"
			<< "  prefetch.global.L1 [a];\n"
			<< "  prefetchu.L1 [a];\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse prefetch/prefetchu example: "
				<< error.what() << "\n";
			return false;
		}
		bool prefetch = false, prefetchu = false;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns) continue;
					if (ptxIns->opcode == PTXInstruction::Prefetch
						&& ptxIns->cacheLevel == PTXInstruction::L1
						&& ptxIns->addressSpace == PTXInstruction::Global) prefetch = true;
					if (ptxIns->opcode == PTXInstruction::Prefetchu
						&& ptxIns->cacheLevel == PTXInstruction::L1) prefetchu = true;
				}
		if (!prefetch || !prefetchu) {
			status << "prefetch=" << prefetch << " prefetchu=" << prefetchu << "\n";
			return false;
		}

		return true;
	}

	// eval_Red used to unconditionally throw "instruction not implemented".
	// red is atom minus the writeback of the pre-update value; it shares
	// evalAtomicRMW with eval_Atom but resolves its operation from the
	// distinct reductionOperation field/enum rather than atomicOperation.
	bool test_Red() {
		PTXU32 mem = 10;
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Red;
		ins.type = PTXOperand::u32;
		ins.addressSpace = PTXInstruction::Global;
		ins.reductionOperation = PTXInstruction::ReductionAdd;
		ins.a = reg("a", PTXOperand::u64, 0);
		ins.a.addressMode = PTXOperand::Indirect;
		ins.b = imm_uint("b", PTXOperand::u32, 5);
		for (int i = 0; i < threadCount; i++) {
			cta->setRegAsU64(i, 0, (PTXU64)&mem);
		}
		cta->eval_Red(cta->getActiveContext(), ins);
		if (mem != 10 + 5u * threadCount) {
			status << "red.global.add.u32 gave " << mem << ", expected "
				<< (10 + 5u * threadCount) << "\n";
			return false;
		}

		return true;
	}

	// fence was entirely unimplemented -- no lexer token, no grammar rule,
	// no Opcode value. Verifies parsing of the basic thread-fence form
	// (sem optional, defaulting to acq_rel) and that execution is a no-op,
	// mirroring the already-implemented membar.
	bool test_Fence() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_fence() {\n"
			<< "  fence.sc.gpu;\n"
			<< "  fence.acquire.cta;\n"
			<< "  fence.sys;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse fence example: " << error.what() << "\n";
			return false;
		}
		std::vector<const PTXInstruction*> fences;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (ptxIns && ptxIns->opcode == PTXInstruction::Fence) fences.push_back(ptxIns);
				}
		if (fences.size() != 3) {
			status << "expected 3 fence instructions, found " << fences.size() << "\n";
			return false;
		}
		if (fences[0]->semantics != PTXInstruction::Sc
			|| fences[0]->level != PTXInstruction::GlobalLevel) {
			status << "fence.sc.gpu: semantics=" << fences[0]->semantics
				<< " level=" << fences[0]->level << "\n";
			return false;
		}
		if (fences[1]->semantics != PTXInstruction::Acquire
			|| fences[1]->level != PTXInstruction::CtaLevel) {
			status << "fence.acquire.cta: semantics=" << fences[1]->semantics
				<< " level=" << fences[1]->level << "\n";
			return false;
		}
		if (fences[2]->semantics != PTXInstruction::AcqRel
			|| fences[2]->level != PTXInstruction::SystemLevel) {
			status << "fence.sys (no .sem, should default to acq_rel): semantics="
				<< fences[2]->semantics << " level=" << fences[2]->level << "\n";
			return false;
		}

		// trivial no-op check
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fence;
		try { cta->eval_Fence(cta->getActiveContext(), ins); }
		catch (const std::exception& error) {
			status << "eval_Fence threw: " << error.what() << "\n";
			return false;
		}

		return true;
	}

	// atom/red previously could not parse the optional {.sem}{.scope}
	// memory-ordering prefix at all. Verifies parsing of all legal atom
	// .sem forms (relaxed/acquire/release/acq_rel, no .sc), all legal red
	// .sem forms (relaxed/release only), optional .scope, and that .sem
	// defaults to .relaxed when omitted, per PTX ISA 9.7.14.5/9.7.14.6.
	bool test_AtomicRedSemantics() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_atomic_red_sem() {\n"
			<< "  .reg .u64 a;\n"
			<< "  .reg .u32 b, d;\n"
			<< "  atom.relaxed.gpu.global.add.u32 d, [a], b;\n"
			<< "  atom.acquire.cta.global.add.u32 d, [a], b;\n"
			<< "  atom.global.add.u32 d, [a], b;\n"
			<< "  red.relaxed.global.add.u32 a, b;\n"
			<< "  red.release.gpu.global.add.u32 a, b;\n"
			<< "  red.global.add.u32 a, b;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse atom/red .sem/.scope example: "
				<< error.what() << "\n";
			return false;
		}
		std::vector<const PTXInstruction*> atoms, reds;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns) continue;
					if (ptxIns->opcode == PTXInstruction::Atom) atoms.push_back(ptxIns);
					if (ptxIns->opcode == PTXInstruction::Red) reds.push_back(ptxIns);
				}
		if (atoms.size() != 3 || reds.size() != 3) {
			status << "expected 3 atom + 3 red instructions, found "
				<< atoms.size() << " atom, " << reds.size() << " red\n";
			return false;
		}
		if (atoms[0]->semantics != PTXInstruction::Relaxed
			|| atoms[0]->scope != PTXInstruction::GlobalLevel) {
			status << "atom.relaxed.gpu: semantics=" << atoms[0]->semantics
				<< " scope=" << atoms[0]->scope << "\n";
			return false;
		}
		if (atoms[1]->semantics != PTXInstruction::Acquire
			|| atoms[1]->scope != PTXInstruction::CtaLevel) {
			status << "atom.acquire.cta: semantics=" << atoms[1]->semantics
				<< " scope=" << atoms[1]->scope << "\n";
			return false;
		}
		if (atoms[2]->semantics != PTXInstruction::Relaxed
			|| atoms[2]->scope != PTXInstruction::Level_Invalid) {
			status << "atom (no .sem/.scope, should default to relaxed): semantics="
				<< atoms[2]->semantics << " scope=" << atoms[2]->scope << "\n";
			return false;
		}
		if (atoms[2]->addressSpace != PTXInstruction::Global) {
			status << "atom (no .sem/.scope): addressSpace corrupted, got "
				<< atoms[2]->addressSpace << "\n";
			return false;
		}
		if (reds[0]->semantics != PTXInstruction::Relaxed
			|| reds[0]->scope != PTXInstruction::Level_Invalid) {
			status << "red.relaxed: semantics=" << reds[0]->semantics
				<< " scope=" << reds[0]->scope << "\n";
			return false;
		}
		if (reds[1]->semantics != PTXInstruction::Release
			|| reds[1]->scope != PTXInstruction::GlobalLevel) {
			status << "red.release.gpu: semantics=" << reds[1]->semantics
				<< " scope=" << reds[1]->scope << "\n";
			return false;
		}
		if (reds[2]->semantics != PTXInstruction::Relaxed
			|| reds[2]->scope != PTXInstruction::Level_Invalid) {
			status << "red (no .sem/.scope, should default to relaxed): semantics="
				<< reds[2]->semantics << " scope=" << reds[2]->scope << "\n";
			return false;
		}
		if (reds[2]->addressSpace != PTXInstruction::Global) {
			status << "red (no .sem/.scope): addressSpace corrupted, got "
				<< reds[2]->addressSpace << "\n";
			return false;
		}

		return true;
	}

	// ld/st .weak / .relaxed.scope / .acquire.scope / .release.scope parsing.
	bool test_LdStOrdering() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_ldst_ordering() {\n"
			<< "  .reg .f32 b, d;\n"
			<< "  .reg .u64 a;\n"
			<< "  ld.weak.global.f32 d, [a];\n"
			<< "  ld.relaxed.gpu.global.f32 d, [a];\n"
			<< "  ld.acquire.cta.global.f32 d, [a];\n"
			<< "  ld.global.f32 d, [a];\n"
			<< "  st.weak.global.f32 [a], b;\n"
			<< "  st.relaxed.gpu.global.f32 [a], b;\n"
			<< "  st.release.cta.global.f32 [a], b;\n"
			<< "  st.global.f32 [a], b;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse ld/st ordering example: "
				<< error.what() << "\n";
			return false;
		}
		std::vector<const PTXInstruction*> lds, sts;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns) continue;
					if (ptxIns->opcode == PTXInstruction::Ld) lds.push_back(ptxIns);
					if (ptxIns->opcode == PTXInstruction::St) sts.push_back(ptxIns);
				}
		if (lds.size() != 4 || sts.size() != 4) {
			status << "expected 4 ld + 4 st instructions, found "
				<< lds.size() << " ld, " << sts.size() << " st\n";
			return false;
		}
		if (lds[0]->semantics != PTXInstruction::Weak) {
			status << "ld.weak: semantics=" << lds[0]->semantics << "\n";
			return false;
		}
		if (lds[1]->semantics != PTXInstruction::Relaxed
			|| lds[1]->scope != PTXInstruction::GlobalLevel) {
			status << "ld.relaxed.gpu: semantics=" << lds[1]->semantics
				<< " scope=" << lds[1]->scope << "\n";
			return false;
		}
		if (lds[2]->semantics != PTXInstruction::Acquire
			|| lds[2]->scope != PTXInstruction::CtaLevel) {
			status << "ld.acquire.cta: semantics=" << lds[2]->semantics
				<< " scope=" << lds[2]->scope << "\n";
			return false;
		}
		if (lds[3]->semantics != PTXInstruction::Weak
			|| lds[3]->scope != PTXInstruction::Level_Invalid) {
			status << "ld (no qualifier, should default to weak): semantics="
				<< lds[3]->semantics << " scope=" << lds[3]->scope << "\n";
			return false;
		}
		if (lds[3]->addressSpace != PTXInstruction::Global) {
			status << "ld (no qualifier): addressSpace corrupted, got "
				<< lds[3]->addressSpace << "\n";
			return false;
		}
		if (sts[0]->semantics != PTXInstruction::Weak) {
			status << "st.weak: semantics=" << sts[0]->semantics << "\n";
			return false;
		}
		if (sts[1]->semantics != PTXInstruction::Relaxed
			|| sts[1]->scope != PTXInstruction::GlobalLevel) {
			status << "st.relaxed.gpu: semantics=" << sts[1]->semantics
				<< " scope=" << sts[1]->scope << "\n";
			return false;
		}
		if (sts[2]->semantics != PTXInstruction::Release
			|| sts[2]->scope != PTXInstruction::CtaLevel) {
			status << "st.release.cta: semantics=" << sts[2]->semantics
				<< " scope=" << sts[2]->scope << "\n";
			return false;
		}
		if (sts[3]->semantics != PTXInstruction::Weak
			|| sts[3]->scope != PTXInstruction::Level_Invalid) {
			status << "st (no qualifier, should default to weak): semantics="
				<< sts[3]->semantics << " scope=" << sts[3]->scope << "\n";
			return false;
		}
		if (sts[3]->addressSpace != PTXInstruction::Global) {
			status << "st (no qualifier): addressSpace corrupted, got "
				<< sts[3]->addressSpace << "\n";
			return false;
		}

		return true;
	}

	bool test_LdStMmio() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_ldst_mmio() {\n"
			<< "  .reg .f32 b, d;\n"
			<< "  .reg .u64 a;\n"
			<< "  ld.mmio.acquire.sys.global.f32 d, [a];\n"
			<< "  ld.mmio.relaxed.sys.f32 d, [a];\n"
			<< "  st.mmio.relaxed.sys.global.f32 [a], b;\n"
			<< "  st.mmio.release.sys.f32 [a], b;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse ld/st mmio example: "
				<< error.what() << "\n";
			return false;
		}
		std::vector<const PTXInstruction*> lds, sts;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptxIns = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptxIns) continue;
					if (ptxIns->opcode == PTXInstruction::Ld) lds.push_back(ptxIns);
					if (ptxIns->opcode == PTXInstruction::St) sts.push_back(ptxIns);
				}
		if (lds.size() != 2 || sts.size() != 2) {
			status << "expected 2 ld + 2 st instructions, found "
				<< lds.size() << " ld, " << sts.size() << " st\n";
			return false;
		}
		if (!lds[0]->mmio || lds[0]->semantics != PTXInstruction::Acquire
			|| lds[0]->scope != PTXInstruction::SystemLevel
			|| lds[0]->addressSpace != PTXInstruction::Global) {
			status << "ld.mmio.acquire.sys.global: mmio=" << lds[0]->mmio
				<< " semantics=" << lds[0]->semantics
				<< " scope=" << lds[0]->scope
				<< " addressSpace=" << lds[0]->addressSpace << "\n";
			return false;
		}
		if (!lds[1]->mmio || lds[1]->semantics != PTXInstruction::Relaxed
			|| lds[1]->scope != PTXInstruction::SystemLevel
			|| lds[1]->addressSpace != PTXInstruction::Global) {
			status << "ld.mmio.relaxed.sys (no .global): mmio=" << lds[1]->mmio
				<< " semantics=" << lds[1]->semantics
				<< " scope=" << lds[1]->scope
				<< " addressSpace=" << lds[1]->addressSpace << "\n";
			return false;
		}
		if (!sts[0]->mmio || sts[0]->semantics != PTXInstruction::Relaxed
			|| sts[0]->scope != PTXInstruction::SystemLevel
			|| sts[0]->addressSpace != PTXInstruction::Global) {
			status << "st.mmio.relaxed.sys.global: mmio=" << sts[0]->mmio
				<< " semantics=" << sts[0]->semantics
				<< " scope=" << sts[0]->scope
				<< " addressSpace=" << sts[0]->addressSpace << "\n";
			return false;
		}
		if (!sts[1]->mmio || sts[1]->semantics != PTXInstruction::Release
			|| sts[1]->scope != PTXInstruction::SystemLevel
			|| sts[1]->addressSpace != PTXInstruction::Global) {
			status << "st.mmio.release.sys (no .global): mmio=" << sts[1]->mmio
				<< " semantics=" << sts[1]->semantics
				<< " scope=" << sts[1]->scope
				<< " addressSpace=" << sts[1]->addressSpace << "\n";
			return false;
		}

		// Regression guard: .mmio must round-trip through the printer, not
		// collapse into the already-supported ld.acquire.sys form.
		std::string printed = lds[0]->toString();
		if (printed.find("mmio") == std::string::npos) {
			status << "ld.mmio toString() lost the mmio marker: " << printed << "\n";
			return false;
		}

		return true;
	}

	// Covers the atom/red type/operation table (PTX ISA Table 35/36) after
	// correcting f32 min/max, and/or/xor bitness, and inc/dec typing.
	bool test_AtomicRedTypeTable() {
		bool result = true;

		// atom.and.b64: verify bitwise AND and old-value writeback.
		{
			PTXU64 mem = 0xFF00FF00FF00FF00ULL;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::b64;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicAnd;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_uint("b", PTXOperand::b64, 0x0F0F0F0F0F0F0F0FULL);
			ins.d = reg("d", PTXOperand::b64, 1);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Atom(cta->getActiveContext(), ins);
			if (mem != 0x0F000F000F000F00ULL) {
				status << "atom.and.b64 gave " << std::hex << mem
					<< ", expected 0x0f000f000f000f00\n" << std::dec;
				result = false;
			}
			if (cta->getRegAsB64(0, 1) != 0xFF00FF00FF00FF00ULL) {
				status << "atom.and.b64 old-value writeback incorrect\n";
				result = false;
			}
		}

		// atom.add.f64: verify float64 add and old-value writeback.
		{
			PTXF64 mem = 10.5;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::f64;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicAdd;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_float("b", PTXOperand::f64, 2.25);
			ins.d = reg("d", PTXOperand::f64, 1);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Atom(cta->getActiveContext(), ins);
			if (mem != 10.5 + 2.25 * threadCount) {
				status << "atom.add.f64 gave " << mem << ", expected "
					<< (10.5 + 2.25 * threadCount) << "\n";
				result = false;
			}
			if (cta->getRegAsF64(0, 1) != 10.5) {
				status << "atom.add.f64 old-value writeback incorrect\n";
				result = false;
			}
		}

		// atom.min.s64: verify signed 64-bit min and old-value writeback.
		{
			PTXS64 mem = 100;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::s64;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicMin;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_int("b", PTXOperand::s64, -50);
			ins.d = reg("d", PTXOperand::s64, 1);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Atom(cta->getActiveContext(), ins);
			if (mem != -50) {
				status << "atom.min.s64 gave " << mem << ", expected -50\n";
				result = false;
			}
			if (cta->getRegAsS64(0, 1) != 100) {
				status << "atom.min.s64 old-value writeback incorrect\n";
				result = false;
			}
		}

		// red.and.b64: verify bitwise AND on a 64-bit value.
		{
			PTXU64 mem = 0xAAAAAAAAAAAAAAAAULL;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Red;
			ins.type = PTXOperand::b64;
			ins.addressSpace = PTXInstruction::Global;
			ins.reductionOperation = PTXInstruction::ReductionAnd;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_uint("b", PTXOperand::b64, 0x0F0F0F0F0F0F0F0FULL);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Red(cta->getActiveContext(), ins);
			if (mem != 0x0A0A0A0A0A0A0A0AULL) {
				status << "red.and.b64 gave " << std::hex << mem
					<< ", expected 0x0a0a0a0a0a0a0a0a\n" << std::dec;
				result = false;
			}
		}

		// red.add.f64: verify float64 add.
		{
			PTXF64 mem = 1.0;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Red;
			ins.type = PTXOperand::f64;
			ins.addressSpace = PTXInstruction::Global;
			ins.reductionOperation = PTXInstruction::ReductionAdd;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_float("b", PTXOperand::f64, 0.5);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Red(cta->getActiveContext(), ins);
			if (mem != 1.0 + 0.5 * threadCount) {
				status << "red.add.f64 gave " << mem << ", expected "
					<< (1.0 + 0.5 * threadCount) << "\n";
				result = false;
			}
		}

		// red.max.u64: verify unsigned 64-bit max.
		{
			PTXU64 mem = 5;
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Red;
			ins.type = PTXOperand::u64;
			ins.addressSpace = PTXInstruction::Global;
			ins.reductionOperation = PTXInstruction::ReductionMax;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_uint("b", PTXOperand::u64, 100);
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, (PTXU64)&mem);
			}
			cta->eval_Red(cta->getActiveContext(), ins);
			if (mem != 100) {
				status << "red.max.u64 gave " << mem << ", expected 100\n";
				result = false;
			}
		}

		// Negative validation cases: now-illegal type/operation combinations
		// must be rejected by PTXInstruction::valid().
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::f32;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicMin;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_float("b", PTXOperand::f32, 1.0);
			ins.d = reg("d", PTXOperand::f32, 1);
			if (ins.valid().empty()) {
				status << "atom.min.f32 was wrongly accepted as valid\n";
				result = false;
			}
		}
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Red;
			ins.type = PTXOperand::f32;
			ins.addressSpace = PTXInstruction::Global;
			ins.reductionOperation = PTXInstruction::ReductionMin;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_float("b", PTXOperand::f32, 1.0);
			if (ins.valid().empty()) {
				status << "red.min.f32 was wrongly accepted as valid\n";
				result = false;
			}
		}
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::s32;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicAnd;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_uint("b", PTXOperand::s32, 1);
			ins.d = reg("d", PTXOperand::s32, 1);
			if (ins.valid().empty()) {
				status << "atom.and.s32 was wrongly accepted as valid\n";
				result = false;
			}
		}
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Atom;
			ins.type = PTXOperand::u64;
			ins.addressSpace = PTXInstruction::Global;
			ins.atomicOperation = PTXInstruction::AtomicInc;
			ins.a = reg("a", PTXOperand::u64, 0);
			ins.a.addressMode = PTXOperand::Indirect;
			ins.b = imm_uint("b", PTXOperand::u64, 1);
			ins.d = reg("d", PTXOperand::u64, 1);
			if (ins.valid().empty()) {
				status << "atom.inc.u64 was wrongly accepted as valid\n";
				result = false;
			}
		}

		return result;
	}

#define argmin(a, b) ((a) > (b) ? (b) : (a))
#define argmax(a, b) ((b) > (a) ? (b) : (a))

	bool test_Min() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_minmax() {\n"
			<< "  .reg .f32 d, a, b;\n"
			<< "  .reg .b16 hd, ha, hb;\n"
			<< "  .reg .b32 pd, pa, pb;\n"
			<< "  min.NaN.f32 d, a, b;\n"
			<< "  min.xorsign.abs.f32 d, a, b;\n"
			<< "  min.NaN.f16 hd, ha, hb; min.xorsign.abs.f16 hd, ha, hb; max.NaN.f16 hd, ha, hb; max.xorsign.abs.f16 hd, ha, hb;\n"
			<< "  min.bf16 hd, ha, hb; min.NaN.bf16 hd, ha, hb; min.xorsign.abs.bf16 hd, ha, hb;\n"
			<< "  max.bf16 hd, ha, hb; max.NaN.bf16 hd, ha, hb; max.xorsign.abs.bf16 hd, ha, hb;\n"
			<< "  min.f16x2 pd, pa, pb; min.ftz.f16x2 pd, pa, pb; min.NaN.f16x2 pd, pa, pb; min.xorsign.abs.f16x2 pd, pa, pb;\n"
			<< "  max.f16x2 pd, pa, pb; max.ftz.f16x2 pd, pa, pb; max.NaN.f16x2 pd, pa, pb; max.xorsign.abs.f16x2 pd, pa, pb;\n"
			<< "  min.bf16x2 pd, pa, pb; min.NaN.bf16x2 pd, pa, pb; min.xorsign.abs.bf16x2 pd, pa, pb;\n"
			<< "  max.bf16x2 pd, pa, pb; max.NaN.bf16x2 pd, pa, pb; max.xorsign.abs.bf16x2 pd, pa, pb;\n"
			<< "  max.NaN.f32 d, a, b;\n"
			<< "  max.xorsign.abs.f32 d, a, b;\n  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse min/max modifier examples: "
				<< error.what() << "\n";
			return false;
		}
		const int expectedBf16x2[] = {0, PTXInstruction::nan,
			PTXInstruction::xorsign | PTXInstruction::abs, 0, PTXInstruction::nan,
			PTXInstruction::xorsign | PTXInstruction::abs};
		unsigned int parsedBf16x2 = 0;
		for (auto kernel = parsed.kernels().begin(); kernel != parsed.kernels().end(); ++kernel)
			for (auto block = kernel->second->cfg()->begin(); block != kernel->second->cfg()->end(); ++block)
				for (auto instruction = block->instructions.begin(); instruction != block->instructions.end(); ++instruction) {
					const PTXInstruction* ptx = dynamic_cast<const PTXInstruction*>(*instruction);
					if (!ptx || ptx->type != PTXOperand::bf16x2) continue;
					if (parsedBf16x2 >= 6 || ptx->modifier != expectedBf16x2[parsedBf16x2]) result = false;
					++parsedBf16x2;
				}
		if (parsedBf16x2 != 6) result = false;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Min;
		ins.type = PTXOperand::b16;
		ins.a = reg("a", PTXOperand::b16, 0);
		ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::b32;
		ins.a = reg("a", PTXOperand::b32, 0);
		ins.b = reg("b", PTXOperand::b32, 1);
		ins.d = reg("d", PTXOperand::b32, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::b64;
		ins.a = reg("a", PTXOperand::b64, 0);
		ins.b = reg("b", PTXOperand::b64, 1);
		ins.d = reg("d", PTXOperand::b64, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::u16;
		ins.a = reg("a", PTXOperand::b16, 0);
		ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		ins.modifier = PTXInstruction::rn;
		if (ins.valid().empty()) result = false;
		ins.modifier = PTXInstruction::sat | PTXInstruction::relu;
		if (ins.valid().empty()) result = false;
		ins.modifier = 0;
		ins.carry = PTXInstruction::CC;
		if (ins.valid().empty()) result = false;
		ins.carry = PTXInstruction::None;
		ins.d.type = PTXOperand::b32;
		if (ins.valid().empty()) result = false;
		const PTXOperand::DataType halfTypes[] = {PTXOperand::f16, PTXOperand::f16x2,
			PTXOperand::bf16, PTXOperand::bf16x2};
		const int halfModifiers[] = {PTXInstruction::nan, PTXInstruction::ftz | PTXInstruction::nan,
			PTXInstruction::nan, PTXInstruction::xorsign | PTXInstruction::abs};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.type = halfTypes[i]; ins.modifier = halfModifiers[i];
			const PTXOperand::DataType container = i % 2 ? PTXOperand::b32 : PTXOperand::b16;
			ins.a = reg("a", container, 0); ins.b = reg("b", container, 1); ins.d = reg("d", container, 2);
			if (!ins.valid().empty()) result = false;
		}
		ins.type = PTXOperand::bf16; ins.modifier = PTXInstruction::ftz;
		ins.a = reg("a", PTXOperand::b16, 0); ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::f16; ins.modifier = 0;
		ins.a = reg("a", PTXOperand::b16, 0); ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		cta->reset();
		struct HalfCase { int modifier; PTXU16 a, b, minimum, maximum; };
		const HalfCase cases[] = {
			{0,0x3c00,0x4000,0x3c00,0x4000}, {0,0x7c00,0x7c00,0x7c00,0x7c00},
			{0,0,0x8000,0x8000,0}, {0,0x8000,0,0x8000,0},
			{0,0x7e00,0x3c00,0x3c00,0x3c00}, {0,0x3c00,0x7e00,0x3c00,0x3c00},
			{0,0x7e00,0xfe00,0x7fff,0x7fff}, {PTXInstruction::nan,0x7e00,0x3c00,0x7fff,0x7fff},
			{0,0x8001,2,0x8001,2}, {PTXInstruction::ftz,0x8001,2,0x8000,0},
			{PTXInstruction::xorsign|PTXInstruction::abs,0xbc00,0x3e00,0xbc00,0xbe00},
			{PTXInstruction::xorsign|PTXInstruction::abs,0xfe00,0x3c00,0xbc00,0xbc00},
			{PTXInstruction::ftz|PTXInstruction::nan|PTXInstruction::xorsign|PTXInstruction::abs,
				0x8001,0x7e00,0x7fff,0x7fff}};
		for (const HalfCase& c : cases) for (int maximum = 0; maximum < 2; ++maximum) {
			ins.opcode = maximum ? PTXInstruction::Max : PTXInstruction::Min;
			ins.modifier = c.modifier; cta->setRegAsU16(0, 0, c.a); cta->setRegAsU16(0, 1, c.b);
			if (maximum) cta->eval_Max(cta->getActiveContext(), ins);
			else cta->eval_Min(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 2) != (maximum ? c.maximum : c.minimum)) result = false;
		}
		ins.type = PTXOperand::bf16; ins.modifier = 0;
		struct Bf16Case { int modifier; PTXU16 a, b, minimum, maximum; };
		const Bf16Case bf16Cases[] = {
			{0,0x3f80,0x4000,0x3f80,0x4000}, {0,0x7c01,0x3f80,0x3f80,0x7c01},
			{0,0,0x8000,0x8000,0}, {0,0x8001,1,0x8001,1},
			{0,0x7fc1,0x3f80,0x3f80,0x3f80}, {0,0x7fc1,0xffc2,0x7fff,0x7fff},
			{PTXInstruction::nan,0x7fc1,0x3f80,0x7fff,0x7fff},
			{PTXInstruction::xorsign|PTXInstruction::abs,0xbf80,0x3fc0,0xbf80,0xbfc0},
			{PTXInstruction::xorsign|PTXInstruction::abs,0xffc1,0x3f80,0xbf80,0xbf80}};
		ins.a = reg("a", PTXOperand::b16, 0); ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		for (const Bf16Case& c : bf16Cases) for (int maximum = 0; maximum < 2; ++maximum) {
			ins.opcode = maximum ? PTXInstruction::Max : PTXInstruction::Min;
			ins.modifier = c.modifier; cta->setRegAsU16(0, 0, c.a); cta->setRegAsU16(0, 1, c.b);
			if (maximum) cta->eval_Max(cta->getActiveContext(), ins);
			else cta->eval_Min(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 2) != (maximum ? c.maximum : c.minimum)) result = false;
		}
		struct PackedHalfCase { PTXOperand::DataType type; int modifier; PTXU32 a, b, minimum, maximum; };
		const PackedHalfCase packedCases[] = {
			{PTXOperand::f16x2,0,0x40003c00,0x3c004000,0x3c003c00,0x40004000},
			{PTXOperand::f16x2,0,0x7e003c00,0x3c007e00,0x3c003c00,0x3c003c00},
			{PTXOperand::f16x2,PTXInstruction::nan,0x7e003c00,0x3c007e00,0x7fff7fff,0x7fff7fff},
			{PTXOperand::f16x2,0,0x00008000,0x80000000,0x80008000,0x00000000},
			{PTXOperand::f16x2,0,0x80010001,0x00020002,0x80010001,0x00020002},
			{PTXOperand::f16x2,PTXInstruction::ftz,0x80010001,0x00020002,0x80000000,0x00000000},
			{PTXOperand::f16x2,PTXInstruction::xorsign|PTXInstruction::abs,0xbc003c00,0x3e00be00,0xbc00bc00,0xbe00be00},
			{PTXOperand::bf16x2,0,0x40003f80,0x3f804000,0x3f803f80,0x40004000},
			{PTXOperand::bf16x2,0,0x7c013f80,0x3f807c01,0x3f803f80,0x7c017c01},
			{PTXOperand::bf16x2,0,0x7fc13f80,0x3f807fc1,0x3f803f80,0x3f803f80},
			{PTXOperand::bf16x2,PTXInstruction::nan,0x7fc13f80,0x3f807fc1,0x7fff7fff,0x7fff7fff},
			{PTXOperand::bf16x2,0,0x80010000,0x00010001,0x80010000,0x00010001},
			{PTXOperand::bf16x2,PTXInstruction::xorsign|PTXInstruction::abs,0xbf803f80,0x3fc0bfc0,0xbf80bf80,0xbfc0bfc0}};
		ins.a = reg("pa", PTXOperand::b32, 0); ins.b = reg("pb", PTXOperand::b32, 1);
		ins.d = reg("pd", PTXOperand::b32, 2);
		for (const PackedHalfCase& c : packedCases) for (int maximum = 0; maximum < 2; ++maximum) {
			ins.type = c.type; ins.opcode = maximum ? PTXInstruction::Max : PTXInstruction::Min;
			ins.modifier = c.modifier; cta->setRegAsU32(0, 0, c.a); cta->setRegAsU32(0, 1, c.b);
			if (maximum) cta->eval_Max(cta->getActiveContext(), ins);
			else cta->eval_Min(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 2) != (maximum ? c.maximum : c.minimum)) result = false;
		}
		ins.type = PTXOperand::f16x2; ins.opcode = PTXInstruction::Min; ins.modifier = 0;
		ins.d = reg("pd", PTXOperand::b32, 0); cta->setRegAsU32(0, 0, 0x40003c00); cta->setRegAsU32(0, 1, 0x3c004000);
		cta->eval_Min(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 0) != 0x3c003c00) result = false;
		ins.d = reg("pd", PTXOperand::b32, 2); ins.opcode = PTXInstruction::Max;
		ins.pg.condition = PTXOperand::Pred; ins.pg.reg = 3; cta->setRegAsPredicate(0, 3, false);
		cta->setRegAsU32(0, 2, 0xa5a5a5a5); cta->eval_Max(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 2) != 0xa5a5a5a5) result = false;
		ins.opcode = PTXInstruction::Min; ins.modifier = 0; ins.carry = PTXInstruction::None;
		ins.pg.condition = PTXOperand::PT; ins.pg.reg = 0; ins.d = reg("d", PTXOperand::b16, 2);
		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			const PTXU16 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU16>::max)(), 0x8000};
			const PTXU16 bValues[] = {0, 0, 1, 0x8000, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, aValues[i % 5]);
				cta->setRegAsU16(i, 1, bValues[i % 5]);
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = argmin(aValues[i % 5], bValues[i % 5]);
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

			const PTXU32 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU32>::max)(), 0x80000000U};
			const PTXU32 bValues[] = {0, 0, 1, 0x80000000U, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, aValues[i % 5]);
				cta->setRegAsU32(i, 1, bValues[i % 5]);
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = argmin(aValues[i % 5], bValues[i % 5]);
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

			const PTXU64 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU64>::max)(), 0x8000000000000000ULL};
			const PTXU64 bValues[] = {0, 0, 1, 0x8000000000000000ULL, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, aValues[i % 5]);
				cta->setRegAsU64(i, 1, bValues[i % 5]);
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = argmin(aValues[i % 5], bValues[i % 5]);
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

			const PTXS16 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS16>::max)(),
				(std::numeric_limits<PTXS16>::min)()};
			const PTXS16 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS16>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, aValues[i % 5]);
				cta->setRegAsS16(i, 1, bValues[i % 5]);
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = argmin(aValues[i % 5], bValues[i % 5]);
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

			const PTXS32 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS32>::max)(),
				(std::numeric_limits<PTXS32>::min)()};
			const PTXS32 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS32>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, aValues[i % 5]);
				cta->setRegAsS32(i, 1, bValues[i % 5]);
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = argmin(aValues[i % 5], bValues[i % 5]);
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

			const PTXS64 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS64>::max)(),
				(std::numeric_limits<PTXS64>::min)()};
			const PTXS64 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS64>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, aValues[i % 5]);
				cta->setRegAsS64(i, 1, bValues[i % 5]);
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Min(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = argmin(aValues[i % 5], bValues[i % 5]);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "min.s64 incorrect\n";
					break;
				}
			}
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsS64(0, 2, 0x123456789LL);
			cta->eval_Min(cta->getActiveContext(), ins);
			if (cta->getRegAsS64(0, 2) != 0x123456789LL) result = false;
			ins.pg.condition = PTXOperand::PT;
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

		if (result) {
			ins.type = PTXOperand::f32;
			ins.modifier = PTXInstruction::nan | PTXInstruction::xorsign
				| PTXInstruction::abs;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);
			cta->setRegAsF32(0, 0,
				hydrazine::bit_cast<PTXF32>(0xffc00000U));
			cta->setRegAsF32(0, 1, 2.0f);
			cta->setRegAsF32(1, 0, -4.0f);
			cta->setRegAsF32(1, 1, 2.0f);
			cta->setRegAsF32(2, 0, 0.0f);
			cta->setRegAsF32(2, 1, -0.0f);
			cta->eval_Min(cta->getActiveContext(), ins);
			result = hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2))
				== 0x7fffffffU && cta->getRegAsF32(1, 2) == -2.0f
				&& std::signbit(cta->getRegAsF32(2, 2));
			ins.opcode = PTXInstruction::Max;
			cta->eval_Max(cta->getActiveContext(), ins);
			result = result && hydrazine::bit_cast<PTXU32>(
				cta->getRegAsF32(0, 2)) == 0x7fffffffU
				&& cta->getRegAsF32(1, 2) == -4.0f
				&& std::signbit(cta->getRegAsF32(2, 2));
			if (!result) status << "min/max modifiers incorrect\n";
		}

		return result;
	}


	bool test_Max() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Max;
		ins.type = PTXOperand::b16;
		ins.a = reg("a", PTXOperand::b16, 0);
		ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::b32;
		ins.a = reg("a", PTXOperand::b32, 0);
		ins.b = reg("b", PTXOperand::b32, 1);
		ins.d = reg("d", PTXOperand::b32, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::b64;
		ins.a = reg("a", PTXOperand::b64, 0);
		ins.b = reg("b", PTXOperand::b64, 1);
		ins.d = reg("d", PTXOperand::b64, 2);
		if (ins.valid().empty()) result = false;
		ins.type = PTXOperand::u16;
		ins.a = reg("a", PTXOperand::b16, 0);
		ins.b = reg("b", PTXOperand::b16, 1);
		ins.d = reg("d", PTXOperand::b16, 2);
		ins.modifier = PTXInstruction::rn;
		if (ins.valid().empty()) result = false;
		ins.modifier = PTXInstruction::sat | PTXInstruction::relu;
		if (ins.valid().empty()) result = false;
		ins.modifier = 0;
		ins.carry = PTXInstruction::CC;
		if (ins.valid().empty()) result = false;
		ins.carry = PTXInstruction::None;
		ins.d.type = PTXOperand::b32;
		if (ins.valid().empty()) result = false;

		// u16
		//
		if (result) {
			ins.type = PTXOperand::u16;
			ins.a = reg("r1", PTXOperand::u16, 0);
			ins.b = reg("r2", PTXOperand::u16, 1);
			ins.d = reg("r3", PTXOperand::u16, 2);

			const PTXU16 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU16>::max)(), 0x8000};
			const PTXU16 bValues[] = {0, 0, 1, 0x8000, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU16(i, 0, aValues[i % 5]);
				cta->setRegAsU16(i, 1, bValues[i % 5]);
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = argmax(aValues[i % 5], bValues[i % 5]);
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

			const PTXU32 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU32>::max)(), 0x80000000U};
			const PTXU32 bValues[] = {0, 0, 1, 0x80000000U, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU32(i, 0, aValues[i % 5]);
				cta->setRegAsU32(i, 1, bValues[i % 5]);
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = argmax(aValues[i % 5], bValues[i % 5]);
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

			const PTXU64 aValues[] = {0, 1, 0, (std::numeric_limits<PTXU64>::max)(), 0x8000000000000000ULL};
			const PTXU64 bValues[] = {0, 0, 1, 0x8000000000000000ULL, 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsU64(i, 0, aValues[i % 5]);
				cta->setRegAsU64(i, 1, bValues[i % 5]);
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = argmax(aValues[i % 5], bValues[i % 5]);
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

			const PTXS16 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS16>::max)(),
				(std::numeric_limits<PTXS16>::min)()};
			const PTXS16 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS16>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, aValues[i % 5]);
				cta->setRegAsS16(i, 1, bValues[i % 5]);
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = argmax(aValues[i % 5], bValues[i % 5]);
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

			const PTXS32 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS32>::max)(),
				(std::numeric_limits<PTXS32>::min)()};
			const PTXS32 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS32>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, aValues[i % 5]);
				cta->setRegAsS32(i, 1, bValues[i % 5]);
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = argmax(aValues[i % 5], bValues[i % 5]);
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

			const PTXS64 aValues[] = {0, -1, 0, (std::numeric_limits<PTXS64>::max)(),
				(std::numeric_limits<PTXS64>::min)()};
			const PTXS64 bValues[] = {0, 0, -1, (std::numeric_limits<PTXS64>::min)(), 0};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, aValues[i % 5]);
				cta->setRegAsS64(i, 1, bValues[i % 5]);
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Max(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = argmax(aValues[i % 5], bValues[i % 5]);
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "max.s64 incorrect\n";
					break;
				}
			}
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsS64(0, 2, 0x123456789LL);
			cta->eval_Max(cta->getActiveContext(), ins);
			if (cta->getRegAsS64(0, 2) != 0x123456789LL) result = false;
			ins.pg.condition = PTXOperand::PT;
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

		// f16x2
		//
		if (result) {
			ins.type = PTXOperand::f16x2;
			ins.modifier = 0;
			ins.a = reg("r1", PTXOperand::b32, 0);
			ins.d = reg("r3", PTXOperand::b32, 2);
			if (!ins.valid().empty()) {
				status << "neg.f16x2 rejected\n";
				return false;
			}
			cta->reset();
			auto packedNeg = [&](int modifier, PTXU32 a, PTXU32 expected) {
				ins.modifier = modifier;
				cta->setRegAsU32(0, 0, a);
				cta->setRegAsU32(0, 2, 0xdeadbeef);
				cta->eval_Neg(cta->getActiveContext(), ins);
				return cta->getRegAsU32(0, 2) == expected;
			};
			result = result && packedNeg(0, 0x3e00be00, 0xbe003e00);
			result = result && packedNeg(PTXInstruction::ftz, 0x80010001,
				0x00008000);
			std::stringstream ptx;
			ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
				<< ".visible .entry test_neg() { .reg .b32 d, a; "
				<< "neg.bf16x2 d, a; ret; }\n";
			try { Module parsed; parsed.load(ptx); }
			catch (const std::exception& error) {
				status << "neg.bf16x2 parse failed: " << error.what() << "\n";
				return false;
			}
			ins.type = PTXOperand::bf16x2;
			ins.modifier = 0;
			if (!ins.valid().empty()) return false;
			result = result && packedNeg(0, 0x3fc03f80, 0xbfc0bf80);
			result = result && packedNeg(0, 0x80010000, 0x00018000);
		}

		// s16
		//
		if (result) {
			ins.type = PTXOperand::s16;
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::s16, 0);
			ins.d = reg("r3", PTXOperand::s16, 2);
			if (ins.valid().empty()) result = false;
			ins.modifier = PTXInstruction::sat;
			if (ins.valid().empty()) result = false;
			ins.modifier = PTXInstruction::ftz;
			if (ins.valid().empty()) result = false;
			ins.modifier = 0;
			ins.carry = PTXInstruction::CC;
			if (ins.valid().empty()) result = false;
			ins.carry = PTXInstruction::None;

			const PTXS16 values[] = {0, 1, -1, (std::numeric_limits<PTXS16>::max)(),
				(std::numeric_limits<PTXS16>::min)()};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS16(i, 0, values[i % 5]);
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = values[i % 5] == values[4] ? values[4] : -values[i % 5];
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

			const PTXS32 values[] = {0, 1, -1, (std::numeric_limits<PTXS32>::max)(),
				(std::numeric_limits<PTXS32>::min)()};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS32(i, 0, values[i % 5]);
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = values[i % 5] == values[4] ? values[4] : -values[i % 5];
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

			const PTXS64 values[] = {0, 1, -1, (std::numeric_limits<PTXS64>::max)(),
				(std::numeric_limits<PTXS64>::min)()};
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsS64(i, 0, values[i % 5]);
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Neg(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = values[i % 5] == values[4] ? values[4] : -values[i % 5];
				if (cta->getRegAsS64(i, 2) != expected) {
					result = false;
					status << "neg.s64 incorrect\n";
					break;
				}
			}
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsS64(0, 0, 1);
			cta->setRegAsS64(0, 2, 0x123456789LL);
			cta->eval_Neg(cta->getActiveContext(), ins);
			if (cta->getRegAsS64(0, 2) != 0x123456789LL) result = false;
			ins.pg.condition = PTXOperand::PT;
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
				cta->setRegAsU16(i, 1, i == 0 ? 0 : (PTXU16)(4 + i));
				cta->setRegAsU16(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU16 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
				cta->setRegAsU32(i, 1, i == 0 ? 0 : (PTXU32)(4 + i));
				cta->setRegAsU32(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU32 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
				cta->setRegAsU64(i, 1, i == 0 ? 0 : (PTXU64)(4 + i));
				cta->setRegAsU64(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXU64 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
				cta->setRegAsS16(i, 1, i == 0 ? 0 : (PTXS16)(4 + i));
				cta->setRegAsS16(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS16 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
				cta->setRegAsS32(i, 1, i == 0 ? 0 : (PTXS32)(4 + i));
				cta->setRegAsS32(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS32 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
				cta->setRegAsS64(i, 1, i == 0 ? 0 : (PTXS64)(4 + i));
				cta->setRegAsS64(i, 2, 0);
			}
			cta->eval_Rem(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXS64 expected = i == 0 ? 0 : ((i * 8 + 8) % (4 + i));
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
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.d = reg("r3", PTXOperand::f32, 2);
			if (!ins.valid().empty()) { status << ins.valid() << "\n"; return false; }
			PTXInstruction invalid = ins;
			invalid.modifier = 0;
			if (invalid.valid().empty()) {
				status << "div.f32 accepted without a mode\n";
				return false;
			}

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
			ins.b = reg("r2", PTXOperand::f64, 1);
			ins.d = reg("r3", PTXOperand::f64, 2);
			PTXInstruction invalid = ins;
			invalid.modifier |= PTXInstruction::ftz;
			if (invalid.valid().empty()) {
				status << "div.ftz.f64 accepted\n";
				return false;
			}

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i * 8 + 8));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + i));
				cta->setRegAsF64(i, 2, 0.0);
			}
			cta->eval_Div(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 expected = ((PTXF64)(i * 8 + 8) / (PTXF64)(4 + i));
				if (std::fabs(cta->getRegAsF64(i, 2) - expected) > 0.1) {
					result = false;
					status << "div.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz,
			PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 expected[][2] = {{0x3dcccccd, 0xbdcccccd},
			{0x3dcccccc, 0xbdcccccc}, {0x3dcccccc, 0xbdcccccd},
			{0x3dcccccd, 0xbdcccccc}};
		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.d.type = PTXOperand::f32;
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			cta->setRegAsF32(0, 0, 1.0f);
			cta->setRegAsF32(0, 1, 10.0f);
			cta->setRegAsF32(1, 0, -1.0f);
			cta->setRegAsF32(1, 1, 10.0f);
			cta->eval_Div(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != expected[i][0]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != expected[i][1]) {
				status << "div.f32 rounding failed\n";
				return false;
			}
		}
		ins.type = PTXOperand::f64;
		ins.a.type = ins.b.type = ins.d.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::rz;
		cta->setRegAsF64(0, 0, 1.0);
		cta->setRegAsF64(0, 1, 10.0);
		cta->eval_Div(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2))
			!= 0x3fb9999999999999ull) return false;

		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.d.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		cta->setRegAsF32(0, 0, 1.0f);
		cta->setRegAsF32(1, 0, -1.0f);
		cta->setRegAsF32(2, 0, std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(3, 0, std::numeric_limits<PTXF32>::quiet_NaN());
		for (int i = 0; i < 4; ++i) {
			cta->setRegAsF32(i, 1, std::numeric_limits<PTXF32>::max());
		}
		cta->eval_Div(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != 0x80000000
			|| !hydrazine::isnan(cta->getRegAsF32(2, 2))
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(3, 2)) != 0) {
			status << "div.approx.f32 large divisor behavior failed\n";
			return false;
		}
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::denorm_min());
		cta->setRegAsF32(0, 1, 1.0f);
		cta->eval_Div(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 1) {
			status << "div.approx.f32 flushed a subnormal without ftz\n";
			return false;
		}

		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->setRegAsF32(0, 0, -std::numeric_limits<PTXF32>::denorm_min());
		cta->setRegAsF32(0, 1, 1.0f);
		cta->eval_Div(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0x80000000) {
			status << "div.approx.ftz.f32 input flushing failed\n";
			return false;
		}

		ins.modifier = PTXInstruction::full | PTXInstruction::ftz;
		if (!ins.valid().empty()) return false;
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::min());
		cta->setRegAsF32(0, 1, 2.0f);
		cta->eval_Div(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0) {
			status << "div.full.ftz.f32 result flushing failed\n";
			return false;
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
			ins.modifier = PTXInstruction::rn;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.c = reg("r3", PTXOperand::f32, 2);
			ins.d = reg("r4", PTXOperand::f32, 3);
			if (!ins.valid().empty()) {
				status << "mad.rn.f32 rejected\n";
				return false;
			}
			PTXInstruction invalid = ins;
			invalid.modifier = 0;
			if (invalid.valid().empty()) {
				status << "mad.f32 accepted without a rounding modifier\n";
				return false;
			}

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
			PTXInstruction invalid = ins;
			invalid.modifier |= PTXInstruction::sat;
			if (invalid.valid().empty()) {
				status << "mad.sat.f64 accepted\n";
				return false;
			}

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)(i - 1));
				cta->setRegAsF64(i, 1, (PTXF64)(4 + 2*i));
				cta->setRegAsF64(i, 2, (PTXF64)i);
				cta->setRegAsF64(i, 3, 0);
			}
			cta->eval_Mad(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 expected = (PTXF64)(i - 1) * (PTXF64)(4 + 2*i) + (PTXF64)(i);
				if (std::fabs(cta->getRegAsF64(i, 3) - expected) > 0.1) {
					result = false;
					status << "mad.f64 incorrect [" << i << "] - expected: " << expected 
						<< ", got " << cta->getRegAsF64(i, 2) << "\n";
					break;
				}
			}
		}

		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f32;
		cta->setRegAsF32(0, 0, 1.0f + std::ldexp(1.0f, -23));
		cta->setRegAsF32(0, 1, 1.0f - std::ldexp(1.0f, -23));
		cta->setRegAsF32(0, 2, -1.0f);
		cta->eval_Mad(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 3)) != 0xa8800000) {
			status << "mad.f32 was not fused\n";
			return false;
		}

		ins.type = PTXOperand::f64;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f64;
		cta->setRegAsF64(0, 0, 1.0 + std::ldexp(1.0, -52));
		cta->setRegAsF64(0, 1, 1.0 - std::ldexp(1.0, -52));
		cta->setRegAsF64(0, 2, -1.0);
		cta->eval_Mad(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 3))
			!= 0xb970000000000000ull) {
			status << "mad.f64 was not fused\n";
			return false;
		}

		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::min());
		cta->setRegAsF32(0, 1, 0.5f);
		cta->setRegAsF32(0, 2, 0.0f);
		cta->eval_Mad(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 3)) != 0) {
			status << "mad.ftz.f32 failed\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn | PTXInstruction::sat;
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::quiet_NaN());
		cta->setRegAsF32(0, 1, 1.0f);
		cta->eval_Mad(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 3) != 0.0f) {
			status << "mad.sat.f32 failed\n";
			return false;
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
		if (result) {
			ins.type = PTXOperand::f16;
			ins.a = reg("r1", PTXOperand::b16, 0);
			ins.b = reg("r2", PTXOperand::b16, 1);
			ins.d = reg("r4", PTXOperand::b16, 3);
			ins.modifier = 0; // PTX defaults to round-to-nearest-even.
			cta->setRegAsU16(0, 0, 0x3c01); // 1 + 2^-10
			cta->setRegAsU16(0, 1, 0x3c01); // 1 + 2^-10
			const int previous = hydrazine::fegetround();
			hydrazine::fesetround(FE_UPWARD);
			cta->eval_Mul(cta->getActiveContext(), ins);
			const bool roundedNearest = cta->getRegAsU16(0, 3) == 0x3c02;
			const bool restored = hydrazine::fegetround() == FE_UPWARD;
			hydrazine::fesetround(previous);
			if (!roundedNearest || !restored) {
				status << "mul.f16 default rounding failed\n";
				result = false;
			}
		}
		ins.modifier = PTXInstruction::rz;
		if (ins.valid().empty()) {
			status << "mul.rz.f16 accepted\n";
			result = false;
		}
		ins.modifier = PTXInstruction::ftz | PTXInstruction::sat;
		if (!ins.valid().empty()) {
			status << "mul.ftz.sat.f16 rejected\n";
			result = false;
		}

		// f16x2
		//
		if (result) {
			ins.type = PTXOperand::f16x2;
			ins.modifier = 0;
			ins.a = reg("r1", PTXOperand::b32, 0);
			ins.b = reg("r2", PTXOperand::b32, 1);
			ins.d = reg("r3", PTXOperand::b32, 2);
			if (!ins.valid().empty()) result = false;
			ins.modifier = PTXInstruction::rp;
			if (ins.valid().empty()) {
				status << "mul.rp.f16x2 accepted\n";
				result = false;
			}
			cta->reset();
			auto packedMul = [&](int modifier, PTXU32 a, PTXU32 b,
				PTXU32 expected, bool alias) {
				ins.modifier = modifier;
				ins.d.reg = alias ? 0 : 2;
				cta->setRegAsU32(0, 0, a);
				cta->setRegAsU32(0, 1, b);
				cta->setRegAsU32(0, 2, 0xdeadbeef);
				cta->eval_Mul(cta->getActiveContext(), ins);
				return cta->getRegAsU32(0, ins.d.reg) == expected;
			};
			result = result && packedMul(0, 0xc000be00, 0x3e00c000,
				0xc2004200, false);
			result = result && packedMul(0, 0x3c033c01, 0x3e003e00,
				0x3e043e02, false);
			result = result && packedMul(PTXInstruction::rn, 0x3c033c01,
				0x3e003e00, 0x3e043e02, false);
			result = result && packedMul(0, 0x00010001, 0x3c003c00,
				0x00010001, false);
			result = result && packedMul(PTXInstruction::ftz, 0x00010001,
				0x3c003c00, 0, false);
			result = result && packedMul(0, 0x84000400, 0x38003800,
				0x82000200, false);
			result = result && packedMul(PTXInstruction::ftz, 0x84000400,
				0x38003800, 0x80000000, false);
			result = result && packedMul(PTXInstruction::sat, 0xbc004000,
				0x3c003c00, 0x00003c00, false);
			result = result && packedMul(PTXInstruction::sat, 0x3c003800,
				0x3c003c00, 0x3c003800, false);
			result = result && packedMul(PTXInstruction::sat, 0x7e007e00,
				0x3c003c00, 0, false);
			result = result && packedMul(0, 0x00008000, 0xc0004000,
				0x80008000, false);
			result = result && packedMul(0, 0xc000be00, 0x3e00c000,
				0xc2004200, true);
			if (result) {
				const int previous = hydrazine::fegetround();
				hydrazine::fesetround(FE_UPWARD);
				const bool defaultRn = packedMul(0, 0x3c033c01,
					0x3e003e00, 0x3e043e02, false);
				const bool restored = hydrazine::fegetround() == FE_UPWARD;
				hydrazine::fesetround(previous);
				if (!defaultRn || !restored) result = false;
			}
			ins.modifier = 0;
			ins.d.reg = 2;
			ins.pg.condition = PTXOperand::Pred;
			ins.pg.reg = 3;
			cta->setRegAsPredicate(0, 3, false);
			cta->setRegAsU32(0, 2, 0xcafebabe);
			cta->eval_Mul(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 2) != 0xcafebabe) result = false;
			ins.pg.condition = PTXOperand::PT;
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

	bool test_MulRounding() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mul;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.b = reg("r2", PTXOperand::f32, 1);
		ins.d = reg("r3", PTXOperand::f32, 2);
		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz,
			PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 expected32[][2] = {{0x7f800000, 0xff800000},
			{0x7f7fffff, 0xff7fffff}, {0x7f7fffff, 0xff800000},
			{0x7f800000, 0xff7fffff}};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::max());
			cta->setRegAsF32(0, 1, 2.0f);
			cta->setRegAsF32(1, 0, -std::numeric_limits<PTXF32>::max());
			cta->setRegAsF32(1, 1, 2.0f);
			cta->eval_Mul(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != expected32[i][0]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != expected32[i][1]) {
				status << "mul.f32 rounding failed\n";
				return false;
			}
		}
		ins.type = PTXOperand::f64;
		ins.a.type = ins.b.type = ins.d.type = PTXOperand::f64;
		const PTXU64 expected64[][2] = {{0x7ff0000000000000ull, 0xfff0000000000000ull},
			{0x7fefffffffffffffull, 0xffefffffffffffffull},
			{0x7fefffffffffffffull, 0xfff0000000000000ull},
			{0x7ff0000000000000ull, 0xffefffffffffffffull}};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			cta->setRegAsF64(0, 0, std::numeric_limits<PTXF64>::max());
			cta->setRegAsF64(0, 1, 2.0);
			cta->setRegAsF64(1, 0, -std::numeric_limits<PTXF64>::max());
			cta->setRegAsF64(1, 1, 2.0);
			cta->eval_Mul(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2)) != expected64[i][0]
				|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2)) != expected64[i][1]) {
				status << "mul.f64 rounding failed\n";
				return false;
			}
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//
	// Floating-point instructions
	//
	/////////////////////////////////////////////////////////////////////////////////////////////////


	bool test_Rcp() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_rcp() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  .reg .f64 d0, d1;\n"
			<< "  rcp.approx.f32 f0, f1;\n"
			<< "  rcp.rz.ftz.f32 f0, f1;\n"
			<< "  rcp.rp.f64 d0, d1;\n"
			<< "  rcp.approx.ftz.f64 d0, d1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 rcp forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Rcp;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) {
			status << "rcp.f32 accepted without .approx or rounding mode\n";
			return false;
		}
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::f64;
		ins.a.type = ins.d.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		if (!ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx;
		if (ins.valid().empty()) return false;

		double freq = 2.0f / (double)threadCount;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.modifier = PTXInstruction::rn;
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
			ins.modifier = PTXInstruction::rn;
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

		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz,
			PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 expected32[][2] = {
			{0x3dcccccdu, 0xbdcccccdu}, {0x3dccccccu, 0xbdccccccu},
			{0x3dccccccu, 0xbdcccccdu}, {0x3dcccccdu, 0xbdccccccu}
		};
		const PTXU64 expected64[][2] = {
			{0x3fb999999999999aull, 0xbfb999999999999aull},
			{0x3fb9999999999999ull, 0xbfb9999999999999ull},
			{0x3fb9999999999999ull, 0xbfb999999999999aull},
			{0x3fb999999999999aull, 0xbfb9999999999999ull}
		};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			ins.type = PTXOperand::f32;
			ins.a.type = ins.d.type = PTXOperand::f32;
			cta->setRegAsF32(0, 0, 10.0f);
			cta->setRegAsF32(1, 0, -10.0f);
			cta->eval_Rcp(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != expected32[i][0]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != expected32[i][1]) {
				status << "rcp.f32 rounding failed\n";
				return false;
			}

			ins.type = PTXOperand::f64;
			ins.a.type = ins.d.type = PTXOperand::f64;
			cta->setRegAsF64(0, 0, 10.0);
			cta->setRegAsF64(1, 0, -10.0);
			cta->eval_Rcp(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2)) != expected64[i][0]
				|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2)) != expected64[i][1]) {
				status << "rcp.f64 rounding failed\n";
				return false;
			}
		}

		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->setRegAsF64(0, 0, 3.0);
		cta->setRegAsF64(1, 0, std::numeric_limits<PTXF64>::quiet_NaN());
		cta->eval_Rcp(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2))
			!= 0x3fd5555500000000ull
			|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2))
				!= 0x7fffffff00000000ull) {
			status << "rcp.approx.ftz.f64 failed\n";
			return false;
		}

		// Without .ftz, .approx must still support subnormal inputs and
		// return a finite approximate reciprocal, not flush to infinity.
		ins.type = PTXOperand::f32;
		ins.a.type = ins.d.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::approx;
		const PTXF32 subnormal = std::numeric_limits<PTXF32>::min()
			- std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 0, subnormal);
		cta->setRegAsF32(1, 0, -subnormal);
		cta->eval_Rcp(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0x7e800001u
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != 0xfe800001u) {
			status << "rcp.approx.f32 subnormal input failed\n";
			return false;
		}

		// With .ftz, the subnormal input flushes to signed zero first, so
		// the reciprocal is signed infinity.
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->setRegAsF32(0, 0, subnormal);
		cta->setRegAsF32(1, 0, -subnormal);
		cta->eval_Rcp(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2) != std::numeric_limits<PTXF32>::infinity()
			|| cta->getRegAsF32(1, 2) != -std::numeric_limits<PTXF32>::infinity()) {
			status << "rcp.approx.ftz.f32 subnormal input should flush to infinity\n";
			return false;
		}

		return result;
	}

	bool test_Cos() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_cos() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  cos.approx.f32 f0, f1;\n"
			<< "  cos.approx.ftz.f32 f0, f1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 cos forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Cos;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx;

		// f32
		//
		if (result) {
			float freq = 2 * 3.14159f / (float)threadCount;

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

		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::denorm_min());
		cta->setRegAsF32(1, 0, -std::numeric_limits<PTXF32>::denorm_min());
		cta->eval_Cos(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2) != 1.0f
			|| cta->getRegAsF32(1, 2) != 1.0f) return false;

		return result;
	}

	bool test_Sin() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_sin() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  sin.approx.f32 f0, f1;\n"
			<< "  sin.approx.ftz.f32 f0, f1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 sin forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sin;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx;

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

		const PTXF32 subnormal = std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 0, subnormal);
		cta->setRegAsF32(1, 0, -subnormal);
		cta->eval_Sin(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 1
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2))
				!= 0x80000001u) return false;

		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->eval_Sin(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2))
				!= 0x80000000u) return false;

		return result;
	}

	bool test_Tanh() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_tanh() {\n"
			<< "  .reg .f32 d, a;\n"
			<< "  .reg .b16 hd, ha; .reg .b32 pd, pa;\n"
			<< "  tanh.approx.f32 d, a;\n"
			<< "  tanh.approx.f16 hd, ha; tanh.approx.f16x2 pd, pa;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse tanh example: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Tanh;
		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::approx;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::f32, 1);
		for (int thread = 0; thread < threadCount; ++thread) {
			const PTXF32 input = (thread - threadCount / 2) / 4.0f;
			cta->setRegAsF32(thread, 1, input);
		}
		cta->eval_Tanh(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			const PTXF32 input = (thread - threadCount / 2) / 4.0f;
			if (std::fabs(cta->getRegAsF32(thread, 0) - std::tanh(input)) > 1e-6f)
				return false;
		}
		const PTXF32 subnormal = std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 1, subnormal);
		cta->setRegAsF32(1, 1, -subnormal);
		cta->eval_Tanh(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 0) != subnormal
			|| cta->getRegAsF32(1, 0) != -subnormal) return false;

		ins.type = PTXOperand::f16;
		ins.d = reg("hd", PTXOperand::b16, 0);
		ins.a = reg("ha", PTXOperand::b16, 1);
		const PTXU16 inputs[] = {0, 0x8000, 0x3c00, 0xbc00,
			0x7c00, 0xfc00, 1, 0x8001};
		const PTXU16 expected[] = {0, 0x8000, 0x3a18, 0xba18,
			0x3c00, 0xbc00, 1, 0x8001};
		for (unsigned int i = 0; i < 8; ++i) {
			cta->setRegAsU16(0, 1, inputs[i]);
			cta->eval_Tanh(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 0) != expected[i]) return false;
		}
		cta->setRegAsU16(0, 1, 0x7e00);
		cta->eval_Tanh(cta->getActiveContext(), ins);
		const PTXU16 nan = cta->getRegAsU16(0, 0);
		if ((nan & 0x7c00) != 0x7c00 || !(nan & 0x03ff)) return false;

		ins.type = PTXOperand::f16x2;
		ins.d = reg("pd", PTXOperand::b32, 0);
		ins.a = reg("pa", PTXOperand::b32, 1);
		cta->setRegAsU32(0, 1, 0x3c00bc00);
		cta->eval_Tanh(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 0) != 0x3a18ba18) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		if (ins.valid().empty()) return false;

		ins.modifier = PTXInstruction::approx;
		ins.type = PTXOperand::bf16;
		ins.d = reg("bd", PTXOperand::b16, 0);
		ins.a = reg("ba", PTXOperand::b16, 1);
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::bf16x2;
		ins.d = reg("pd", PTXOperand::b32, 0);
		ins.a = reg("pa", PTXOperand::b32, 1);
		return !ins.valid().empty();
	}
	
	bool test_CopySign() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::CopySign;

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
			cta->eval_CopySign(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF32 got = cta->getRegAsF32(i, 2);
				
				PTXF32 a = cta->getRegAsF32(i, 0);
				PTXF32 b = cta->getRegAsF32(i, 1);
				
				PTXF32 exp = std::copysign(b, a);
					
				if (got != exp) {
					result = false;
					status << "copysign.f32 incorrect [" << i << "] - expected: "
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
			cta->eval_CopySign(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 got = cta->getRegAsF64(i, 2);
				
				PTXF64 a = cta->getRegAsF64(i, 0);
				PTXF64 b = cta->getRegAsF64(i, 1);
				
				PTXF64 exp = std::copysign(b, a);
					
				if (got != exp) {
					result = false;
					status << "copysign.f64 incorrect [" << i << "] - expected: "
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
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_ex2() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  ex2.approx.f32 f0, f1;\n"
			<< "  ex2.approx.ftz.f32 f0, f1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 ex2 forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Ex2;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx;

		// f32
		//
		if (result) {
			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF32(i, 0,
					-4.0f + (PTXF32)i / (PTXF32)threadCount * 8.0f);
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Ex2(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				const PTXF32 value = -4.0f
					+ (PTXF32)i / (PTXF32)threadCount * 8.0f;
				const PTXF32 expected = (PTXF32)std::exp2((double)value);
				const PTXU32 actualBits = hydrazine::bit_cast<PTXU32>(
					cta->getRegAsF32(i, 2));
				const PTXU32 expectedBits = hydrazine::bit_cast<PTXU32>(expected);
				const PTXU32 ulps = actualBits > expectedBits
					? actualBits - expectedBits : expectedBits - actualBits;
				if (ulps > 2) {
					result = false;
					status << "ex2.f32 incorrect [" << i << "] - expected: "
						<< expected
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		cta->setRegAsF32(0, 0, -std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(1, 0, -0.0f);
		cta->setRegAsF32(2, 0, 0.0f);
		cta->setRegAsF32(3, 0, std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(4, 0, std::numeric_limits<PTXF32>::quiet_NaN());
		cta->eval_Ex2(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0
			|| cta->getRegAsF32(1, 2) != 1.0f
			|| cta->getRegAsF32(2, 2) != 1.0f
			|| cta->getRegAsF32(3, 2) != std::numeric_limits<PTXF32>::infinity()
			|| !hydrazine::isnan(cta->getRegAsF32(4, 2))) return false;

		cta->setRegAsF32(0, 0, -149.0f);
		cta->eval_Ex2(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 1) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->eval_Ex2(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0) return false;

		return result;
	}

	bool test_Fma() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fma;
		ins.modifier = PTXInstruction::rn;

		// f32
		//
		if (result) {
			ins.type = PTXOperand::f32;
			ins.a = reg("r1", PTXOperand::f32, 0);
			ins.b = reg("r2", PTXOperand::f32, 1);
			ins.c = reg("r4", PTXOperand::f32, 3);
			ins.d = reg("r3", PTXOperand::f32, 2);
			if (!ins.valid().empty()) {
				status << "fma.rn.f32 rejected\n";
				return false;
			}
			PTXInstruction invalid = ins;
			invalid.modifier = 0;
			if (invalid.valid().empty()) {
				status << "fma.f32 accepted without a rounding modifier\n";
				return false;
			}
			invalid = ins;
			invalid.modifier |= PTXInstruction::rz;
			if (invalid.valid().empty()) {
				status << "fma.f32 accepted multiple rounding modifiers\n";
				return false;
			}
			invalid = ins;
			invalid.c.type = PTXOperand::s32;
			if (invalid.valid().empty()) {
				status << "fma.f32 accepted a mismatched C operand\n";
				return false;
			}

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
			PTXInstruction invalid = ins;
			invalid.modifier |= PTXInstruction::ftz;
			if (invalid.valid().empty()) {
				status << "fma.ftz.f64 accepted\n";
				return false;
			}
			invalid = ins;
			invalid.modifier |= PTXInstruction::sat;
			if (invalid.valid().empty()) {
				status << "fma.sat.f64 accepted\n";
				return false;
			}

			for (int i = 0; i < threadCount; i++) {
				cta->setRegAsF64(i, 0, (PTXF64)((double)i / (double)threadCount * 4.5));
				cta->setRegAsF64(i, 1, (PTXF64)((double)i / (double)threadCount * 2.25));
				cta->setRegAsF64(i, 3, (PTXF64)((double)i / (double)threadCount * 0.55));
				cta->setRegAsF64(i, 2, 0);
			}
			cta->eval_Fma(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				PTXF64 got = cta->getRegAsF64(i, 2);
				PTXF64 exp = (double)i / (double)threadCount * 4.5 * (double)i / (double)threadCount * 2.25 +
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

		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz,
			PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 expected32[][2] = {{0x7f800000, 0xff800000},
			{0x7f7fffff, 0xff7fffff}, {0x7f7fffff, 0xff800000},
			{0x7f800000, 0xff7fffff}};
		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f32;
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::max());
			cta->setRegAsF32(0, 1, 2.0f);
			cta->setRegAsF32(0, 3, 0.0f);
			cta->setRegAsF32(1, 0, -std::numeric_limits<PTXF32>::max());
			cta->setRegAsF32(1, 1, 2.0f);
			cta->setRegAsF32(1, 3, 0.0f);
			cta->setRegAsF32(2, 0, 1.0f + std::ldexp(1.0f, -23));
			cta->setRegAsF32(2, 1, 1.0f - std::ldexp(1.0f, -23));
			cta->setRegAsF32(2, 3, -1.0f);
			cta->eval_Fma(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != expected32[i][0]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != expected32[i][1]
				|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(2, 2)) != 0xa8800000) {
				status << "fma.f32 fused rounding failed\n";
				return false;
			}
		}

		const PTXU64 expected64[][2] = {{0x7ff0000000000000ull, 0xfff0000000000000ull},
			{0x7fefffffffffffffull, 0xffefffffffffffffull},
			{0x7fefffffffffffffull, 0xfff0000000000000ull},
			{0x7ff0000000000000ull, 0xffefffffffffffffull}};
		ins.type = PTXOperand::f64;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f64;
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			cta->setRegAsF64(0, 0, std::numeric_limits<PTXF64>::max());
			cta->setRegAsF64(0, 1, 2.0);
			cta->setRegAsF64(0, 3, 0.0);
			cta->setRegAsF64(1, 0, -std::numeric_limits<PTXF64>::max());
			cta->setRegAsF64(1, 1, 2.0);
			cta->setRegAsF64(1, 3, 0.0);
			cta->setRegAsF64(2, 0, 1.0 + std::ldexp(1.0, -52));
			cta->setRegAsF64(2, 1, 1.0 - std::ldexp(1.0, -52));
			cta->setRegAsF64(2, 3, -1.0);
			cta->eval_Fma(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2)) != expected64[i][0]
				|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2)) != expected64[i][1]
				|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(2, 2)) != 0xb970000000000000ull) {
				status << "fma.f64 fused rounding failed\n";
				return false;
			}
		}

		ins.type = PTXOperand::f32;
		ins.a.type = ins.b.type = ins.c.type = ins.d.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		if (!ins.valid().empty()) {
			status << "fma.rn.ftz.f32 rejected\n";
			return false;
		}
		cta->setRegAsF32(0, 0, std::numeric_limits<PTXF32>::min());
		cta->setRegAsF32(0, 1, 0.5f);
		cta->setRegAsF32(0, 3, 0.0f);
		cta->setRegAsF32(1, 0, std::numeric_limits<PTXF32>::denorm_min());
		cta->setRegAsF32(1, 1, std::numeric_limits<PTXF32>::max());
		cta->setRegAsF32(1, 3, 0.0f);
		cta->setRegAsF32(2, 0, -std::numeric_limits<PTXF32>::min());
		cta->setRegAsF32(2, 1, 0.5f);
		cta->setRegAsF32(2, 3, -0.0f);
		cta->eval_Fma(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(2, 2)) != 0x80000000) {
			status << "fma.rn.ftz.f32 failed\n";
			return false;
		}

		ins.modifier = PTXInstruction::rn | PTXInstruction::sat;
		cta->setRegAsF32(0, 0, 2.0f);
		cta->setRegAsF32(0, 1, 1.0f);
		cta->setRegAsF32(0, 3, 0.0f);
		cta->setRegAsF32(1, 0, -1.0f);
		cta->setRegAsF32(1, 1, 1.0f);
		cta->setRegAsF32(1, 3, 0.0f);
		cta->setRegAsF32(2, 0, std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(2, 1, 0.0f);
		cta->setRegAsF32(2, 3, 0.0f);
		cta->eval_Fma(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2) != 1.0f
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(2, 2)) != 0) {
			status << "fma.rn.sat.f32 failed\n";
			return false;
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

		auto scalarFma = [&](int modifier, PTXU16 a, PTXU16 b, PTXU16 c,
			PTXU16 expected) {
			ins.modifier = modifier;
			cta->setRegAsU16(0, 0, a); cta->setRegAsU16(0, 1, b);
			cta->setRegAsU16(0, 3, c); cta->setRegAsU16(0, 2, 0xbeef);
			cta->eval_Fma(cta->getActiveContext(), ins);
			return cta->getRegAsU16(0, 2) == expected;
		};
		if (!scalarFma(PTXInstruction::rn, 0x3fe0, 0x3f14, 0xab80, 0x3f81)
			|| !scalarFma(PTXInstruction::rn, 0xbfe0, 0x3f14, 0x2b80, 0xbf81)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::relu,
				0xbf80, 0x3f80, 0, 0x0000)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::relu,
				0x7fc0, 0x3f80, 0, 0x7fff)
			|| !scalarFma(PTXInstruction::rn, 0x7f7f, 0x4000, 0, 0x7f80)) {
			status << "fma.bf16 scalar case failed\n";
			return false;
		}
		auto checkValid = [&](int modifier, bool shouldAccept) {
			ins.modifier = modifier;
			return ins.valid().empty() == shouldAccept;
		};
		if (!checkValid(PTXInstruction::rn, true)
			|| !checkValid(PTXInstruction::rz, false)
			|| !checkValid(PTXInstruction::rn | PTXInstruction::ftz, false)
			|| !checkValid(PTXInstruction::rn | PTXInstruction::sat, false)
			|| !checkValid(PTXInstruction::rn | PTXInstruction::relu, true)) {
			status << "fma.bf16 validation failed\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn;
		ins.type = PTXOperand::bf16x2;
		ins.a = reg("r1", PTXOperand::b32, 0); ins.b = reg("r2", PTXOperand::b32, 1);
		ins.c = reg("r4", PTXOperand::b32, 3); ins.d = reg("r3", PTXOperand::b32, 2);
		if (!checkValid(PTXInstruction::rn, true)) {
			status << "fma.rn.bf16x2 rejected\n";
			return false;
		}

		auto packedFma = [&](int modifier, PTXU32 a, PTXU32 b, PTXU32 c,
			PTXU32 expected) {
			ins.modifier = modifier;
			cta->setRegAsU32(0, 0, a); cta->setRegAsU32(0, 1, b);
			cta->setRegAsU32(0, 3, c); cta->setRegAsU32(0, 2, 0xdeadbeef);
			cta->eval_Fma(cta->getActiveContext(), ins);
			return cta->getRegAsU32(0, 2) == expected;
		};
		if (!packedFma(PTXInstruction::rn, 0xbfe03fe0, 0x3f143f14,
				0x2b80ab80, 0xbf813f81)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::relu,
				0x3f80bf80, 0x3f803f80, 0, 0x3f800000)) {
			status << "fma.bf16x2 packed case failed\n";
			return false;
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

		auto scalarFma = [&](int modifier, PTXU16 a, PTXU16 b, PTXU16 c,
			PTXU16 expected) {
			ins.modifier = modifier;
			cta->setRegAsU16(0, 0, a); cta->setRegAsU16(0, 1, b);
			cta->setRegAsU16(0, 3, c); cta->setRegAsU16(0, 2, 0xbeef);
			cta->eval_Fma(cta->getActiveContext(), ins);
			return cta->getRegAsU16(0, 2) == expected;
		};
		if (!scalarFma(PTXInstruction::rn, 0x3bfe, 0x9001, 0x3c01, 0x3c01)
			|| !scalarFma(PTXInstruction::rn, 0x0001, 0x3c00, 0, 0x0001)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x0001, 0x3c00, 0, 0x0000)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x0400, 0x3800, 0, 0x0000)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x8400, 0x3800, 0, 0x8000)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::sat,
				0x4000, 0x4000, 0, 0x3c00)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::relu,
				0xbc00, 0x3c00, 0, 0x0000)
			|| !scalarFma(PTXInstruction::rn | PTXInstruction::relu,
				0x7e00, 0x3c00, 0, 0x7fff)) {
			status << "fma.f16 scalar case failed\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn;
		if (!ins.valid().empty()) {
			status << "fma.rn.f16 rejected\n";
			return false;
		}
		ins.modifier = PTXInstruction::rz;
		if (ins.valid().empty()) {
			status << "fma.rz.f16 accepted\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn | PTXInstruction::sat
			| PTXInstruction::relu;
		if (ins.valid().empty()) {
			status << "fma.rn.sat.relu.f16 accepted\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz
			| PTXInstruction::relu;
		if (!ins.valid().empty()) {
			status << "fma.rn.ftz.relu.f16 rejected\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn;

		ins.type = PTXOperand::f16x2;
		ins.a = reg("a", PTXOperand::b32, 0);
		ins.b = reg("b", PTXOperand::b32, 1);
		ins.c = reg("c", PTXOperand::b32, 3);
		ins.d = reg("d", PTXOperand::b32, 2);
		if (!ins.valid().empty()) {
			status << "fma.rn.f16x2 rejected\n";
			return false;
		}
		ins.modifier = PTXInstruction::rm;
		if (ins.valid().empty()) {
			status << "fma.rm.f16x2 accepted\n";
			return false;
		}
		ins.modifier = PTXInstruction::rn;
		cta->reset();
		auto packedFma = [&](int modifier, PTXU32 a, PTXU32 b, PTXU32 c,
			PTXU32 expected, bool alias) {
			ins.modifier = modifier;
			ins.d.reg = alias ? 0 : 2;
			cta->setRegAsU32(0, 0, a); cta->setRegAsU32(0, 1, b);
			cta->setRegAsU32(0, 3, c); cta->setRegAsU32(0, 2, 0xdeadbeef);
			cta->eval_Fma(cta->getActiveContext(), ins);
			return cta->getRegAsU32(0, ins.d.reg) == expected;
		};
		if (!packedFma(PTXInstruction::rn, 0x40003c01, 0x42003c01,
			0x3c00bc02, 0x47000010, false)
			|| !packedFma(PTXInstruction::rn, 0x3c033c01, 0x3e003e00,
				0x00018001, 0x3e053e01, false)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x40003c01, 0x42003c01, 0x3c00bc02, 0x47000000, false)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x3c033c01, 0x3e003e00, 0x00018001, 0x3e043e02, false)
			|| !packedFma(PTXInstruction::rn, 0x00010001, 0x3c003c00,
				0, 0x00010001, false)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::ftz,
				0x00010001, 0x3c003c00, 0, 0, false)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::sat,
				0x7e004000, 0x3c003c00, 0, 0x00003c00, false)
			|| !packedFma(PTXInstruction::rn | PTXInstruction::relu,
				0xbc007e00, 0x3c003c00, 0, 0x00007fff, false)
			|| !packedFma(PTXInstruction::rn, 0x40003c01, 0x42003c01,
				0x3c00bc02, 0x47000010, true)) return false;
		const int previous = hydrazine::fegetround();
		hydrazine::fesetround(FE_UPWARD);
		const bool hostRn = packedFma(PTXInstruction::rn, 0x3c003c00,
			0x3c013c01, 0x90009000, 0x3c003c00, false);
		const bool restored = hydrazine::fegetround() == FE_UPWARD;
		hydrazine::fesetround(previous);
		if (!hostRn || !restored) return false;
		ins.modifier = PTXInstruction::rn;
		ins.d.reg = 2;
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 4;
		cta->setRegAsPredicate(0, 4, false);
		cta->setRegAsU32(0, 2, 0xcafebabe);
		cta->eval_Fma(cta->getActiveContext(), ins);
		ins.pg.condition = PTXOperand::PT;
		if (cta->getRegAsU32(0, 2) != 0xcafebabe) return false;

		return true;
	}

	bool test_MmaWarpParticipation() {
		for (const char* form : {
			"m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%r0,%r1},{%r4},{%r5},{%r2,%r3}",
			"m8n8k4.row.col.f64.f64.f64.f64 {%r0,%r1},{%r4},{%r5},{%r2,%r3}",
			"m8n8k16.row.col.s32.s8.s8.s32 {%r0,%r1},{%r4},{%r5},{%r2,%r3}",
			"m8n8k4.row.col.f16.f16.f16.f16 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r10,%r11},{%r4,%r5,%r6,%r7}",
			"m16n8k8.row.col.f32.f16.f16.f32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r10},{%r4,%r5,%r6,%r7}",
			"sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r0,%r1},{%r8,%r9},{%r10,%r11},{%r4,%r5},%r12,0",
			"sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r10,%r11},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r10,%r11},{%r4,%r5,%r6,%r7},%r12,3",
			"sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r10,%r11},{%r4,%r5,%r6,%r7},%r12,2",
			"sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%r0,%r1},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5},%r12,0",
			"sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%r0,%r1,%r2,%r3},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {%r0,%r1,%r2,%r3},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%r0,%r1,%r2,%r3},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r16,%r17},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k64.row.col.s32.u8.s8.s32 {%r0,%r1,%r2,%r3},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5,%r6,%r7},%r12,0",
			"sp.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%r0,%r1,%r2,%r3},{%r8,%r9},{%r16,%r17},{%r4,%r5,%r6,%r7},%r12,1",
			"sp.sync.aligned.m16n8k128.row.col.s32.u4.s4.s32 {%r0,%r1,%r2,%r3},{%r8,%r9,%r10,%r11},{%r16,%r17,%r18,%r19},{%r4,%r5,%r6,%r7},%r12,0"}) {
			const bool fp64 = std::string(form).find("f64") != std::string::npos;
			const bool sparse = std::string(form).find("sp.") == 0;
			std::stringstream source(std::string(".version 8.0\n.target sm_86\n.entry warp_mma() { .reg .pred %p; .reg .") +
				(fp64 ? "b64" : "b32") + " %r<20>; @%p mma." + (sparse ? "" : "sync.aligned.") + form + "; ret; }");
			Module parsed; parsed.load(source);
			EmulatedKernel localKernel(parsed.getKernel("warp_mma"), nullptr);
			for (int size : {33, 65, 1, 31}) {
				localKernel.setKernelShape(size, 1, 1);
				CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
				for (const auto& ins : localKernel.instructions) if (ins.opcode == PTXInstruction::Mma) {
					const int completeThreads = (size / 32) * 32;
					for (int scenario : {1, 0, 2, 3, 4}) {
						if (scenario == 3 && completeThreads == 0) continue;
						localCTA.reset();
						auto& context = localCTA.getActiveContext();
						for (int thread = 0; thread < size; ++thread) {
							for (unsigned regID = 0; regID < localKernel.registerCount(); ++regID) localCTA.setRegAsB64(thread, regID, 0);
							for (const auto& dest : ins.d.array) localCTA.setRegAsB64(thread, dest.reg, 0xdeadbeef);
							if (sparse) localCTA.setRegAsB32(thread, ins.q.reg, 0x44444444u);
							const bool predicate = scenario == 2 || scenario == 4 ||
								(scenario != 0 && thread < completeThreads && (scenario != 3 || thread != 0));
							localCTA.setRegAsPredicate(thread, ins.pg.reg, predicate);
							if (scenario == 4 && thread >= completeThreads) context.active.reset(thread);
						}
						bool rejected = false;
						try { localCTA.eval_Mma(context, ins); } catch (const RuntimeException&) { rejected = true; }
						if (rejected != (scenario == 2 || scenario == 3)) {
							status << "mma participation mismatch: " << form << " CTA " << size << " scenario " << scenario << "\n";
							return false;
						}
						if (!rejected) for (int thread = 0; thread < size; ++thread) for (const auto& dest : ins.d.array) {
							const PTXU64 expected = scenario != 0 && thread < completeThreads ? 0 : 0xdeadbeef;
							if (localCTA.getRegAsB64(thread, dest.reg) != expected) { status << "mma inactive lane changed or active result incorrect\n"; return false; }
						}
					}
				}
			}
		}
		return true;
	}

	bool test_VoteSyncDivergentArrival() {
		for (int size : {1, 31, 32}) for (const char* mode : {"all", "any", "uni", "ballot"}) {
			const bool ballot = std::string(mode) == "ballot";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry delayed() { .reg .u64 ptr; .reg .u32 lane,delay,result; .reg .pred p,a,d; "
				"mov.u32 lane,%tid.x; mov.u32 result,0xdeadbeef; setp.ge.u32 p,lane,16; selp.u32 delay,1,0,p; setp.eq.u32 a,lane,16; "
				"loop: setp.eq.u32 p,lane,31; @p bra done; setp.ne.u32 p,delay,0; @p bra later; vote.sync." + std::string(mode) +
				(ballot ? ".b32 result" : ".pred d") + ",a,0x7fffffff; " + (ballot ? "" : "selp.u32 result,1,0,d; ") +
				"bra done; later: sub.u32 delay,delay,1; bra loop; done: st.global.u32 [ptr],result; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("delayed"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (int run = 0; run < 2; ++run) {
				local.reset(); PTXU32 output[32] = {};
				for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::St)
					for (int lane = 0; lane < size; ++lane) local.setRegAsU64(lane, ins.d.reg, (PTXU64)&output[lane]);
				local.execute();
				for (int lane = 0; lane < size; ++lane) {
					const PTXU32 expected = lane == 31 ? 0xdeadbeef : ballot ? (size > 16 ? 0x10000 : 0) : std::string(mode) == "any" ? size > 16 : std::string(mode) == "uni" && size == 1;
					if (output[lane] != expected) { status << "divergent vote arrival mismatch " << mode << " lane " << lane << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_VoteSync() {
		for (const char* mode : {"all", "any", "uni", "ballot"}) for (bool invert : {false, true}) for (bool alias : {false, true}) {
			const bool ballot = std::string(mode) == "ballot";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry vote_execute() { .reg .pred a, d, g; .reg .b32 r, m; @!g vote.sync." +
				std::string(mode) + (ballot ? ".b32 r, " : ".pred d, ") + (invert ? "!a" : "a") + ", m; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("vote_execute"), nullptr);
			for (int size : {1, 31, 32, 33, 64, 65}) for (int scenario = 0; scenario < 4; ++scenario) {
				kernel.setKernelShape(size, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
				for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Vote) {
					if (alias) ins.d.reg = ballot ? ins.b.reg : ins.a.reg;
					auto& context = local.getActiveContext(); std::vector<PTXU32> initial(size);
					for (int thread = 0; thread < size; ++thread) {
						const int lane = thread % 32;
						const PTXU32 truth = thread < 32 ? 0x00000003u : 0x80008000u;
						const PTXU32 mask = scenario == 0 ? ~PTXU32(0) : scenario == 1 ? (lane % 2 ? 0xaaaaaaaau : 0x55555555u) : scenario == 2 ? 0x80000081u : 0x1cu;
						local.setRegAsPredicate(thread, ins.a.reg, (truth & (PTXU32(1) << lane)) != 0);
						local.setRegAsU32(thread, ins.b.reg, mask); local.setRegAsPredicate(thread, ins.pg.reg, false);
						if (!alias) { if (ballot) local.setRegAsB32(thread, ins.d.reg, 0xdeadbeef); else local.setRegAsPredicate(thread, ins.d.reg, true); }
						context.active[thread] = scenario != 2 || (mask & (PTXU32(1) << lane));
						initial[thread] = ballot ? local.getRegAsB32(thread, ins.d.reg) : local.getRegAsPredicate(thread, ins.d.reg);
					}

					local.eval_WarpSync(context, ins);
					for (int thread = 0; thread < size; ++thread) {
						const int lane = thread % 32, warpStart = thread / 32 * 32;
						const PTXU32 members = scenario == 0 ? ~PTXU32(0) : scenario == 1 ? (lane % 2 ? 0xaaaaaaaau : 0x55555555u) : scenario == 2 ? 0x80000081u : 0x1cu;
						PTXU32 bits = 0; int count = 0, ones = 0;
						for (int member = 0; member < 32 && warpStart + member < size; ++member) if (members & (PTXU32(1) << member)) {
							++count; const bool value = ((warpStart == 0 ? 0x00000003u : 0x80008000u) & (PTXU32(1) << member)) != 0;
							if (value != invert) { ++ones; bits |= PTXU32(1) << member; }
						}
						const PTXU32 result = ballot ? bits : std::string(mode) == "all" ? ones == count : std::string(mode) == "any" ? ones != 0 : ones == 0 || ones == count;
						const PTXU32 actual = ballot ? local.getRegAsB32(thread, ins.d.reg) : local.getRegAsPredicate(thread, ins.d.reg);
						if (actual != (context.active[thread] ? result : initial[thread])) { status << "vote.sync mismatch " << mode << " CTA " << size << " thread " << thread << "\n"; return false; }
					}
				}
			}
		}
		return true;
	}

	bool test_VoteSyncDifferentMasks() {
		for (const char* mode : {"all", "any", "uni", "ballot"}) {
			const bool ballot = std::string(mode) == "ballot";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry unchecked() { .reg .pred a,d; .reg .b32 r,m; vote.sync." +
				std::string(mode) + (ballot ? ".b32 r" : ".pred d") + ", a,m; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("unchecked"), nullptr);
			kernel.setKernelShape(2,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Vote)
			for (bool allTrue : {false, true}) {
				local.reset(); auto& context = local.getActiveContext();
				for (int lane = 0; lane < 2; ++lane) {
					local.setRegAsPredicate(lane, ins.a.reg, lane == 0 || allTrue);
					local.setRegAsU32(lane, ins.b.reg, lane == 0 ? 1 : 3);
				}
				try { local.eval_WarpSync(context, ins); }
				catch (const RuntimeException& error) { status << "different vote mask threw: " << error.what() << "\n"; return false; }
				const PTXU32 expected = ballot ? (allTrue ? 3 : 1) :
					std::string(mode) == "any" ? 1 : allTrue;
				const PTXU32 actual = ballot ? local.getRegAsB32(1, ins.d.reg) : local.getRegAsPredicate(1, ins.d.reg);
				if (actual != expected) { status << "different vote mask result mismatch\n"; return false; }
			}
		}
		return true;
	}

	bool test_ReduxSynchronization() {
		const struct { const char* form; PTXU32 base, first, last, expected; } cases[] = {
			{"add.s32", 0xfffffff0u, 0x80000000u, 0x12345678u, 0x12345498u},
			{"add.u32", 0xfffffff0u, 0x80000000u, 0x12345678u, 0x12345498u},
			{"min.s32", 0x00000009u, 0xfffffff0u, 0x80000000u, 0x80000000u},
			{"min.u32", 0x0000007bu, 0x00000007u, 0x00000000u, 0x00000000u},
			{"max.s32", 0xffffffffu, 0x7ffffff0u, 0x7fffffffu, 0x7fffffffu},
			{"max.u32", 0x00000001u, 0xfffffff0u, 0xffffffffu, 0xffffffffu},
			{"and.b32", 0xffffffffu, 0xffff0000u, 0x0000ffffu, 0x0000ffffu},
			{"or.b32", 0x00000000u, 0x00020000u, 0x00010000u, 0x00010000u},
			{"xor.b32", 0x00000000u, 0x00020000u, 0x00010000u, 0x00010000u},
		};
		for (const auto& test : cases) for (const char* destination : {"result", "a", "m"}) for (int size : {1, 31, 32}) {
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry delayed_redux() { .reg .u64 ptr; .reg .b32 lane,delay,result,a,m,special,round; .reg .pred p; "
				<< "mov.u32 lane,%tid.x; mov.u32 result,0xdeadbeef; mov.u32 round,2; again: mov.u32 m,0x7fffffff; "
				<< "setp.eq.u32 p,round,2; selp.u32 special," << test.first << "," << test.last << ",p; "
				<< "setp.eq.u32 p,lane,16; selp.u32 a,special," << test.base << ",p; setp.ge.u32 p,lane,16; selp.u32 delay,1,0,p; "
				<< "loop: setp.eq.u32 p,lane,31; @p bra done; setp.ne.u32 p,delay,0; @p bra later; redux.sync." << test.form << " " << destination << ",a,m; "
				<< "sub.u32 round,round,1; setp.ne.u32 p,round,0; @p bra again; bra done; later: sub.u32 delay,delay,1; bra loop; "
				<< "done: st.global.u32 [ptr]," << destination << "; ret; }";
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("delayed_redux"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (int run = 0; run < 2; ++run) {
				local.reset(); PTXU32 output[32] = {};
				for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::St)
					for (int lane = 0; lane < size; ++lane) local.setRegAsU64(lane, ins.d.reg, (PTXU64)&output[lane]);
				local.execute();
				for (int lane = 0; lane < size; ++lane) {
					const PTXU32 untouched = std::string(destination) == "result" ? 0xdeadbeef : std::string(destination) == "a" ? test.base : 0x7fffffff;
					const PTXU32 expected = lane == 31 ? untouched : size == 1 ? test.base : test.expected;
					if (output[lane] != expected) { status << "redux synchronization mismatch " << test.form << " " << destination << " CTA " << size << " lane " << lane << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_ReduxExecution() {
		const struct { const char* form; PTXU32 expected[4]; } cases[] = {
			{"add.s32", {0x7ffffff0u, 0x7fffffeeu, 0x80000000u, 0x7dfdfddeu}},
			{"add.u32", {0x7ffffff0u, 0x7fffffeeu, 0x80000000u, 0x7dfdfddeu}},
			{"min.s32", {0x80000000u, 0x88888887u, 0x81010101u, 0x89898986u}},
			{"min.u32", {0x11111110u, 0x1111110fu, 0x10101011u, 0x1010100eu}},
			{"max.s32", {0x77777776u, 0x7fffffffu, 0x76767677u, 0x7efefefeu}},
			{"max.u32", {0xfffffffeu, 0xffffffffu, 0xfefefeffu, 0xfefefefeu}},
			{"and.b32", {0x00000000u, 0x00000001u, 0x00000001u, 0x00000000u}},
			{"or.b32", {0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xfffffffeu}},
			{"xor.b32", {0x80000000u, 0x7fffffeeu, 0x80000000u, 0x7fffffeeu}},
		};
		for (const auto& test : cases) for (int alias : {0, 1, 2}) for (int scenario : {0, 1, 2}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry reduce() { .reg .b32 d,a,m; .reg .pred p; @!p redux.sync." +
				std::string(test.form) + " d,a,m; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("reduce"), nullptr);
			kernel.setKernelShape(64,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Redux) {
				if (alias) ins.d.reg = alias == 1 ? ins.a.reg : ins.b.reg;
				auto& context = local.getActiveContext(); PTXU32 initial[64];
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32;
					const PTXU32 value = (lane == 0 ? 0x80000000u : lane == 1 ? 0x7fffffffu : PTXU32(lane) * 0x11111111u) ^ (thread / 32 * 0x01010101u);
					local.setRegAsB32(thread, ins.d.reg, 0xdeadbeef);
					local.setRegAsB32(thread, ins.a.reg, value);
					local.setRegAsB32(thread, ins.b.reg, lane % 2 ? 0xaaaaaaaau : 0x55555555u);
					local.setRegAsPredicate(thread, ins.pg.reg, scenario == 2);
					context.active[thread] = scenario != 1 || lane % 2;
					initial[thread] = local.getRegAsB32(thread, ins.d.reg);
				}
				local.eval_WarpSync(context, ins);
				for (int thread = 0; thread < 64; ++thread) {
					const PTXU32 expected = scenario != 2 && context.active[thread] ? test.expected[(thread / 32) * 2 + thread % 2] : initial[thread];
					if (local.getRegAsB32(thread, ins.d.reg) != expected) { status << "redux mismatch " << test.form << " thread " << thread << " alias " << alias << " scenario " << scenario << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_ShflSyncSynchronization() {
		for (const char* mode : {"up", "down", "bfly", "idx"}) for (int size : {1, 31, 32})
		for (const char* destination : {"result", "a", "b", "c", "m", "_", "sinkp", "nonep"}) {
			const std::string name = mode, dest = destination;
			const std::string data = dest == "_" || dest == "sinkp" || dest == "nonep" ? "result" : dest;
			const std::string operands = dest == "sinkp" ? "result|_" : dest == "nonep" ? "result" : dest + "|q";
			const int offset = size == 1 ? 0 : 16, clamp = size == 1 || name == "up" ? 0 : 30;
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry delayed_shuffle() { .reg .u64 ptr,out,off; .reg .u32 lane,delay,result,a,b,c,m,round,predout,slot; .reg .pred p,q; "
				<< "mov.u32 lane,%tid.x; mov.u32 result,0xdeadbeef; mov.u32 predout,0xdeadbeef; mov.u32 round,2; "
				<< "again: mov.u32 b," << offset << "; mov.u32 c," << clamp << "; mov.u32 m," << (size == 1 ? "1" : "0x7fffffff") << "; mul.lo.u32 a,round,100; add.u32 a,a,lane; "
				<< "setp." << (name == "up" ? "lt" : "ge") << ".u32 p,lane,16; selp.u32 delay,1,0,p; "
				<< "loop: setp.eq.u32 p,lane,31; @p bra done; setp.ne.u32 p,delay,0; @p bra later; shfl.sync." << name << ".b32 " << operands << ",a,b,c,m; "
				<< (dest == "sinkp" || dest == "nonep" ? "" : "selp.u32 predout,1,0,q; ")
				<< "sub.u32 slot,2,round; mul.wide.u32 off,slot,8; add.u64 out,ptr,off; st.global.u32 [out]," << data << "; st.global.u32 [out+4],predout; "
				<< "sub.u32 round,round,1; setp.ne.u32 p,round,0; @p bra again; bra done; later: sub.u32 delay,delay,1; bra loop; done: ret; }";
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("delayed_shuffle"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (int run = 0; run < 2; ++run) {
				local.reset(); PTXU32 output[32][4];
				for (auto& row : output) for (auto& word : row) word = 0xdeadbeef;
				for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Add && ins.type == PTXOperand::u64)
					for (int lane = 0; lane < size; ++lane) local.setRegAsU64(lane, ins.a.reg, (PTXU64)&output[lane]);
				local.execute();
				for (int lane = 0; lane < size; ++lane) for (int round = 0; round < 2; ++round) {
					int from = lane; bool valid = true;
					if (size != 1) {
						if (name == "up") { valid = lane >= 16; if (valid) from = lane - 16; }
						if (name == "down") { valid = lane < 15; if (valid) from = lane + 16; }
						if (name == "bfly") { valid = lane != 15; if (valid) from = lane ^ 16; }
						if (name == "idx") from = 16;
					}
					const PTXU32 expected = lane == 31 || dest == "_" ? 0xdeadbeef : (2 - round) * 100 + from;
					const PTXU32 predicate = lane == 31 || dest == "sinkp" || dest == "nonep" ? 0xdeadbeef : valid;
					if (output[lane][round * 2] != expected || output[lane][round * 2 + 1] != predicate) { status << "shfl.sync waiting mismatch " << name << " " << dest << " CTA " << size << " lane " << lane << " round " << round << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_ShflSyncExecution() {
		// Frozen source lanes use digits 0..9,a..v; predicate bits describe in-range lanes.
		const struct { const char* mode; PTXU32 b, c; const char* sources; PTXU32 predicates; } cases[] = {
			{"up", 0x00000001u, 0x00000000u, "00123456789abcdefghijklmnopqrstu", 0xfffffffeu},
			{"up", 0x80000021u, 0x00001800u, "00123456889abcdegghijklmoopqrstu", 0xfefefefeu},
			{"up", 0x00000002u, 0x00000005u, "012345656789abcdefghijklmnopqrst", 0xffffff80u},
			{"down", 0x00000001u, 0x0000001fu, "123456789abcdefghijklmnopqrstuvv", 0x7fffffffu},
			{"down", 0x00000007u, 0x00001807u, "71234567f9abcdefnhijklmnvpqrstuv", 0x01010101u},
			{"down", 0x00000007u, 0x0000150au, "789abcde89abcdefnopqrstuopqrstuv", 0x00ff00ffu},
			{"bfly", 0x00000001u, 0x0000001fu, "1032547698badcfehgjilknmporqtsvu", 0xffffffffu},
			{"bfly", 0x80000010u, 0x00001807u, "0123456789abcdef0123456789abcdef", 0xffff0000u},
			{"bfly", 0x0000001fu, 0x0000150au, "0123456789abcdeffedcba9876543210", 0xffff0000u},
			{"idx", 0x80000021u, 0x0000181fu, "1111111199999999hhhhhhhhpppppppp", 0xffffffffu},
			{"idx", 0x0000001fu, 0x0000150au, "ababefefababefefqrqruvuvqrqruvuv", 0xffffffffu},
			{"idx", 0x0000000fu, 0x00000003u, "0123456789abcdefghijklmnopqrstuv", 0x00000000u},
		};
		const std::string alphabet = "0123456789abcdefghijklmnopqrstuv";
		for (int mode = 0; mode < 4; ++mode) for (int phase = 0; phase < 3; ++phase)
		for (int alias = 0; alias < 5; ++alias) for (bool disabled : {false, true}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry shuffle() { .reg .b32 d,a,b,c,m; .reg .pred p; @!p shfl.sync." +
				std::string(cases[mode * 3].mode) + ".b32 d|p,a,b,c,m; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("shuffle"), nullptr);
			kernel.setKernelShape(64,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::ShflSync) {
				if (alias) ins.d.reg = alias == 1 ? ins.a.reg : alias == 2 ? ins.b.reg : alias == 3 ? ins.c.reg : ins.q.reg;
				PTXU32 initial[64];
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32; const auto& test = cases[mode * 3 + (phase + lane) % 3];
					local.setRegAsB32(thread, ins.d.reg, 0xdeadbeef);
					local.setRegAsB32(thread, ins.a.reg, PTXU32(lane) * 0x11111111u ^ (thread / 32 * 0x01010101u));
					local.setRegAsU32(thread, ins.b.reg, test.b | 0x40000000u); local.setRegAsU32(thread, ins.c.reg, test.c | 0xe000e0e0u);
					local.setRegAsU32(thread, ins.q.reg, 0xffffffffu); local.setRegAsPredicate(thread, ins.pg.reg, disabled);
					initial[thread] = local.getRegAsB32(thread, ins.d.reg);
				}
				local.execute();
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32; const auto& test = cases[mode * 3 + (phase + lane) % 3];
					const PTXU32 from = alphabet.find(test.sources[lane]);
					const PTXU32 expected = disabled ? initial[thread] : (from * 0x11111111u) ^ (thread / 32 * 0x01010101u);
					if (local.getRegAsB32(thread, ins.d.reg) != expected || local.getRegAsPredicate(thread, ins.pq.reg) != (disabled || (test.predicates & (PTXU32(1) << lane)))) {
						status << "shfl.sync execution mismatch " << test.mode << " thread " << thread << " alias " << alias << "\n"; return false;
					}
				}
			}
		}
		return true;
	}

	bool test_ShflSyncParser() {
		const std::string prefix = ".version 8.0\n.target sm_86\n.entry shuffle() { .reg .b32 d,a,b,c,m; .reg .b64 w; .reg .f16x2 h; .reg .pred p; .reg .v2 .b32 v; mov.b32 m,0xffffffff; ";
		for (const char* mode : {"up", "down", "bfly", "idx"}) for (const char* guard : {"", "@!p "})
		for (const char* dest : {"d", "d|p", "d|_", "_|p"}) for (const char* inputs : {"a,b,c,m", "h,1,31,0xffffffff", "v.x,33,0x1807,-1", "7,WARP_SZ,31,WARP_SZ", "WARP_SZ,b,c,m"}) {
			const std::string operation = "shfl.sync." + std::string(mode) + ".b32 ";
			std::stringstream source(prefix + guard + operation + dest + "," + inputs + "; ret; }"), printed;
			Module parsed; parsed.load(source); parsed.write(printed); Module again; again.load(printed);
			EmulatedKernel kernel(again.getKernel("shuffle"), nullptr); int found = 0;
			for (const auto& ins : kernel.instructions) if (PTXInstruction::toString(ins.opcode) == "shfl.sync") {
				++found; if (!ins.valid().empty() || !ins.canObserveSideEffects()) return false;
				auto invalid = ins; invalid.modifier = PTXInstruction::sat; if (invalid.valid().empty()) return false;
				if (ins.q.addressMode == PTXOperand::Register) {
					for (const auto& definition : kernel.instructions) if (definition.opcode == PTXInstruction::Mov && definition.d.reg != ins.q.reg) return false;
					auto copy = ins; analysis::DataflowGraph graph; const auto converted = graph.convert(copy); bool maskFound = false;
					for (const auto& source : converted.s) maskFound |= source.pointer == &copy.q.reg;
					if (!maskFound) return false;
				}
			}
			if (found != 1) return false;
		}
		for (const char* operands : {"w,a,b,c,m", "d,w,b,c,m", "d,a,w,c,m", "d,a,b,w,m", "d,a,b,c,w", "d,a,b,c,0f3f800000", "d,a,0f3f800000,c,m", "d,a,b,0f3f800000,m", "d|!p,a,b,c,m", "d|w,a,b,c,m", "d,v,b,c,m"}) {
			std::stringstream source(prefix + "shfl.sync.idx.b32 " + operands + "; ret; }"); bool rejected = false;
			try { Module bad; bad.load(source); } catch (const std::exception&) { rejected = true; }
			if (!rejected) { status << "accepted malformed shfl.sync\n"; return false; }
		}
		for (const char* instruction : {"shfl.sync.idx.b64 d,a,b,c,m;", "shfl.sync.idx.b32 d,a,b,c;", "shfl.sync.bad.b32 d,a,b,c,m;", "shfl.sync.idx.b32 d,a,b,c,m,m;", "shfl.sync.idx.b32 d,a,b,c,v;"}) {
			std::stringstream source(prefix + instruction + " ret; }"); bool rejected = false;
			try { Module bad; bad.load(source); } catch (const std::exception&) { rejected = true; }
			if (!rejected) return false;
		}
		std::stringstream legacy(prefix + "shfl.up.b32 d,a,b,c; ret; }"), printed;
		Module parsed; parsed.load(legacy); parsed.write(printed); Module again; again.load(printed);
		return printed.str().find("shfl.up.b32") != std::string::npos;
	}

	bool test_WarpRegisters() {
		for (const auto shape : {Dim3(1,1,1), Dim3(31,1,1), Dim3(32,1,1), Dim3(33,1,1), Dim3(8,4,2), Dim3(7,5,2)})
		for (const char* name : {"WARP_SZ", "%laneid", "%lanemask_eq", "%lanemask_lt", "%lanemask_le", "%lanemask_gt", "%lanemask_ge"}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry warp_regs() { .reg .u32 d; .reg .pred p; @!p mov.u32 d," + std::string(name) + "; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("warp_regs"), nullptr);
			kernel.setKernelShape(shape.x,shape.y,shape.z); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Mov) for (int scenario = 0; scenario < 4; ++scenario) {
				local.reset(); auto& context = local.getActiveContext();
				for (int thread = 0; thread < local.threadCount; ++thread) {
					context.active[thread] = scenario != 1 || thread % 2;
					local.setRegAsPredicate(thread, ins.pg.reg, scenario == 3 || (scenario == 2 && thread % 3 == 0));
					local.setRegAsU32(thread, ins.d.reg, 0xdeadbeef);
				}
				local.eval_Mov(context, ins);
				for (int thread = 0; thread < local.threadCount; ++thread) {
					const int lane = thread % 32; const auto reg = ins.a.special;
					PTXU32 expected = reg == PTXOperand::warpSize ? 32 : reg == PTXOperand::laneId ? lane : 0;
					for (int bit = 0; bit < 32; ++bit) {
						const bool selected = reg == PTXOperand::lanemask_eq ? bit == lane : reg == PTXOperand::lanemask_lt ? bit < lane :
							reg == PTXOperand::lanemask_le ? bit <= lane : reg == PTXOperand::lanemask_gt ? bit > lane : reg == PTXOperand::lanemask_ge && bit >= lane;
						if (selected) expected |= PTXU32(1) << bit;
					}
					if (!context.predicated(thread, ins)) expected = 0xdeadbeef;
					if (local.getRegAsU32(thread, ins.d.reg) != expected) { status << "warp register mismatch " << name << " thread " << thread << " CTA " << local.threadCount << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_BarWarpSynchronization() {
		for (int size : {1, 31, 32, 64}) for (bool delayed : {false, true}) for (bool registerMask : {false, true}) {
			if (delayed && size > 32) continue;
			const int supplier = size == 1 ? 0 : 16;
			const std::string mask = size == 1 ? "1" : "0x7fffffff", operand = registerMask ? "m" : mask;
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry warp_barrier() { .shared .align 4 .u32 data[64]; .reg .u64 ptr; .reg .u32 tid,lane,addr,readaddr,delay,result,round,value,m; .reg .pred p; "
				<< "mov.u32 tid,%tid.x; and.b32 lane,tid,31; mul.lo.u32 addr,tid,4; sub.u32 readaddr,tid,lane; add.u32 readaddr,readaddr," << supplier << "; mul.lo.u32 readaddr,readaddr,4; "
				<< "mov.u32 result,0xdeadbeef; mov.u32 round,2; mov.u32 m," << mask << "; again: mul.lo.u32 value,round,100; add.u32 value,value,tid; "
				<< "setp.ge.u32 p,lane,16; selp.u32 delay," << (delayed ? 1 : 0) << ",0,p; loop: setp.eq.u32 p,lane,31; @p bra done; setp.ne.u32 p,delay,0; @p bra later; "
				<< "st.shared.u32 [addr],value; bar.warp.sync " << operand << "; ld.shared.u32 result,[readaddr]; bar.warp.sync " << operand << "; "
				<< "sub.u32 round,round,1; setp.ne.u32 p,round,0; @p bra again; bra done; later: sub.u32 delay,delay,1; bra loop; done: st.global.u32 [ptr],result; ret; }";
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("warp_barrier"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (int run = 0; run < 2; ++run) {
				local.reset(); PTXU32 output[64] = {};
				std::memset(local.functionCallStack.sharedMemoryPointer(), 0, 256);
				for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::St && ins.addressSpace == PTXInstruction::Global)
					for (int thread = 0; thread < size; ++thread) local.setRegAsU64(thread, ins.d.reg, (PTXU64)&output[thread]);
				local.execute();
				for (int thread = 0; thread < size; ++thread) {
					const PTXU32 expected = thread % 32 == 31 ? 0xdeadbeef : 100 + thread / 32 * 32 + supplier;
					if (output[thread] != expected) { status << "warp barrier shared-memory mismatch CTA " << size << " delayed " << delayed << " thread " << thread << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_BarWarpParser() {
		const std::string prefix = ".version 8.0\n.target sm_86\n.entry warp_barrier() { .reg .u32 m; .reg .b32 b; .reg .s32 s; .reg .b64 w; .reg .f32 f; .reg .pred p; .reg .v2 .u32 v; ";
		for (const char* mask : {"m", "b", "s", "0xffffffff", "0", "-1", "v.x"}) for (const char* guard : {"", "@p ", "@!p "}) {
			std::stringstream source(prefix + guard + "bar.warp.sync " + mask + "; ret; }"), printed;
			Module parsed; parsed.load(source); parsed.write(printed); Module again; again.load(printed);
			EmulatedKernel kernel(again.getKernel("warp_barrier"), nullptr); int found = 0;
			for (const auto& ins : kernel.instructions) if (PTXInstruction::toString(ins.opcode) == "bar.warp.sync") {
				++found; if (!ins.valid().empty() || !ins.hasSideEffects() || !ins.canObserveSideEffects()) return false;
				auto invalid = ins; invalid.modifier = PTXInstruction::sat; if (invalid.valid().empty()) return false;
			}
			if (found != 1) return false;
		}
		for (const char* instruction : {"bar.warp.sync w;", "bar.warp.sync f;", "bar.warp.sync p;", "bar.warp.sync v;", "bar.warp.sync 0f3f800000;", "bar.warp.sync;", "bar.warp.sync m,m;", "bar.warp.sync.u32 m;", "bar.warp.arrive m;", "bar.warp m;"}) {
			std::stringstream source(prefix + instruction + " ret; }"); bool rejected = false;
			try { Module bad; bad.load(source); } catch (const std::exception&) { rejected = true; }
			if (!rejected) { status << "accepted invalid warp barrier\n"; return false; }
		}
		return true;
	}

	bool test_MatchSynchronization() {
		for (bool all : {false, true}) for (bool wide : {false, true}) for (int size : {1, 31, 32})
		for (const char* destination : {"result", "a", "m", "_", "sinkp"}) {
			const std::string dest = destination;
			if ((wide && dest == "a") || (!all && (dest == "_" || dest == "sinkp"))) continue;
			const std::string outputReg = dest == "_" || dest == "sinkp" ? "result" : dest;
			const std::string operands = !all ? dest : dest == "sinkp" ? "result|_" : dest + "|q";
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry delayed_match() { .reg .u64 ptr; .reg .b64 w; .reg .b32 lane,delay,result,a,m,round,predout; .reg .pred p,q; "
				<< "mov.u32 lane,%tid.x; mov.u32 result,0xdeadbeef; mov.u32 predout,0xdeadbeef; mov.u32 round,2; again: mov.u32 m,0x7fffffff; "
				<< "setp.ge.u32 p,lane,16; selp.u32 delay,1,0,p; mov.b32 a,0; mov.b64 w,0; setp.eq.u32 p,lane,16; @p mov.b32 a,round; "
				<< "setp.eq.u32 p,round,1; @p mov.b32 a,0; cvt.u64.u32 w,a; shl.b64 w,w,32; "
				<< "loop: setp.eq.u32 p,lane,31; @p bra done; setp.ne.u32 p,delay,0; @p bra later; match." << (all ? "all" : "any") << ".sync." << (wide ? "b64 " : "b32 ")
				<< operands << "," << (wide ? "w" : "a") << ",m; " << (all && dest != "sinkp" ? "selp.u32 predout,1,0,q; " : "")
				<< "sub.u32 round,round,1; setp.ne.u32 p,round,0; @p bra again; bra done; later: sub.u32 delay,delay,1; bra loop; "
				<< "done: st.global.u32 [ptr]," << outputReg << "; st.global.u32 [ptr+4],predout; ret; }";
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("delayed_match"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (int run = 0; run < 2; ++run) {
				local.reset(); PTXU32 output[32][2] = {};
				for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::St)
					for (int lane = 0; lane < size; ++lane) local.setRegAsU64(lane, ins.d.reg, (PTXU64)&output[lane]);
				local.execute();
				for (int lane = 0; lane < size; ++lane) {
					const PTXU32 members = size == 1 ? 1 : 0x7fffffffu;
					const PTXU32 expected = lane == 31 ? (dest == "a" ? 0 : dest == "m" ? 0x7fffffffu : 0xdeadbeef) : dest == "_" ? 0xdeadbeef : members;
					const PTXU32 predicate = lane == 31 || !all || dest == "sinkp" ? 0xdeadbeef : 1;
					if (output[lane][0] != expected || output[lane][1] != predicate) { status << "match synchronization mismatch " << all << " " << wide << " " << dest << " CTA " << size << " lane " << lane << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_MatchExecution() {
		for (bool all : {false, true}) for (bool wide : {false, true}) for (int size : {1, 31, 64})
		for (int alias : {0, 1, 2}) for (int scenario : {0, 1, 2, 3}) {
			if (wide && alias == 1) continue;
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matching() { .reg .b32 d,a,m; .reg .b64 w; .reg .pred p; @!p match." +
				std::string(all ? "all" : "any") + ".sync." + (wide ? "b64 " : "b32 ") + (all ? "d|p," : "d,") + (wide ? "w" : "a") + ",m; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("matching"), nullptr);
			kernel.setKernelShape(size,1,1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Match) {
				if (alias) ins.d.reg = alias == 1 ? ins.a.reg : ins.b.reg;
				auto& context = local.getActiveContext(); PTXU32 initial[64];
				for (int thread = 0; thread < size; ++thread) {
					const int lane = thread % 32;
					const PTXU64 value = scenario == 1 ? lane % 2 : wide ? (PTXU64(lane % 4) << 32) | (lane % 2) : lane % 4;
					local.setRegAsB32(thread, ins.d.reg, 0xdeadbeef);
					if (wide) local.setRegAsB64(thread, ins.a.reg, value); else local.setRegAsB32(thread, ins.a.reg, value);
					local.setRegAsB32(thread, ins.b.reg, lane % 2 ? 0xaaaaaaaau : 0x55555555u);
					local.setRegAsPredicate(thread, ins.pg.reg, scenario == 3);
					context.active[thread] = scenario != 2 || lane % 2;
					initial[thread] = local.getRegAsB32(thread, ins.d.reg);
				}
				local.eval_WarpSync(context, ins);
				for (int thread = 0; thread < size; ++thread) {
					const int lane = thread % 32;
					const bool executing = scenario != 3 && (scenario != 2 || lane % 2), uniform = scenario == 1 || size == 1;
					const PTXU32 existing = size == 1 ? 1 : size == 31 ? 0x7fffffffu : 0xffffffffu;
					const PTXU32 members = (lane % 2 ? 0xaaaaaaaau : 0x55555555u) & existing;
					const PTXU32 expected = !executing ? initial[thread] : all ? (uniform ? members : 0) : (scenario == 1 ? members : (0x11111111u << (lane % 4)) & existing);
					if (local.getRegAsB32(thread, ins.d.reg) != expected || (all && local.getRegAsPredicate(thread, ins.pq.reg) != (executing ? uniform : scenario == 3))) {
						status << "match execution mismatch at thread " << thread << "\n"; return false;
					}
				}
			}
		}
		return true;
	}

	bool test_MatchParser() {
		const std::string prefix = ".version 8.0\n.target sm_86\n.entry matching() { .reg .b32 d,a,m; .reg .b64 w; .reg .pred p; .reg .f16x2 h; ";
		for (const char* mode : {"any", "all"}) for (const char* type : {"b32", "b64"})
		for (const char* mask : {"m", "0xffffffff", "-1", "0"}) for (const char* guard : {"", "@!p "}) {
			const std::string operation = "match." + std::string(mode) + ".sync." + type + " ";
			const std::string input = std::string(type) == "b64" ? "w" : "a";
			for (const char* dest : {"d", "d|p", "_|p", "d|_"}) {
				if (std::string(mode) == "any" && std::string(dest) != "d") continue;
				std::stringstream source(prefix + guard + operation + dest + "," + input + "," + mask + "; ret; }"), printed;
				Module parsed; parsed.load(source); parsed.write(printed); Module again; again.load(printed);
				EmulatedKernel kernel(again.getKernel("matching"), nullptr); int found = 0;
				for (const auto& ins : kernel.instructions) if (PTXInstruction::toString(ins.opcode) == "match") {
					++found; if (!ins.valid().empty() || !ins.canObserveSideEffects()) return false;
					auto invalid = ins; invalid.modifier = PTXInstruction::sat; if (invalid.valid().empty()) return false;
				}
				if (found != 1) return false;
			}
			for (const char* operands : {"w,a,m", "d,a,w", "d,a,0f3f800000", "d|w,a,m", "d|!p,a,m", "_|_,a,m"}) {
				std::stringstream source(prefix + operation + operands + "; ret; }"); bool rejected = false;
				try { Module bad; bad.load(source); } catch (const std::exception&) { rejected = true; }
				if (!rejected) { status << "accepted invalid match operands\n"; return false; }
			}
		}
		for (const char* instruction : {"match.any.b32 d,a,m;", "match.any.sync.b16 d,a,m;", "match.uni.sync.b32 d,a,m;", "match.any.sync.b32 d|p,a,m;", "match.any.sync.b32 d,a;", "match.all.sync.b64 d,a,m;"}) {
			std::stringstream source(prefix + instruction + " ret; }"); bool rejected = false;
			try { Module bad; bad.load(source); } catch (const std::exception&) { rejected = true; }
			if (!rejected) { status << "accepted invalid match form\n"; return false; }
		}
		return true;
	}

	bool test_ReduxParser() {
		const std::string prefix = ".version 8.0\n.target sm_86\n.entry redux_parse() { .reg .b32 d,a,m; .reg .b64 w; .reg .pred p; .reg .v2 .b32 v; ";
		for (const char* form : {"add.s32", "add.u32", "min.s32", "min.u32", "max.s32", "max.u32", "and.b32", "or.b32", "xor.b32"})
		for (const char* mask : {"m", "0xffffffff", "-1", "0"}) for (const char* guard : {"", "@!p "})
		for (const char* registers : {"d,a,", "v.y,v.x,"}) {
			const std::string operation = "redux.sync." + std::string(form) + " ";
			std::stringstream source(prefix + guard + operation + registers + mask + "; ret; }"), printed;
			Module parsed;
			try { parsed.load(source); } catch (const std::exception& error) { status << "redux parse failed: " << error.what() << "\n"; return false; }
			parsed.write(printed); Module roundtrip; roundtrip.load(printed);
			EmulatedKernel kernel(roundtrip.getKernel("redux_parse"), nullptr);
			int found = 0;
			for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Redux) {
				++found;
				if (!ins.valid().empty() || !ins.canObserveSideEffects()) return false;
				auto invalid = ins; invalid.modifier = PTXInstruction::sat;
				if (invalid.valid().empty()) return false;
			}
			if (found != 1) return false;
			for (const char* operands : {"w,a,m", "d,w,m", "d,a,w", "d,a,0f3f800000", "d,1,m", "v,a,m", "d,v,m", "d,a,v"}) {
				std::stringstream invalid(prefix + operation + operands + "; ret; }"); bool rejected = false;
				try { Module bad; bad.load(invalid); } catch (const std::exception&) { rejected = true; }
				if (!rejected) { status << "accepted malformed redux operands\n"; return false; }
			}
		}
		for (const char* form : {"add.b32", "min.b32", "max.f32", "add.u64", "and.u32", "or.s32", "inc.u32", "dec.u32"}) {
			std::stringstream invalid(prefix + "redux.sync." + form + " d,a,m; ret; }"); bool rejected = false;
			try { Module bad; bad.load(invalid); } catch (const std::exception&) { rejected = true; }
			if (!rejected) { status << "accepted illegal redux operation/type\n"; return false; }
		}
		for (const char* instruction : {"redux.add.u32 d,a,m;", "redux.sync.add.u32 d,a;", "redux.sync.add.sat.u32 d,a,m;"}) {
			std::stringstream invalid(prefix + instruction + " ret; }"); bool rejected = false;
			try { Module bad; bad.load(invalid); } catch (const std::exception&) { rejected = true; }
			if (!rejected) return false;
		}
		for (const char* operation : {"and", "or", "xor"}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry packed() { .reg .f16x2 d,a; redux.sync." +
				std::string(operation) + ".b32 d,a,0xffffffff; ret; }");
			Module parsed;
			try { parsed.load(source); } catch (const std::exception& error) { status << "packed redux container: " << error.what() << "\n"; return false; }
		}
		return true;
	}

	bool test_VoteSyncParser() {
		for (const char* mode : {"all", "any", "uni", "ballot"}) for (const char* mask : {"0xffffffff", "m", "-1", "0"})
		for (bool invert : {false, true}) {
			const bool ballot = std::string(mode) == "ballot";
			const std::string prefix = ".version 8.0\n.target sm_86\n.entry vote_parser() { .reg .pred a, d, g; .reg .b32 r, m; .reg .b64 wide; ";
			const std::string operation = "vote.sync." + std::string(mode) + (ballot ? ".b32 r, " : ".pred d, ");
			const std::string sourcePredicate = invert ? "!a" : "a";
			const std::string text = prefix + "mov.b32 m, 0xffffffff; vote.any.pred d, a; @!g " + operation +
				sourcePredicate + ", " + mask + "; vote.uni.pred d, a; ret; }";
			std::stringstream source(text), printed; Module parsed; parsed.load(source); parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("vote_parser"), nullptr);
			int found = 0, legacy = 0, maskRegister = -1;
			for (const auto& ins : kernel.instructions) {
				if (ins.opcode == PTXInstruction::Mov) maskRegister = ins.d.reg;
				if (ins.opcode != PTXInstruction::Vote) continue;
				if (!ins.voteSync) { ++legacy; continue; }
				++found;
				if (!ins.valid().empty() || !ins.canObserveSideEffects() || ins.b.bytes() != 4 ||
					ins.pg.condition != PTXOperand::InvPred || (ins.a.condition == PTXOperand::InvPred) != invert ||
					(std::string(mask) == "m" && int(ins.b.reg) != maskRegister)) return false;
			}
			if (found != 1 || legacy != 2) return false;
			for (const std::string& bad : {operation + sourcePredicate + ", wide;", operation + sourcePredicate + ", 0f3f800000;",
				operation + "%laneid, " + mask + ";", operation + sourcePredicate + ";",
				std::string("vote.sync.any.pred !d, a, 0xffffffff;"),
				"vote.sync." + std::string(mode) + (ballot ? ".pred d, " : ".b32 r, ") + sourcePredicate + ", " + mask + ";"}) {
				std::stringstream input(prefix + bad + " ret; }"); bool rejected = false;
				try { Module invalid; invalid.load(input); } catch (const std::exception&) { rejected = true; }
				if (!rejected) { status << "accepted invalid vote.sync: " << bad << "\n"; return false; }
			}
		}
		for (bool invert : {false, true}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry vote_constant() { .reg .pred d; vote.sync.any.pred d, " +
				std::string(invert ? "!%pt" : "%pt") + ", 0xffffffff; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("vote_constant"), nullptr);
			for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Vote &&
				ins.a.condition != (invert ? PTXOperand::nPT : PTXOperand::PT)) {
				status << "vote.sync predicate constant lost its value\n"; return false;
			}
		}
		return true;
	}

	bool test_Activemask() {
		for (const char* type : {"b32", "u32", "s32", "f32"}) {
			const std::string text = ".version 8.0\n.target sm_86\n.entry active_mask() { .reg ." + std::string(type) +
				" r; .reg .pred p; @p activemask.b32 r; ret; }";
			std::stringstream source(text), printed; Module parsed; parsed.load(source); parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("active_mask"), nullptr);
			int found = 0;
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Activemask) {
				++found;
				if (!ins.canObserveSideEffects() || ins.isMemoryInstruction()) return false;
				for (int size : {1, 31, 32, 33, 64, 65}) for (int scenario = 0; scenario < 5; ++scenario) {
					kernel.setKernelShape(size, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
					auto& context = local.getActiveContext();
					ins.pg.condition = scenario == 3 ? PTXOperand::InvPred : PTXOperand::Pred;
					for (int thread = 0; thread < size; ++thread) {
						const PTXU32 bit = PTXU32(1) << (thread % 32);
						const PTXU32 active = scenario == 1 ? (thread < 32 ? 0x80000081u : 0x40000202u) : ~PTXU32(0);
						const PTXU32 predicate = scenario == 4 ? 0 : scenario >= 2 ? 0x55555555u : ~PTXU32(0);
						context.active[thread] = (active & bit) != 0;
						local.setRegAsPredicate(thread, ins.pg.reg, scenario == 3 ? !(predicate & bit) : bool(predicate & bit));
						local.setRegAsB32(thread, ins.d.reg, 0xdeadbeef);
					}
					local.eval_Activemask(context, ins);
					for (int thread = 0; thread < size; ++thread) {
						const int warp = thread / 32, count = std::min(32, size - warp * 32);
						PTXU32 mask = ~PTXU32(0) >> (32 - count);
						if (scenario == 1) mask &= warp == 0 ? 0x80000081u : 0x40000202u;
						if (scenario >= 2) mask &= scenario == 4 ? 0 : 0x55555555u;
						const PTXU32 expected = mask & (PTXU32(1) << (thread % 32)) ? mask : 0xdeadbeef;
						if (local.getRegAsB32(thread, ins.d.reg) != expected) { status << "activemask mismatch CTA " << size << " scenario " << scenario << " thread " << thread << "\n"; return false; }
					}
				}
			}
			if (found != 1) return false;
			for (const auto& change : {std::make_pair(std::string("activemask.b32 r"), std::string("activemask.b64 r")),
				std::make_pair(std::string(type) + " r;", std::string("b64 r;")),
				std::make_pair(std::string("activemask.b32 r"), std::string("activemask.b32 0")),
				std::make_pair(std::string("activemask.b32 r"), std::string("activemask.b32 {r}"))}) {
				std::string bad = text; bad.replace(bad.find(change.first), change.first.size(), change.second);
				std::stringstream input(bad); bool rejected = false;
				try { Module invalid; invalid.load(input); } catch (const std::exception&) { rejected = true; }
				if (!rejected) return false;
			}
		}
		return true;
	}

	bool test_Movmatrix() {
		for (bool alias : {false, true}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_transpose() { .reg .b32 a, d; "
				"movmatrix.sync.aligned.m8n8.trans.b16 d, a; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("matrix_transpose"), nullptr);
			kernel.setKernelShape(64, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Movmatrix) {
				if (alias) ins.d.reg = ins.a.reg;
				PTXU32 original[64] = {}, expected[64] = {};
				// Construct the transpose oracle by assigning each logical input element to its output slot.
				for (int warp = 0; warp < 2; ++warp) for (int row = 0; row < 8; ++row) for (int col = 0; col < 8; ++col) {
					const PTXU32 value = 0x8001 + warp * 1024 + row * 8 + col;
					original[warp * 32 + row * 4 + col / 2] |= value << (16 * (col % 2));
					expected[warp * 32 + col * 4 + row / 2] |= value << (16 * (row % 2));
				}
				for (int thread = 0; thread < 64; ++thread) local.setRegAsB32(thread, ins.a.reg, original[thread]);
				for (int pass = 0; pass < 2; ++pass) {
					local.eval_Movmatrix(local.getActiveContext(), ins);
					for (int thread = 0; thread < 64; ++thread) {
						if (local.getRegAsB32(thread, ins.d.reg) != (pass ? original[thread] : expected[thread])) {
							status << "movmatrix transpose mismatch thread " << thread << " pass " << pass << " alias " << alias << "\n"; return false;
						}
					}
					ins.a = ins.d;
				}
			}
		}
		return true;
	}

	bool test_MovmatrixControls() {
		std::stringstream source(".version 8.0\n.target sm_86\n.entry transpose_controls() { .reg .b32 a, d; .reg .pred p; "
			"@p movmatrix.sync.aligned.m8n8.trans.b16 d, a; ret; }");
		Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("transpose_controls"), nullptr);
		for (int size : {1, 31, 32, 33, 64, 65}) {
			kernel.setKernelShape(size, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
			for (auto original : kernel.instructions) if (original.opcode == PTXInstruction::Movmatrix)
			for (int scenario = 0; scenario < 12; ++scenario) {
				const int complete = (size / 32) * 32;
				if (!complete && (scenario == 3 || scenario == 4 || scenario == 8)) continue;
				auto ins = original; local.reset(); auto& context = local.getActiveContext();
				if (scenario == 6) ins.pg.condition = PTXOperand::InvPred;
				if (scenario == 7) ins.d.type = PTXOperand::b64;
				if (scenario == 9) ins.a.addressMode = PTXOperand::Immediate;
				if (scenario == 10) ins.type = PTXOperand::b32;
				if (scenario == 11) ins.matrixTranspose = false;
				for (int thread = 0; thread < size; ++thread) {
					local.setRegAsB32(thread, original.a.reg, 0x80018001u + (thread / 32) * 0x10001000u);
					local.setRegAsB32(thread, original.d.reg, 0xdeadbeef);
					bool predicate = scenario == 0 || scenario == 2 || thread < complete;
					if ((scenario == 3 && thread == 0) || (scenario == 4 && thread < 32) || scenario == 5) predicate = false;
					local.setRegAsPredicate(thread, ins.pg.reg, scenario == 6 ? !predicate : predicate);
					if ((scenario == 2 && thread >= complete) || (scenario == 8 && thread == 0)) context.active.reset(thread);
				}
				bool rejected = false;
				try { local.eval_Movmatrix(context, ins); } catch (const RuntimeException&) { rejected = true; }
				const bool shouldReject = (scenario == 0 && size % 32) || scenario == 3 || scenario == 7 || scenario >= 8;
				if (rejected != shouldReject) {
					status << "movmatrix control mismatch CTA " << size << " scenario " << scenario << "\n"; return false;
				}
				for (int thread = 0; thread < size; ++thread) {
					const PTXU32 expected = !rejected && context.predicated(thread, ins)
						? 0x80018001u + (thread / 32) * 0x10001000u : 0xdeadbeef;
					if (local.getRegAsB32(thread, original.d.reg) != expected) {
						status << "movmatrix destination mismatch CTA " << size << " scenario " << scenario << " thread " << thread << "\n"; return false;
					}
				}
			}
		}
		return true;
	}

	bool test_MovmatrixParser() {
		for (const char* type : {"b32", "u32", "s32", "f32"}) {
			const std::string text = ".version 8.0\n.target sm_86\n.entry transpose_parser() { .reg ." + std::string(type) +
				" a, d; movmatrix.sync.aligned.m8n8.trans.b16 d, a; ret; }";
			std::stringstream source(text), printed; Module parsed; parsed.load(source); parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("transpose_parser"), nullptr);
			int found = 0;
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Movmatrix) {
				++found;
				if (!ins.valid().empty() || !ins.canObserveSideEffects() || ins.isMemoryInstruction()) return false;
				ins.pg.type = PTXOperand::pred; ins.pg.condition = PTXOperand::InvPred; ins.pg.identifier = "p";
				if (ins.guard() != "@!p ") return false;
				ins.pg.identifier.clear(); ins.pg.reg = 7;
				if (ins.guard() != "@!%p7 ") return false;
				ins.modifier = PTXInstruction::ftz; if (ins.valid().empty()) return false;
			}
			if (found != 1) return false;
			for (const auto& change : {std::make_pair(std::string(".sync"), std::string("")),
				std::make_pair(std::string(".aligned"), std::string("")),
				std::make_pair(std::string(".trans"), std::string("")),
				std::make_pair(std::string(".m8n8"), std::string(".m16n16")),
				std::make_pair(std::string(".b16 d"), std::string(".b32 d")),
				std::make_pair(std::string(" d, a;"), std::string(" d, 0;")),
				std::make_pair(std::string(" d, a;"), std::string(" {d}, a;")),
				std::make_pair(std::string(" d, a;"), std::string(" d, %laneid;")),
				std::make_pair(std::string(type) + " a", std::string("b64 a"))}) {
				std::string bad = text; bad.replace(bad.find(change.first), change.first.size(), change.second);
				std::stringstream input(bad); bool rejected = false;
				try { Module invalid; invalid.load(input); } catch (const std::exception&) { rejected = true; }
				if (!rejected) { status << "accepted invalid movmatrix: " << bad << "\n"; return false; }
			}
		}
		for (bool alias : {false, true}) {
			const std::string destination = alias ? "v.x" : "v.y";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry transpose_components() { .reg .v2 .b32 v; .reg .pred p; "
				"mov.b32 v.x, 0; mov.b32 v.y, 0; @!p movmatrix.sync.aligned.m8n8.trans.b16 " + destination + ", v.x; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("transpose_components"), nullptr);
			int components[2] = {-1, -1}, count = 0;
			for (const auto& ins : kernel.instructions) {
				if (ins.opcode == PTXInstruction::Mov && count < 2) components[count++] = ins.d.reg;
				if (ins.opcode == PTXInstruction::Movmatrix && (count != 2 || components[0] == components[1] ||
					int(ins.a.reg) != components[0] || int(ins.d.reg) != components[alias ? 0 : 1] ||
					ins.pg.condition != PTXOperand::InvPred)) {
					status << "movmatrix component assignment mismatch alias " << alias << "\n"; return false;
				}
			}
		}
		return true;
	}

	bool test_LdmatrixParser() {
		for (int count : {1, 2, 4}) for (bool transpose : {false, true})
		for (const char* space : {"", ".shared", ".shared::cta"}) for (const char* width : {"32", "64"}) {
			std::string fragment = "{r0";
			for (int i = 1; i < count; ++i) fragment += ",r" + std::to_string(i);
			fragment += "}";
			const std::string text = ".version 8.0\n.target sm_86\n.entry matrix_parser() { .reg .b32 r<4>; .reg .b" + std::string(width) +
				" addr; mov.b" + width + " addr, 0; ldmatrix.sync.aligned.m8n8.x" + std::to_string(count) +
				(transpose ? ".trans" : "") + space + ".b16 " + fragment + ", [addr+16]; ret; }";
			std::stringstream source(text), printed; Module parsed; parsed.load(source); parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("matrix_parser"), nullptr);
			int addressRegister = -1;
			for (const auto& ins : kernel.instructions) {
				if (ins.opcode == PTXInstruction::Mov) addressRegister = ins.d.reg;
				if (ins.opcode != PTXInstruction::Ldmatrix) continue;
				if (int(ins.vec) != count || ins.matrixTranspose != transpose ||
					int(ins.a.reg) != addressRegister || ins.a.offset != 16 || !ins.valid().empty()) return false;
			}
			for (const auto& change : {std::make_pair(std::string(".b16 "), std::string(".b8 ")),
				std::make_pair(std::string(".sync"), std::string("")),
				std::make_pair(std::string(".aligned"), std::string("")),
				std::make_pair(std::string(".b32 r"), std::string(".b64 r"))}) {
				std::string bad = text; bad.replace(bad.find(change.first), change.first.size(), change.second);
				std::stringstream input(bad); bool rejected = false;
				try { Module invalid; invalid.load(input); } catch (const std::exception&) { rejected = true; }
				if (!rejected) return false;
			}
		}
		for (const char* address : {"16", "smem+16"}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_symbol() { .shared .align 16 .b8 smem[256]; .reg .b32 r; "
				"ldmatrix.sync.aligned.m8n8.x1.b16 {r}, [" + std::string(address) + "]; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel kernel(roundtrip.getKernel("matrix_symbol"), nullptr);
			for (const auto& ins : kernel.instructions) if (ins.opcode == PTXInstruction::Ldmatrix &&
				(!ins.valid().empty() || (address[0] == 's' && ins.addressSpace != PTXInstruction::Shared))) return false;
		}
		return true;
	}

	bool test_LdmatrixAddress() {
		for (const char* address : {"addr.x", "addr.y+16", "addr.x-2147483648", "addr.x - 2147483648", "addr.x+2147483647", "4294967295", "0"}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_address() { .reg .v2 .b64 addr; .reg .b32 d; "
				"ldmatrix.sync.aligned.m8n8.x1.shared.b16 {d}, [" + std::string(address) + "]; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed);
		}
		for (const char* address : {"addr", "addr.z", "addr.x+2147483648", "addr.x-2147483649",
			"addr.x+4294967296", "4294967296", "-1", "0f3f800000", "addr.x 16"}) {
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_address() { .reg .v2 .b64 addr; .reg .b32 d; "
				"ldmatrix.sync.aligned.m8n8.x1.shared.b16 {d}, [" + std::string(address) + "]; ret; }");
			bool rejected = false;
			try { Module invalid; invalid.load(source); } catch (const std::exception&) { rejected = true; }
			if (!rejected) return false;
		}
		return true;
	}

	bool test_Ldmatrix() {
		for (int count : {1, 2, 4}) for (bool transpose : {false, true}) for (bool generic : {false, true})
		for (bool address32 : {false, true}) for (bool alias : {false, true}) {
			if (alias && !address32) continue;
			std::string fragment = "{r0";
			for (int i = 1; i < count; ++i) fragment += ",r" + std::to_string(i);
			std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_load() { .shared .align 16 .b16 storage[520]; "
				".reg .b32 r<4>; .reg .b" + std::string(address32 ? "32" : "64") + " addr; ldmatrix.sync.aligned.m8n8.x" + std::to_string(count) +
				(transpose ? ".trans" : "") + (generic ? "" : ".shared") + ".b16 " + fragment + "}, [addr+16]; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("matrix_load"), nullptr);
			kernel.setKernelShape(64, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
			auto* memory = static_cast<PTXU16*>(local.functionCallStack.sharedMemoryPointer());
			auto value = [](int warp, int matrix, int row, int col) { return PTXU32(0x8001 + warp * 0x4000 + matrix * 0x800 + row * 32 + col); };
			for (int warp = 0; warp < 2; ++warp) for (int m = 0; m < 4; ++m)
			for (int row = 0; row < 8; ++row) for (int col = 0; col < 8; ++col)
				memory[8 + (warp * 4 + m) * 64 + ((row * 3 + 5) % 8) * 8 + col] = value(warp, m, row, col);
			for (auto ins : kernel.instructions) if (ins.opcode == PTXInstruction::Ldmatrix) {
				if (alias) ins.d.array[0].reg = ins.a.reg;
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32, warp = thread / 32;
					PTXU64 address = (warp * 4 + lane / 8) * 128 + ((lane % 8 * 3 + 5) % 8) * 16;
					if (generic) address += reinterpret_cast<PTXU64>(memory);
					if (lane >= count * 8) address = ~PTXU64(0);
					if (address32) local.setRegAsU32(thread, ins.a.reg, PTXU32(address));
					else local.setRegAsU64(thread, ins.a.reg, address);
				}
				try { local.eval_Ldmatrix(local.getActiveContext(), ins); }
				catch (const RuntimeException& error) { status << error.what() << "\n"; return false; }
				for (int thread = 0; thread < 64; ++thread) for (int m = 0; m < count; ++m) {
					const int lane = thread % 32, row = lane / 4, col = (lane % 4) * 2;
					PTXU32 expected = 0;
					for (int j = 0; j < 2; ++j) expected |= value(thread / 32, m, transpose ? col + j : row, transpose ? row : col + j) << (16 * j);
					if (local.getRegAsB32(thread, ins.d.array[m].reg) != expected) {
						status << "ldmatrix mismatch x" << count << " trans " << transpose << " thread " << thread << " register " << m << "\n"; return false;
					}
				}
			}
		}
		return true;
	}

	bool test_LdmatrixControls() {
		std::stringstream source(".version 8.0\n.target sm_86\n.entry matrix_controls() { .shared .align 16 .b16 storage[512]; "
			".reg .b32 r<4>; .reg .b64 addr; .reg .pred p; @p ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr]; ret; }");
		Module parsed; parsed.load(source); EmulatedKernel kernel(parsed.getKernel("matrix_controls"), nullptr);
		for (int size : {1, 31, 32, 33, 64, 65}) {
			kernel.setKernelShape(size, 1, 1); CooperativeThreadArray local(&kernel, Dim3(), false);
			auto* memory = static_cast<PTXU16*>(local.functionCallStack.sharedMemoryPointer());
			std::fill(memory, memory + 512, 0);
			for (auto original : kernel.instructions) if (original.opcode == PTXInstruction::Ldmatrix)
			// Active/partial CTA, predicated/inactive tails, divergence, inactive warp,
			// bounds, generic-window and IR errors, all-off, misalignment, inverted guard.
			for (int scenario = 0; scenario < 11; ++scenario) {
				const int complete = size / 32 * 32;
				if (complete == 0 && ((scenario >= 3 && scenario <= 7) || scenario == 9)) continue;
				auto ins = original; local.reset(); auto& context = local.getActiveContext();
				if (scenario == 6) ins.addressSpace = PTXInstruction::Generic;
				if (scenario == 7) ins.d.array[0].type = PTXOperand::b64;
				if (scenario == 10) ins.pg.condition = PTXOperand::InvPred;
				for (int thread = 0; thread < size; ++thread) {
					for (const auto& d : ins.d.array) local.setRegAsB32(thread, d.reg, 0xdeadbeef);
					bool predicate = scenario == 0 || scenario == 2 || thread < complete;
					if ((scenario == 3 && thread == 0) || (scenario == 4 && thread < 32) || scenario == 8) predicate = false;
					local.setRegAsPredicate(thread, ins.pg.reg, scenario == 10 ? !predicate : predicate);
					if (scenario == 2 && thread >= complete) context.active.reset(thread);
					PTXU64 address = (thread % 32) * 16;
					if (scenario == 6) address += reinterpret_cast<PTXU64>(memory);
					if (scenario == 5 && thread == 0) address = 1024;
					if (scenario == 9 && thread == 0) address = 1;
					if ((scenario == 6 && thread == 0) || scenario == 8) address = ~PTXU64(0);
					local.setRegAsU64(thread, ins.a.reg, address);
				}
				bool rejected = false;
				try { local.eval_Ldmatrix(context, ins); } catch (const RuntimeException&) { rejected = true; }
				const bool shouldReject = (scenario == 0 && size % 32) || scenario == 3 || scenario == 5 || scenario == 6 || scenario == 7 || scenario == 9;
				if (rejected != shouldReject) { status << "ldmatrix control mismatch CTA " << size << " scenario " << scenario << "\n"; return false; }
				for (int thread = 0; thread < size; ++thread) for (const auto& d : ins.d.array) {
					const PTXU32 expected = !rejected && context.predicated(thread, ins) ? 0 : 0xdeadbeef;
					if (local.getRegAsB32(thread, d.reg) != expected) return false;
				}
			}
		}
		return true;
	}

	bool test_LdmatrixIR() {
		for (auto count : {PTXOperand::v1, PTXOperand::v2, PTXOperand::v4}) {
			PTXInstruction ins(PTXInstruction::Ldmatrix); ins.type = PTXOperand::b16;
			ins.addressSpace = PTXInstruction::Shared; ins.vec = count;
			ins.a = PTXOperand(PTXOperand::Indirect, PTXOperand::u64, 0);
			ins.d = PTXOperand(PTXOperand::Register, PTXOperand::b32, 1, 0, count);
			for (int i = 0; i < int(count); ++i) ins.d.array.emplace_back(PTXOperand::Register, PTXOperand::b32, i + 1);
			if (!ins.valid().empty() || !ins.isLoad() || !ins.isMemoryInstruction()) return false;
			for (auto addressType : {PTXOperand::b32, PTXOperand::b64, PTXOperand::s32,
				PTXOperand::s64, PTXOperand::u32, PTXOperand::u64}) {
				ins.a.type = addressType;
				if (!ins.valid().empty()) return false;
			}
			auto symbol = ins; symbol.a.addressMode = PTXOperand::Address; symbol.a.type = PTXOperand::b8;
			if (!symbol.valid().empty()) return false;
			for (int fault = 0; fault < 5; ++fault) {
				auto bad = ins;
				if (fault == 0) bad.d.array.pop_back();
				if (fault == 1) bad.d.array[0].type = PTXOperand::b64;
				if (fault == 2) bad.addressSpace = PTXInstruction::Global;
				if (fault == 3) bad.type = PTXOperand::b8;
				if (fault == 4) bad.a.vec = PTXOperand::v2;
				if (bad.valid().empty()) return false;
			}
		}
		return true;
	}

	bool test_SparseMmaParser() {
		const std::string mnemonic = "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32";
		const std::string prefix = ".version 8.0\n.target sm_86\n.entry sparse_parser() { .reg .b32 r<12>; ";
		const std::string fragments = " {r0,r1,r2,r3}, {r4,r5}, {r6,r7}, {r8,r9,r10,r11}, e, ";
		for (const char* metadataType : {"b32", "u32", "s32"}) for (unsigned selector = 0; selector < 4; ++selector) {
			std::stringstream source(prefix + ".reg ." + metadataType + " e; mov.b32 e, 0; " +
				mnemonic + fragments + (selector == 3 ? "0x3" : std::to_string(selector)) + "; ret; }");
			Module parsed;
			try {
				parsed.load(source);
				std::stringstream printed; parsed.write(printed);
				Module roundtrip; roundtrip.load(printed);
				if (printed.str().find(mnemonic) == std::string::npos) return false;
				EmulatedKernel localKernel(parsed.getKernel("sparse_parser"), nullptr);
				unsigned metadataRegister = 0;
				bool foundMma = false;
				for (const auto& ins : localKernel.instructions) {
					if (ins.opcode == PTXInstruction::Mov) metadataRegister = ins.d.reg;
					if (ins.opcode == PTXInstruction::Mma) foundMma = true;
					if (ins.opcode == PTXInstruction::Mma && (!ins.mmaSparse ||
						ins.q.reg != metadataRegister || PTXOperand::toString(ins.q.type) != metadataType ||
						ins.mmaSelector.imm_uint != selector || !ins.valid().empty())) return false;
				}
				if (!foundMma) return false;
			} catch (const std::exception& error) {
				status << "sparse MMA parse/round-trip failed: " << error.what() << "\n";
				return false;
			}
		}
		const std::string valid = prefix + ".reg .b32 e; " + mnemonic + fragments + "0; ret; }";
		for (const auto& replacement : {
			std::make_pair(".b32 e", ".b64 e"), std::make_pair(".b32 e", ".pred e"),
			std::make_pair(".b32 e", ".f32 e"), std::make_pair(".b32 e", ".v2 .b32 e"),
			std::make_pair("e, 0;", "missing, 0;"), std::make_pair("e, 0;", "0, 0;"),
			std::make_pair("e, 0;", "e, e;"), std::make_pair("e, 0;", "e, 4;"),
			std::make_pair("e, 0;", "e, 0f00000000;"),
			std::make_pair("e, 0;", "e, -1;"), std::make_pair("{r4,r5}", "{r4}"),
			std::make_pair("{r6,r7}", "{r6}"), std::make_pair("{r0,r1,r2,r3}", "{r0,r1}"),
			std::make_pair("{r8,r9,r10,r11}", "{r8,r9}"), std::make_pair("row.col", "row.row"),
			std::make_pair(".sp.sync", ".sp::ordered_metadata.sync"),
			std::make_pair("m16n8k8", "m16n8k16"), std::make_pair("tf32.tf32", "bf16.bf16"),
			std::make_pair(".col.f32", ".col.satfinite.f32")}) {
			std::string bad = valid;
			bad.replace(bad.find(replacement.first), std::string(replacement.first).size(), replacement.second);
			std::stringstream source(bad);
			bool rejected = false;
			try { Module parsed; parsed.load(source); }
			catch (const std::exception&) { rejected = true; }
			if (!rejected) { status << "invalid sparse MMA accepted: " << bad << "\n"; return false; }
		}
		return true;
	}

	bool test_SparseHalfMmaParser() {
		for (bool wide : {false, true})
		for (const char* types : {"f16.f16.f16.f16", "f32.f16.f16.f32", "f32.bf16.bf16.f32"}) {
			const bool half = std::string(types).find("f16.") == 0;
			const std::string accum = half ? "{r0,r1}" : "{r0,r1,r2,r3}";
			const std::string a = wide ? "{a0,a1,a2,a3}" : "{a0,a1}", b = wide ? "{b0,b1,b2,b3}" : "{b0,b1}";
			const std::string shape = wide ? "m16n8k32" : "m16n8k16";
			for (const char* container : {"b32", "f16x2"}) for (unsigned selector = 0; selector < (wide ? 2u : 4u); ++selector) {
				const std::string mnemonic = "mma.sp.sync.aligned." + shape + ".row.col." + std::string(types);
				const std::string valid = ".version 8.0\n.target sm_86\n.entry sparse_half() { .reg .b32 r<4>; .reg .b32 e; .reg ." +
					std::string(container) + " a<4>; .reg ." + container + " b<4>; " + mnemonic + " " + accum +
					", " + a + ", " + b + ", " + accum + ", e, " + std::to_string(selector) + "; ret; }";
				try {
					std::stringstream source(valid); Module parsed; parsed.load(source);
					std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
					if (printed.str().find(mnemonic) == std::string::npos) return false;
					EmulatedKernel localKernel(parsed.getKernel("sparse_half"), nullptr);
					for (const auto& ins : localKernel.instructions) if (ins.opcode == PTXInstruction::Mma) {
						if (!ins.mmaSparse || ins.mmaShape != (wide ? PTXInstruction::MmaM16N8K32 : PTXInstruction::MmaM16N8K16) ||
							ins.mmaSelector.imm_uint != selector || !ins.valid().empty()) return false;
					}
				} catch (const std::exception& error) {
					status << "sparse half MMA round-trip failed: " << error.what() << "\n"; return false;
				}
				for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
					std::make_pair(a, "{a0}"), std::make_pair(b, "{b0}"),
					std::make_pair(shape, "m16n8k64"), std::make_pair("row.col", "col.col"),
					std::make_pair(", e, " + std::to_string(selector) + ";", wide ? ", e, 2;" : ", e, 4;"),
					std::make_pair(".col.", ".col.satfinite."), std::make_pair(".b32 r", ".b64 r"),
					std::make_pair(" e;", " e_unused;"), std::make_pair(", e, ", ", 0, "),
					std::make_pair("; ret;", ", e; ret;"),
					std::make_pair(std::string(types) + " ", std::string(types).substr(0, std::string(types).rfind('.') + 1) + (half ? "f32 " : "f16 ")),
					std::make_pair(".col." + std::string(types), ".col.f16.bf16.bf16.f16"),
					std::make_pair(".col." + std::string(types), ".col.f32.f16.bf16.f32")}) {
					std::string bad = valid;
					bad.replace(bad.find(replacement.first), std::string(replacement.first).size(), replacement.second);
					std::stringstream source(bad); bool rejected = false;
					try { Module parsed; parsed.load(source); } catch (const std::exception&) { rejected = true; }
					if (!rejected) { status << "invalid sparse half MMA accepted: " << bad << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_SparseTf32Mma(bool wide = false) {
		const int kSize = wide ? 16 : 8;
		const std::string fragments = wide ? "{r4,r5,r12,r13}, {r6,r7,r14,r15}" : "{r4,r5}, {r6,r7}";
		std::stringstream source(".version 8.0\n.target sm_86\n.entry sparse_tf32_run() { .reg .b32 r<16>; .reg .b32 e; "
			"mma.sp.sync.aligned.m16n8k" + std::to_string(kSize) + ".row.col.f32.tf32.tf32.f32 {r0,r1,r2,r3}, " + fragments + ", {r0,r1,r2,r3}, e, 0; ret; }");
		Module parsed; parsed.load(source); EmulatedKernel localKernel(parsed.getKernel("sparse_tf32_run"), nullptr);
		std::stringstream printed; parsed.write(printed);
		Module roundtrip; roundtrip.load(printed);
		localKernel.setKernelShape(64, 1, 1); CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
		for (auto original : localKernel.instructions) if (original.opcode == PTXInstruction::Mma)
		for (unsigned selector = 0; selector < (wide ? 2u : 4u); ++selector) for (bool aliasSources : {false, true}) {
			PTXInstruction ins = original; ins.mmaSelector = PTXOperand(selector, PTXOperand::u32); localCTA.reset();
			for (int invalid = 0; invalid < 5; ++invalid) {
				PTXInstruction bad = ins;
				if (invalid == 0) bad.mmaSelector = PTXOperand(wide ? 2u : 4u, PTXOperand::u32);
				if (invalid == 1) bad.a.array.pop_back();
				if (invalid == 2) bad.b.array.pop_back();
				if (invalid == 3) bad.c.array[0].type = PTXOperand::b64;
				if (invalid == 4) bad.modifier = PTXInstruction::satfinite;
				if (bad.valid().empty()) return false;
			}
			if (aliasSources) {
				ins.d.array[0] = ins.a.array[0]; ins.d.array[1] = ins.a.array[1];
				ins.d.array[2] = ins.b.array[0]; ins.d.array[3] = ins.q;
			}
			auto position = [selector](int row, int chunk, int warp) { return (row + row / 8 + chunk + chunk / 4 + warp + selector) % 2; };
			auto aValue = [&](int row, int k, int warp) {
				return k % 2 == position(row, k / 2, warp) ? ((row * 5 + k * 3 + warp * 7) % 13 - 6) * 0.25f : 0.0f;
			};
			auto bValue = [](int k, int col, int warp) { return ((k * 7 + col * 3 + warp * 5) % 11 - 5) * 0.5f; };
			for (int thread = 0; thread < 64; ++thread) {
				const int lane = thread % 32, warp = thread / 32, group = lane / 4, t = lane % 4;
				PTXU32 metadata = 0;
				for (int rowHalf = 0; rowHalf < 2; ++rowHalf) for (int chunk = 0; chunk < 4; ++chunk) {
					const PTXU32 index = position(group + rowHalf * 8, chunk + (wide ? (t & 1) * 4 : 0), warp) ? 0xe : 0x4;
					metadata |= index << (rowHalf * 16 + chunk * 4);
				}
				localCTA.setRegAsB32(thread, ins.q.reg, (wide ? t / 2 : t) == int(selector) ? metadata : 0xeeeeeeeeu);
				for (int i = 0; i < (wide ? 4 : 2); ++i) {
					const int row = group + (i & 1) * 8, chunk = t + (i / 2) * 4;
					const int k = chunk * 2 + position(row, chunk, warp);
					localCTA.setRegAsF32(thread, ins.a.array[i].reg, aValue(row, k, warp));
					localCTA.setRegAsF32(thread, ins.b.array[i].reg, bValue(t + i * 4, group, warp));
				}
				for (int i = 0; i < 4; ++i) localCTA.setRegAsF32(thread, ins.c.array[i].reg,
					group + (i / 2) * 8 + (t * 2 + (i & 1)) * 0.125f - 8 + warp);
			}
			try { localCTA.eval_Mma(localCTA.getActiveContext(), ins); }
			catch (const RuntimeException& error) { status << "sparse TF32 execution failed: " << error.what() << "\n"; return false; }
			for (int thread = 0; thread < 64; ++thread) for (int i = 0; i < 4; ++i) {
				const int lane = thread % 32, warp = thread / 32, row = lane / 4 + (i / 2) * 8, col = (lane % 4) * 2 + (i & 1);
				float expected = row + col * 0.125f - 8 + warp;
				for (int k = 0; k < kSize; ++k) expected += aValue(row, k, warp) * bValue(k, col, warp);
				if (localCTA.getRegAsF32(thread, ins.d.array[i].reg) != expected) {
					status << "sparse TF32 mismatch selector " << selector << " thread " << thread << " row " << row << " col " << col << "\n"; return false;
				}
			}
		}
		return true;
	}

	bool test_SparseVectorFragments() {
		std::stringstream inverted(".version 8.0\n.target sm_86\n.entry inverted() { .reg .b32 d; .reg .pred p,q; "
			"selp.b32 d, %tid.x, %ntid.y, !p; setp.eq.and.u32 p, %tid.x, %ntid.y, !q; ret; }");
		Module predicates; predicates.load(inverted); std::stringstream predicateText; predicates.write(predicateText);
		Module predicateRoundtrip; predicateRoundtrip.load(predicateText);
		for (const char* form : {
			"m16n8k16.row.col.f16.f16.f16.f16", "m16n8k16.row.col.f32.bf16.bf16.f32",
			"m16n8k32.row.col.f16.f16.f16.f16", "m16n8k32.row.col.f32.bf16.bf16.f32",
			"m16n8k8.row.col.f32.tf32.tf32.f32", "m16n8k16.row.col.f32.tf32.tf32.f32",
			"m16n8k32.row.col.s32.s8.u8.s32", "m16n8k64.row.col.s32.u8.s8.s32",
			"m16n8k64.row.col.s32.s4.u4.s32", "m16n8k128.row.col.s32.u4.s4.s32"}) {
			const std::string op(form); const bool integer = op.find("s32") != std::string::npos;
			const bool half = op.find(".col.f16") != std::string::npos, bf16 = op.find("bf16") != std::string::npos;
			const bool tf32 = op.find("tf32") != std::string::npos, nibble = op.find("u4") != std::string::npos;
			const int k = std::stoi(op.substr(op.find('k') + 1));
			const int abCount = integer ? (k == (nibble ? 128 : 64) ? 4 : 2) : tf32 ? (k == 16 ? 4 : 2) : k == 32 ? 4 : 2;
			const std::string prefix = ".version 8.0\n.target sm_86\n.entry vector_fragments() { .reg .v" + std::to_string(abCount) +
				" .b32 va; .reg .v" + std::to_string(abCount) + " .b32 vb; .reg .v" + std::to_string(half ? 2 : 4) +
				" .b32 vc; .reg .v" + std::to_string(half ? 2 : 4) + " .b32 vd; .reg .b32 e; ";
			std::stringstream source(prefix + "mma.sp.sync.aligned." + op + " vd, va, vb, vc, e, 0; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel localKernel(roundtrip.getKernel("vector_fragments"), nullptr);
			localKernel.setKernelShape(32, 1, 1); CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
			const PTXU32 one = integer ? (nibble ? 0x11111111u : 0x01010101u) : tf32 ? 0x3f800000u : bf16 ? 0x3f803f80u : 0x3c003c00u;
			for (const auto& ins : localKernel.instructions) if (ins.opcode == PTXInstruction::Mma) {
				for (int lane = 0; lane < 32; ++lane) {
					for (const auto& c : ins.c.array) localCTA.setRegAsB32(lane, c.reg, 0);
					for (const auto* fragment : {&ins.a, &ins.b}) for (const auto& element : fragment->array) localCTA.setRegAsB32(lane, element.reg, one);
					localCTA.setRegAsB32(lane, ins.q.reg, 0x44444444u);
				}
				localCTA.eval_Mma(localCTA.getActiveContext(), ins);
				const PTXU32 halfBits = hydrazine::bit_cast<PTXU16>(PTXF16(k / 2));
				const PTXU32 expected = integer ? k / 2 : half ? halfBits | (halfBits << 16) : hydrazine::bit_cast<PTXU32>(float(k / 2));
				for (int lane = 0; lane < 32; ++lane) for (const auto& d : ins.d.array) if (localCTA.getRegAsB32(lane, d.reg) != expected) return false;
			}
			std::string bad = printed.str();
			if (abCount == 2) bad.replace(bad.find("va.x"), 4, "va.z");
			else bad.replace(bad.find(".v4 .b32 va"), 11, ".v2 .b32 va");
			std::stringstream invalid(bad); bool rejected = false;
			try { Module module; module.load(invalid); } catch (const std::exception&) { rejected = true; }
			if (!rejected) return false;
		}
		return true;
	}

	bool test_SparseIntegerMmaParser(bool subByte = false) {
		for (bool wide : {false, true}) for (auto signedA : {false, true}) for (auto signedB : {false, true})
		for (bool saturate : {false, true}) for (unsigned selector = 0; selector < (wide ? 1u : 2u); ++selector)
		for (auto metadataType : {"b32", "s32", "u32"}) {
			const std::string aType = subByte ? (signedA ? "s4" : "u4") : (signedA ? "s8" : "u8");
			const std::string bType = subByte ? (signedB ? "s4" : "u4") : (signedB ? "s8" : "u8");
			const std::string kSize = subByte ? (wide ? "128" : "64") : (wide ? "64" : "32");
			const std::string a = wide ? "{r4,r5,r12,r13}" : "{r4,r5}", b = wide ? "{r6,r7,r14,r15}" : "{r6,r7}";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry sparse_int_parser() { .reg .b32 r<16>; .reg ." + std::string(metadataType) + " e; mov.b32 e, 0; "
				"mma.sp.sync.aligned.m16n8k" + kSize + ".row.col" + (saturate ? ".satfinite" : "") + ".s32." + aType + "." + bType + ".s32 {r0,r1,r2,r3}, " + a + ", " + b + ", {r0,r1,r2,r3}, e, " + std::to_string(selector) + "; ret; }");
			Module parsed; parsed.load(source); std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed); EmulatedKernel localKernel(parsed.getKernel("sparse_int_parser"), nullptr);
			unsigned metadataRegister = 0;
			for (const auto& ins : localKernel.instructions) {
				if (ins.opcode == PTXInstruction::Mov) metadataRegister = ins.d.reg;
				if (ins.opcode != PTXInstruction::Mma) continue;
				if (!ins.mmaSparse || ins.q.reg != metadataRegister || ins.mmaSelector.imm_uint != selector ||
					bool(ins.modifier & PTXInstruction::satfinite) != saturate || !ins.valid().empty()) return false;
				for (int malformed = 0; malformed < 9; ++malformed) {
					PTXInstruction bad = ins;
					if (malformed == 0) bad.q.type = PTXOperand::b64;
					if (malformed == 1) bad.mmaSelector = PTXOperand(wide ? 1u : 2u, PTXOperand::u32);
					if (malformed == 2) bad.a.array.pop_back();
					if (malformed == 3) bad.b.array.pop_back();
					if (malformed == 4) bad.c.array[0].type = PTXOperand::b64;
					if (malformed == 5) bad.modifier |= PTXInstruction::ftz;
					if (malformed == 6) bad.b.type = PTXOperand::bf16;
					if (malformed == 7) bad.mmaShape = PTXInstruction::MmaM16N8K16;
					if (malformed == 8) bad.a.array[0].vec = PTXOperand::v2;
					if (bad.valid().empty()) return false;
				}
			}
		}
		return true;
	}

	bool test_SparseIntegerMma(bool wide, bool subByte = false) {
		const int positions[12][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3},
			{1,0},{2,0},{3,0},{2,1},{3,1},{3,2}};
		const int bits = subByte ? 4 : 8, elements = 32 / bits, chunkSize = subByte ? 8 : 4;
		const unsigned mask = (1u << bits) - 1;
		const int kSize = (wide ? 64 : 32) * (subByte ? 2 : 1);
		for (bool signedA : {false, true}) for (bool signedB : {false, true}) {
			const std::string a = wide ? "{r4,r5,r12,r13}" : "{r4,r5}", b = wide ? "{r6,r7,r14,r15}" : "{r6,r7}";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry sparse_int_run() { .reg .b32 r<16>; .reg .b32 e; mma.sp.sync.aligned.m16n8k" +
				std::to_string(kSize) + ".row.col.s32." + (signedA ? "s" : "u") + std::to_string(bits) + "." + (signedB ? "s" : "u") + std::to_string(bits) + ".s32 {r0,r1,r2,r3}, " + a + ", " + b + ", {r0,r1,r2,r3}, e, 0; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel localKernel(parsed.getKernel("sparse_int_run"), nullptr);
			localKernel.setKernelShape(64, 1, 1); CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
			for (auto original : localKernel.instructions) if (original.opcode == PTXInstruction::Mma)
			for (unsigned selector = 0; selector < (wide ? 1u : 2u); ++selector) for (bool saturate : {false, true})
			for (PTXS32 base : {0, (std::numeric_limits<PTXS32>::max)() - 64, (std::numeric_limits<PTXS32>::min)() + 64})
			for (bool aliasSources : {false, true}) {
				PTXInstruction ins = original;
				if (aliasSources) {
					ins.d.array[0] = ins.a.array[0]; ins.d.array[1] = ins.a.array[1];
					ins.d.array[2] = ins.b.array[0]; ins.d.array[3] = ins.q;
				}
				ins.mmaSelector = PTXOperand(selector, PTXOperand::u32); ins.modifier = saturate ? PTXInstruction::satfinite : 0; localCTA.reset();
				auto pair = [&](int row, int chunk, int warp) -> const int* { return positions[(row + chunk + warp) % 12]; };
				auto aValue = [&](int row, int k, int warp) {
					const int* p = pair(row, k / chunkSize, warp);
					const int position = (k % chunkSize) / (subByte ? 2 : 1);
					if (position != p[0] && position != p[1]) return 0;
					if (subByte) return signedA ? (row & 1 ? -1 : 1) * (1 + (row * 7 + k * 3 + warp * 11) % 7) : 8 + (row * 7 + k * 3 + warp * 11) % 8;
					return signedA ? (row & 1 ? -1 : 1) * (20 + (row * 7 + k * 3 + warp * 11) % 100) : 128 + (row * 7 + k * 3 + warp * 11) % 120;
				};
				auto bValue = [&](int k, int col, int warp) {
					if (subByte) return signedB ? (col & 1 ? -1 : 1) * (1 + (k * 11 + col * 5 + warp * 13) % 7) : 8 + (k * 11 + col * 5 + warp * 13) % 8;
					return signedB ? (col & 1 ? -1 : 1) * (10 + (k * 11 + col * 5 + warp * 13) % 110) : 128 + (k * 11 + col * 5 + warp * 13) % 120;
				};
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32, warp = thread / 32, group = lane / 4, t = lane % 4;
					PTXU32 metadata = 0;
					for (int chunk = 0; chunk < 8; ++chunk) for (int j = 0; j < 2; ++j)
						metadata |= PTXU32(pair(group + (t & 1) * 8, chunk + (wide ? (t / 2) * 8 : 0), warp)[j]) << (chunk * 4 + j * 2);
					localCTA.setRegAsB32(thread, ins.q.reg, wide || t / 2 == int(selector) ? metadata : 0xffffffffu);
					for (unsigned r = 0; r < ins.a.array.size(); ++r) {
						PTXU32 aReg = 0, bReg = 0; const int row = group + (r & 1) * 8;
						for (int j = 0; j < elements; ++j) {
							const int chunk = t * 2 + (r / 2) * 8 + j / (subByte ? 4 : 2);
							const int index = pair(row, chunk, warp)[(j / (subByte ? 2 : 1)) & 1];
							const int k = chunk * chunkSize + index * (subByte ? 2 : 1) + (subByte ? j & 1 : 0);
							aReg |= (PTXU32(aValue(row, k, warp)) & mask) << (j * bits);
							bReg |= (PTXU32(bValue(t * elements + r * elements * 4 + j, group, warp)) & mask) << (j * bits);
						}
						localCTA.setRegAsB32(thread, ins.a.array[r].reg, aReg); localCTA.setRegAsB32(thread, ins.b.array[r].reg, bReg);
					}
					for (int i = 0; i < 4; ++i) localCTA.setRegAsS32(thread, ins.c.array[i].reg,
						base + (group + (i / 2) * 8 + t * 2 + (i & 1)) % 13 - 6);
				}
				try { localCTA.eval_Mma(localCTA.getActiveContext(), ins); }
				catch (const RuntimeException& error) { status << "sparse integer execution failed: " << error.what() << "\n"; return false; }
				for (int thread = 0; thread < 64; ++thread) for (int i = 0; i < 4; ++i) {
					const int warp = thread / 32, row = (thread % 32) / 4 + (i / 2) * 8, col = (thread % 4) * 2 + (i & 1);
					int64_t expected = int64_t(base) + (row + col) % 13 - 6;
					for (int k = 0; k < kSize; ++k) expected += int64_t(aValue(row, k, warp)) * bValue(k, col, warp);
					if (saturate) expected = std::max<int64_t>((std::numeric_limits<PTXS32>::min)(), std::min<int64_t>((std::numeric_limits<PTXS32>::max)(), expected));
					if (localCTA.getRegAsB32(thread, ins.d.array[i].reg) != PTXU32(expected)) { status << "sparse integer mismatch K=" << kSize << " bits=" << bits << " thread " << thread << " register " << i << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_SparseHalfMma(bool wide = false) {
		const int positions[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
		const int kSize = wide ? 32 : 16;
		for (const char* types : {"f16.f16.f16.f16", "f32.f16.f16.f32", "f32.bf16.bf16.f32"}) {
			const bool half = std::string(types).find("f16.") == 0, bf16 = std::string(types).find("bf16") != std::string::npos;
			const std::string accum = half ? "{c0,c1}" : "{c0,c1,c2,c3}";
			const std::string fragments = wide ? ", {a0,a1,a2,a3}, {b0,b1,b2,b3}, " : ", {a0,a1}, {b0,b1}, ";
			std::stringstream source(".version 8.0\n.target sm_86\n.entry sparse_half_run() { .reg .b32 a<4>; .reg .b32 b<4>; .reg .b32 c<4>; .reg .b32 e; "
				"mma.sp.sync.aligned.m16n8k" + std::to_string(kSize) + ".row.col." + types + " " + accum + fragments + accum + ", e, 0; ret; }");
			Module parsed; parsed.load(source); EmulatedKernel localKernel(parsed.getKernel("sparse_half_run"), nullptr);
			localKernel.setKernelShape(64, 1, 1); CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
			auto encode = [](float value, bool bfloat) -> PTXU16 {
				return bfloat ? hydrazine::bit_cast<PTXU32>(value) >> 16 : hydrazine::bit_cast<PTXU16>(PTXF16(value));
			};
			for (auto original : localKernel.instructions) if (original.opcode == PTXInstruction::Mma)
			for (unsigned selector = 0; selector < (wide ? 2u : 4u); ++selector) for (bool aliasSources : {false, true}) {
				PTXInstruction ins = original;
				if (aliasSources) {
					ins.d.array[0] = ins.a.array[0]; ins.d.array[1] = ins.a.array[1];
					if (!half) { ins.d.array[2] = ins.b.array[0]; ins.d.array[3] = ins.q; }
				}
				ins.mmaSelector = PTXOperand(selector, PTXOperand::u32); localCTA.reset();
				float A[2][16][32] = {}, B[2][32][8];
				PTXU32 metadata[2][16] = {};
				for (int warp = 0; warp < 2; ++warp) for (int row = 0; row < 16; ++row) for (int chunk = 0; chunk < kSize / 4; ++chunk) {
					const auto& pair = positions[(row * 5 + chunk + selector + warp) % 6];
					for (int j = 0; j < 2; ++j) {
						const int k = chunk * 4 + pair[j];
						A[warp][row][k] = ((row * 5 + k * 3 + warp * 7) % 13 - 6) * 0.25f;
						metadata[warp][row % 8 + (chunk / 4) * 8] |= PTXU32(pair[j]) << ((row / 8) * 16 + (chunk % 4) * 4 + j * 2);
					}
				}
				for (int warp = 0; warp < 2; ++warp) for (int k = 0; k < kSize; ++k) for (int col = 0; col < 8; ++col)
					B[warp][k][col] = ((k * 7 + col * 3 + warp * 5) % 11 - 5) * 0.5f;
				for (int thread = 0; thread < 64; ++thread) {
					const int lane = thread % 32, warp = thread / 32, group = lane / 4, t = lane % 4;
					const int metadataIndex = group + (wide ? (t & 1) * 8 : 0);
					const bool suppliesMetadata = (wide ? t / 2 : t) == int(selector);
					localCTA.setRegAsB32(thread, ins.q.reg, suppliesMetadata ? metadata[warp][metadataIndex] : 0xffffffffu);
					for (int regIndex = 0; regIndex < (wide ? 4 : 2); ++regIndex) {
						const int row = group + (regIndex & 1) * 8, chunk = t + (regIndex / 2) * 4;
						const auto& pair = positions[(row * 5 + chunk + selector + warp) % 6];
						PTXU32 a = 0, b = 0;
						for (int j = 0; j < 2; ++j) {
							a |= PTXU32(encode(A[warp][row][chunk * 4 + pair[j]], bf16)) << (16 * j);
							b |= PTXU32(encode(B[warp][t * 2 + j + regIndex * 8][group], bf16)) << (16 * j);
						}
						localCTA.setRegAsB32(thread, ins.a.array[regIndex].reg, a);
						localCTA.setRegAsB32(thread, ins.b.array[regIndex].reg, b);
					}
					for (int i = 0; i < 4; ++i) {
						const float c = (group + (i / 2) * 8) + (t * 2 + (i & 1)) * 0.125f - 8 + warp;
						if (!half) localCTA.setRegAsF32(thread, ins.c.array[i].reg, c);
						else if (!(i & 1)) localCTA.setRegAsB32(thread, ins.c.array[i / 2].reg, encode(c, false) | PTXU32(encode(c + 0.125f, false)) << 16);
					}
				}
				try { localCTA.eval_Mma(localCTA.getActiveContext(), ins); }
				catch (const RuntimeException& error) { status << "sparse half MMA execution failed: " << error.what() << "\n"; return false; }
				for (int thread = 0; thread < 64; ++thread) for (int i = 0; i < 4; ++i) {
					const int lane = thread % 32, warp = thread / 32;
					const int row = lane / 4 + (i / 2) * 8, col = (lane % 4) * 2 + (i & 1);
					float expected = row + col * 0.125f - 8 + warp;
					for (int k = 0; k < kSize; ++k) expected += A[warp][row][k] * B[warp][k][col];
					const bool correct = half ? PTXU16(localCTA.getRegAsB32(thread, ins.d.array[i / 2].reg) >> (16 * (i & 1))) == encode(expected, false)
						: localCTA.getRegAsF32(thread, ins.d.array[i].reg) == expected;
					if (!correct) { status << "sparse half MMA mismatch " << types << " selector " << selector << " row " << row << " col " << col << "\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_MmaNamedDestination() {
		// A named D vector must not become part of the following A fragment.
		for (const char* shape : {"m8n8k16", "m16n8k16", "m16n8k32"}) {
			const bool small = std::string(shape) == "m8n8k16";
			const bool wide = std::string(shape) == "m16n8k32";
			const unsigned aCount = small ? 1 : wide ? 4 : 2;
			const unsigned bCount = wide ? 2 : 1;
			const unsigned dCount = small ? 2 : 4;
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry named_mma() { .reg .v"
				<< dCount << " .b32 vd; .reg .b32 a<4>; .reg .b32 b<2>; .reg .b32 c<4>; "
				<< "mma.sync.aligned." << shape << ".row.col.s32.s8.u8.s32 vd, "
				<< (small ? "{a0}" : wide ? "{a0,a1,a2,a3}" : "{a0,a1}")
				<< (wide ? ", {b0,b1}, " : ", {b0}, ")
				<< (small ? "{c0,c1}" : "{c0,c1,c2,c3}") << "; ret; }";
			Module parsed;
			try { parsed.load(source); }
			catch (const std::exception& error) {
				status << "named MMA destination corrupted next fragment: " << error.what() << "\n";
				return false;
			}
			EmulatedKernel localKernel(parsed.getKernel("named_mma"), nullptr);
			localKernel.setKernelShape(32, 1, 1);
			CooperativeThreadArray localCTA(&localKernel, Dim3(), false);
			for (const auto& ins : localKernel.instructions) {
				if (ins.opcode != PTXInstruction::Mma) continue;
				if (ins.a.array.size() != aCount || ins.b.array.size() != bCount ||
					ins.c.array.size() != dCount || ins.d.array.size() != dCount) return false;
				for (int lane = 0; lane < 32; ++lane) {
					for (const auto& a : ins.a.array)
						localCTA.setRegAsB32(lane, a.reg, 0x01010101);
					for (const auto& b : ins.b.array)
						localCTA.setRegAsB32(lane, b.reg, 0x02020202);
					for (unsigned i = 0; i < dCount; ++i)
						localCTA.setRegAsB32(lane, ins.c.array[i].reg, lane * 10 + i);
				}
				localCTA.eval_Mma(localCTA.getActiveContext(), ins);
				for (int lane = 0; lane < 32; ++lane) for (unsigned i = 0; i < dCount; ++i) {
					const PTXU32 expected = (wide ? 64u : 32u) + lane * 10 + i;
					if (localCTA.getRegAsB32(lane, ins.d.array[i].reg) != expected) {
						status << "named MMA destination result mismatch: " << shape << "\n";
						return false;
					}
				}
			}
		}
		return true;
	}

	bool test_MmaFloatShapes() {
		for (const char* shape : {"m8n8k128", "m16n8k128", "m16n8k256"}) for (const char* op : {"xor", "and"}) {
			const bool small = std::string(shape) == "m8n8k128", k256 = std::string(shape) == "m16n8k256";
			const std::string d = small ? "{%r0,%r1}" : "{%r0,%r1,%r2,%r3}";
			const std::string a = small ? "{%r4}" : k256 ? "{%r4,%r5,%r6,%r7}" : "{%r4,%r5}";
			const std::string b = k256 ? "{%r8,%r9}" : "{%r8}";
			const std::string c = small ? "{%r10,%r11}" : "{%r10,%r11,%r12,%r13}";
			const std::string mnemonic = std::string("mma.sync.aligned.") + shape + ".row.col.s32.b1.b1.s32." + op + ".popc";
			const std::string source = ".version 8.0\n.target sm_86\n.entry binary_mma() { .reg .b32 %r<14>; " + mnemonic + " " + d + ", " + a + ", " + b + ", " + c + "; ret; }";
			std::stringstream input(source); Module parsed; parsed.load(input);
			std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
			if (printed.str().find(mnemonic) == std::string::npos) return false;
			for (const auto& statement : parsed.statements()) if (statement.instruction.opcode == PTXInstruction::Mma) {
				const auto& ins = statement.instruction;
				PTXInstruction invalid = ins; invalid.booleanOperator = PTXInstruction::BoolOr;
				if (invalid.valid().empty()) return false;
				if (ins.a.type != PTXOperand::b1 || ins.booleanOperator != (std::string(op) == "xor" ? PTXInstruction::BoolXor : PTXInstruction::BoolAnd)) return false;
				PTXInstruction run = ins;
				for (auto* fragment : {&run.a, &run.b, &run.c, &run.d})
					for (auto& element : fragment->array) element.reg = std::stoi(element.identifier.substr(2));
				run.c = run.d;
				auto aBit = [](int row, int k) { return (row * 13 + k * 7 + (k / 32) * 3) % 11 < 5; };
				auto bBit = [](int k, int col) { return (k * 5 + col * 3 + k / 64) % 13 < 6; };
				for (PTXU32 base : {0x7fffffc0u, 0xffffffe0u}) {
					cta->reset();
					for (int lane = 0; lane < 32; ++lane) {
						const int group = lane / 4, t = lane % 4;
						for (unsigned i = 0; i < run.a.array.size(); ++i) {
							PTXU32 packed = 0;
							for (int bit = 0; bit < 32; ++bit) packed |= PTXU32(aBit(group + (i & 1) * 8, (t + (i / 2) * 4) * 32 + bit)) << bit;
							cta->setRegAsB32(lane, run.a.array[i].reg, packed);
						}
						for (unsigned i = 0; i < run.b.array.size(); ++i) {
							PTXU32 packed = 0;
							for (int bit = 0; bit < 32; ++bit) packed |= PTXU32(bBit((t + i * 4) * 32 + bit, group)) << bit;
							cta->setRegAsB32(lane, run.b.array[i].reg, packed);
						}
						for (unsigned i = 0; i < run.c.array.size(); ++i)
							cta->setRegAsB32(lane, run.c.array[i].reg, base + (group + (i / 2) * 8) * 8 + 2 * t + (i & 1));
					}
					cta->eval_Mma(cta->getActiveContext(), run);
					for (int lane = 0; lane < 32; ++lane) for (unsigned i = 0; i < run.d.array.size(); ++i) {
						const int row = lane / 4 + (i / 2) * 8, col = (lane % 4) * 2 + (i & 1);
						PTXU32 expected = base + row * 8 + col;
						for (int k = 0; k < (k256 ? 256 : 128); ++k)
							expected += run.booleanOperator == PTXInstruction::BoolXor ? aBit(row, k) != bBit(k, col) : aBit(row, k) && bBit(k, col);
						if (cta->getRegAsB32(lane, run.d.array[i].reg) != expected) { status << "binary mma mismatch " << shape << " " << op << " lane " << lane << " register " << i << "\n"; return false; }
					}
				}
			}
			for (const auto& replacement : {std::make_pair(std::string(".popc"), std::string("")),
				std::make_pair(std::string(".") + op + ".popc", std::string(".or.popc")),
				std::make_pair(std::string("row.col"), std::string("col.col")),
				std::make_pair(std::string("row.col"), std::string("row.col.satfinite")),
				std::make_pair(std::string(shape), std::string("m8n8k16")),
				std::make_pair(std::string(".reg .b32"), std::string(".reg .b1")),
				std::make_pair(d, std::string("{%r0}"))}) {
				std::string bad = source; bad.replace(bad.find(replacement.first), replacement.first.size(), replacement.second);
				std::stringstream badInput(bad); bool rejected = false;
				try { Module module; module.load(badInput); } catch (const std::exception&) { rejected = true; }
				if (!rejected) { status << "invalid binary mma accepted: " << bad << "\n"; return false; }
			}
		}
		// Every fragment must retain its exact register count and register mode.
		for (const char* operation : {
			"m8n8k4.row.col.f64.f64.f64.f64",
			"m8n8k4.row.col.f16.f16.f16.f16",
			"m16n8k16.row.col.f32.f16.f16.f32",
			"m16n8k4.row.col.f32.tf32.tf32.f32",
			"m8n8k16.row.col.s32.s8.u8.s32",
			"m8n8k32.row.col.s32.s4.u4.s32",
			"m8n8k128.row.col.s32.b1.b1.s32.xor.popc"}) {
			const std::string op(operation);
			const bool fp64 = op.find("f64") != std::string::npos;
			const bool legacy = op.find("f16.f16.f16.f16") != std::string::npos;
			const bool modern = op.find("m16n8") == 0;
			const std::string a = op.find("m16n8k16") == 0 ? "{%r4,%r5,%r11,%r12}" :
				modern || legacy ? "{%r4,%r5}" : "{%r4}";
			const std::string d = modern || legacy ? "{%r0,%r1,%r2,%r3}" : "{%r0,%r1}";
			const std::string b = op.find("m16n8k16") == 0 || legacy ? "{%r6,%r13}" : "{%r6}";
			const std::string c = modern || legacy ? "{%r7,%r8,%r9,%r10}" : "{%r7,%r8}";
			const std::string instruction = "mma.sync.aligned." + op + " " + d + ", " + a + ", " + b + ", " + c + ";";
			const std::string prefix = ".version 8.0\n.target sm_86\n.global ." + std::string(fp64 ? "b64" : "b32") +
				" ga;\n.entry fragments() { .reg ." + (fp64 ? "b64" : "b32") + " %r<14>; " +
				".reg .v2 ." + (fp64 ? "b64" : "b32") + " va; " +
				".reg .v4 ." + (fp64 ? "b64" : "b32") + " vb; ";
			std::stringstream validSource(prefix + instruction + " ret; }");
			Module validModule; validModule.load(validSource);
			for (const std::string& fragment : {d, a, b, c}) {
				for (int malformed = 0; malformed < 4; ++malformed) {
					std::string bad = instruction;
					const auto position = bad.find(fragment);
					if (position == std::string::npos) continue;
					if (malformed == 0) bad.insert(position + fragment.size() - 1, ",%r13");
					else bad.replace(position + 1, 3, malformed == 1 ? "ga" : malformed == 2 ? "va" : "vb");
					std::stringstream source(prefix + bad + " ret; }");
					bool rejected = false;
					try { Module module; module.load(source); }
					catch (const std::exception&) { rejected = true; }
					if (!rejected) { status << "invalid mma fragment accepted: " << bad << "\n"; return false; }
				}
			}
		}
		// PTX 8.0: m8n8k16/m16n8k32/m16n8k64 are integer-only shapes.
		for (const char* shape : {"m16n8k4", "m16n8k8", "m16n8k16", "m8n8k16", "m16n8k32", "m16n8k64", "m8n8k128", "m16n8k128", "m16n8k256"}) {
			for (const char* input : {"f16", "bf16", "tf32"}) {
				for (const char* accumulator : {"f16", "f32"}) {
					const bool small = std::string(shape) == "m16n8k8";
					const bool k4 = std::string(shape) == "m16n8k4";
					const bool tf32 = std::string(input) == "tf32";
					const bool half = std::string(accumulator) == "f16";
					const bool legalShape = k4 ? tf32 : small || std::string(shape) == "m16n8k16";
					const bool legal = legalShape && (!tf32 || small || k4)
						&& (!half || std::string(input) == "f16");
					std::stringstream ptx;
					ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
						<< ".visible .entry mma_shapes() {\n.reg ."
						<< (half ? "f16x2" : "b32") << " %d<4>;\n.reg ."
						<< (tf32 ? "b32" : "f16x2") << " %r<10>;\n"
						<< ".reg ." << (half ? "f16x2" : "b32") << " %c<4>;\n"
						<< "mma.sync.aligned." << shape << ".row.col."
						<< accumulator << "." << input << "." << input << "."
						<< accumulator << (half ? " {%d0,%d1}, " : " {%d0,%d1,%d2,%d3}, ")
						<< (k4 || (small && !tf32) ? "{%r4,%r5}, {%r8}, "
							: "{%r4,%r5,%r6,%r7}, {%r8,%r9}, ")
						<< (half ? "{%c0,%c1};\n" : "{%c0,%c1,%c2,%c3};\n")
						<< "ret;\n}\n";
					Module parsed;
					bool accepted = true;
					try {
						parsed.load(ptx);
						std::stringstream printed; parsed.write(printed);
						Module roundtrip; roundtrip.load(printed);
					}
					catch (const std::exception&) { accepted = false; }
					if (accepted != legal) {
						status << "mma " << shape << " " << input << " -> "
							<< accumulator << (accepted ? " incorrectly accepted\n"
								: " incorrectly rejected\n");
						return false;
					}
				}
			}
		}
		for (const char* a : {"s4", "u4"}) for (const char* b : {"s4", "u4"}) {
			for (const char* saturation : {"", "satfinite."}) {
				const std::string instruction = std::string("mma.sync.aligned.m8n8k32.row.col.") + saturation +
					"s32." + a + "." + b + ".s32 {%r0,%r1}, {%r2}, {%r3}, {%r4,%r5};";
				const std::string source = ".version 8.0\n.target sm_86\n.entry int4() { .reg .b32 %r<6>; " + instruction + " ret; }";
				std::stringstream stream(source); Module parsed; parsed.load(stream);
				std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
				if (printed.str().find(instruction.substr(0, instruction.find(' '))) == std::string::npos) return false;
				const std::string wideInstruction = std::string("mma.sync.aligned.m16n8k32.row.col.") + saturation +
					"s32." + a + "." + b + ".s32 {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%r6}, {%r7,%r8,%r9,%r10};";
				std::stringstream wideSource(".version 8.0\n.target sm_86\n.entry int4_wide() { .reg .b32 %r<11>; " + wideInstruction + " ret; }");
				Module wideParsed; wideParsed.load(wideSource);
				std::stringstream widePrinted; wideParsed.write(widePrinted); Module wideRoundtrip; wideRoundtrip.load(widePrinted);
				if (widePrinted.str().find(wideInstruction.substr(0, wideInstruction.find(' '))) == std::string::npos) return false;
				const std::string k64Instruction = std::string("mma.sync.aligned.m16n8k64.row.col.") + saturation +
					"s32." + a + "." + b + ".s32 {%r0,%r1,%r2,%r3}, {%r4,%r5,%r6,%r7}, {%r8,%r9}, {%r10,%r11,%r12,%r13};";
				std::stringstream k64Source(".version 8.0\n.target sm_86\n.entry int4_k64() { .reg .b32 %r<14>; " + k64Instruction + " ret; }");
				Module k64Parsed; k64Parsed.load(k64Source);
				std::stringstream k64Printed; k64Parsed.write(k64Printed); Module k64Roundtrip; k64Roundtrip.load(k64Printed);
				if (k64Printed.str().find(k64Instruction.substr(0, k64Instruction.find(' '))) == std::string::npos) return false;
				for (const char* invalid : {"m8n8k16", "m16n8k32"}) {
					std::string bad = source;
					bad.replace(bad.find("m8n8k32"), 7, invalid);
					std::stringstream input(bad); bool rejected = false;
					try { Module module; module.load(input); }
					catch (const std::exception&) { rejected = true; }
					if (!rejected) { status << "invalid sub-byte mma shape/fragments accepted\n"; return false; }
				}
			}
		}
		return true;
	}

	bool test_Mma() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Mma;
		ins.mmaShape = PTXInstruction::MmaM16N8K16;
		ins.type = PTXOperand::f32;
		ins.modifier = 0;

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
		PTXInstruction sparse = ins;
		sparse.mmaSparse = true;
		sparse.mmaShape = PTXInstruction::MmaM16N8K8;
		sparse.a = vector(PTXOperand::tf32, PTXOperand::b32, PTXOperand::v2, 10, 2);
		sparse.b.type = PTXOperand::tf32;
		sparse.q = reg("metadata", PTXOperand::b32, 12);
		for (unsigned int selector = 0; selector < 4; ++selector) {
			sparse.mmaSelector = PTXOperand(selector, PTXOperand::u32);
			if (!sparse.valid().empty() || sparse.toString().find("mma.sp.sync.aligned.") != 0) {
				status << "legal sparse TF32 IR rejected: " << sparse.valid() << "\n";
				return false;
			}
		}
		for (auto type : {PTXOperand::b32, PTXOperand::s32, PTXOperand::u32}) {
			sparse.q.type = type;
			if (!sparse.valid().empty()) return false;
		}
		for (int malformed = 0; malformed < 12; ++malformed) {
			PTXInstruction bad = sparse;
			if (malformed == 0) bad.q.type = PTXOperand::b64;
			if (malformed == 1) bad.q.type = PTXOperand::pred;
			if (malformed == 2) bad.mmaSelector = reg("selector", PTXOperand::u32, 13);
			if (malformed == 3) bad.mmaSelector = PTXOperand(4u, PTXOperand::u32);
			if (malformed == 4) bad.a.array.pop_back();
			if (malformed == 5) bad.b.array.pop_back();
			if (malformed == 6) bad.c.array[0].type = PTXOperand::b64;
			if (malformed == 7) bad.a.type = PTXOperand::f16;
			if (malformed == 8) bad.q.addressMode = PTXOperand::Invalid;
			if (malformed == 9) bad.q.vec = PTXOperand::v2;
			if (malformed == 10) bad.c.array.pop_back();
			if (malformed == 11) bad.d.array.pop_back();
			if (bad.valid().empty()) {
				status << "invalid sparse TF32 IR accepted: " << malformed << "\n";
				return false;
			}
		}
		for (auto& operand : ins.a.array) operand.type = PTXOperand::f16x2;
		for (auto& operand : ins.b.array) operand.type = PTXOperand::f16x2;
		if (!ins.valid().empty()) {
			status << "mma rejected f16x2 input fragments: " << ins.valid() << "\n";
			return false;
		}
		auto rejects = [&](const PTXInstruction& invalid) {
			bool rejected = false;
			try { cta->eval_Mma(cta->getActiveContext(), invalid); }
			catch (const RuntimeException&) { rejected = true; }
			if (!rejected || invalid.valid().empty()) {
				status << "invalid mma accepted (shape " << invalid.mmaShape
					<< "): " << invalid.toString() << "\n";
				return false;
			}
			return true;
		};
		for (int fragment = 0; fragment < 4; ++fragment) {
			PTXInstruction invalid = ins;
			PTXOperand* operands[] = {&invalid.a, &invalid.b, &invalid.c, &invalid.d};
			operands[fragment]->addressMode = PTXOperand::Address;
			if (!rejects(invalid)) return false;
			for (int malformed = 0; malformed < 2; ++malformed) {
				invalid = ins;
				auto& element = operands[fragment]->array[0];
				if (malformed == 0) element.vec = PTXOperand::v2;
				else element.array.push_back(element);
				if (!rejects(invalid)) return false;
			}
		}
		for (auto shape : {PTXInstruction::MmaShape_Invalid,
			PTXInstruction::MmaM8N8K16, PTXInstruction::MmaM16N8K32}) {
			PTXInstruction invalid = ins;
			invalid.mmaShape = shape;
			if (!rejects(invalid)) return false;
		}
		for (int bad = 0; bad < 4; ++bad) {
			PTXInstruction invalid = ins;
			if (bad == 0) invalid.a.type = PTXOperand::s32;
			if (bad == 1) invalid.b.type = PTXOperand::bf16;
			if (bad == 2) invalid.modifier = PTXInstruction::rn;
			if (bad == 3) invalid.modifier = PTXInstruction::satfinite;
			if (!rejects(invalid)) return false;
		}
		PTXInstruction tf32 = ins;
		tf32.mmaShape = PTXInstruction::MmaM16N8K8;
		tf32.a.type = tf32.b.type = PTXOperand::tf32;
		if (tf32.valid().empty()) {
			status << "tf32 mma incorrectly accepted f16x2 fragments\n";
			return false;
		}

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

		// TF32 m16n8k4: A(r,k)=r-2k+1, B(k,c)=3k+c-2.
		ins.mmaShape = PTXInstruction::MmaM16N8K4;
		ins.a = vector(PTXOperand::tf32, PTXOperand::b32, PTXOperand::v2, 0, 2);
		ins.b = vector(PTXOperand::tf32, PTXOperand::b32, PTXOperand::v1, 4, 1);
		if (!ins.valid().empty()) { status << ins.valid() << "\n"; return false; }
		cta->reset();
		for (int lane = 0; lane < 32; ++lane) {
			const int group = lane >> 2, t = lane & 3;
			cta->setRegAsF32(lane, 0, group - 2 * t + 1);
			cta->setRegAsF32(lane, 1, group + 8 - 2 * t + 1);
			cta->setRegAsF32(lane, 4, 3 * t + group - 2);
			for (int i = 0; i < 4; ++i)
				cta->setRegAsF32(lane, 6 + i, (group + (i >= 2 ? 8 : 0)) * 8 + t * 2 + (i & 1));
		}
		cta->eval_Mma(cta->getActiveContext(), ins);
		for (int lane = 0; lane < 32; ++lane) {
			for (int i = 0; i < 4; ++i) {
				const int row = (lane >> 2) + (i >= 2 ? 8 : 0), col = (lane & 3) * 2 + (i & 1);
				int expected = row * 8 + col;
				for (int k = 0; k < 4; ++k) expected += (row - 2 * k + 1) * (3 * k + col - 2);
				if (cta->getRegAsF32(lane, i) != expected) {
					status << "mma.m16n8k4.tf32 incorrect lane " << lane << " register " << i << "\n";
					return false;
				}
			}
		}
		ins.mmaShape = PTXInstruction::MmaM8N8K4;
		ins.type = PTXOperand::f64;
		ins.a = vector(PTXOperand::f64, PTXOperand::f64, PTXOperand::v1, 0, 1);
		ins.b = vector(PTXOperand::f64, PTXOperand::f64, PTXOperand::v1, 4, 1);
		ins.c = vector(PTXOperand::f64, PTXOperand::f64, PTXOperand::v2, 6, 2);
		ins.d = vector(PTXOperand::f64, PTXOperand::f64, PTXOperand::v2, 0, 2);
		for (auto* operand : {&ins.a, &ins.b, &ins.c, &ins.d}) for (auto& element : operand->array) element.identifier = "%f" + std::to_string(element.reg);
		PTXInstruction conflictingRounding = ins;
		conflictingRounding.modifier = PTXInstruction::rn | PTXInstruction::rz;
		if (!rejects(conflictingRounding)) return false;
		for (int mode : {0, (int)PTXInstruction::rn, (int)PTXInstruction::rz, (int)PTXInstruction::rm, (int)PTXInstruction::rp}) {
			ins.modifier = mode;
			if (!ins.valid().empty()) { status << ins.valid() << "\n"; return false; }
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry fp64() { .reg .f64 %f<8>; " << ins.toString() << "; ret; }";
			Module parsed; parsed.load(source);
			std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
			for (int scenario : {0, 1, -1}) {
				cta->reset();
				for (int lane = 0; lane < 32; ++lane) {
					const int row = lane >> 2, k = lane & 3;
					cta->setRegAsF64(lane, 0, scenario ? (k == 0 ? scenario * 0x1p-53 : 0) : row - 2 * k + 1);
					cta->setRegAsF64(lane, 4, scenario ? 1 : 3 * k + row - 2);
					for (int i = 0; i < 2; ++i) cta->setRegAsF64(lane, 6 + i, scenario ? scenario : row * 8 + 2 * k + i);
				}
				cta->eval_Mma(cta->getActiveContext(), ins);
				for (int lane = 0; lane < 32; ++lane) for (int i = 0; i < 2; ++i) {
					const int row = lane >> 2, col = 2 * (lane & 3) + i;
					double expected = scenario ? scenario : row * 8 + col;
					if (!scenario) for (int k = 0; k < 4; ++k) expected += (row - 2 * k + 1) * (3 * k + col - 2);
					else if ((scenario > 0 && mode == PTXInstruction::rp) || (scenario < 0 && mode == PTXInstruction::rm)) expected = std::nextafter(expected, 2 * expected);
					if (cta->getRegAsF64(lane, i) != expected) { status << "f64 mma mismatch lane " << lane << " mode " << mode << "\n"; return false; }
				}
			}
		}

		PTXInstruction wrongLayout = ins;
		wrongLayout.mmaAColumnMajor = true;
		if (wrongLayout.valid().empty()) { status << "f64 mma accepted col.col\n"; return false; }
		ins.mmaShape = PTXInstruction::MmaM16N8K8;
		bool rejectedF64Shape = false;
		try { cta->eval_Mma(cta->getActiveContext(), ins); }
		catch (RuntimeException&) { rejectedF64Shape = true; }
		if (!rejectedF64Shape) { status << "unsupported f64 mma shape executed\n"; return false; }
		ins.mmaShape = PTXInstruction::MmaM8N8K4;

		// Four different products, nonuniform rows/columns/K, and A/D aliasing.
		ins.modifier = 0;
		ins.type = PTXOperand::f16;
		ins.a = vector(PTXOperand::f16, PTXOperand::f16x2, PTXOperand::v2, 0, 2);
		ins.b = vector(PTXOperand::f16, PTXOperand::b32, PTXOperand::v2, 4, 2);
		ins.c = vector(PTXOperand::f16, PTXOperand::f16x2, PTXOperand::v4, 6, 4);
		ins.d = vector(PTXOperand::f16, PTXOperand::b32, PTXOperand::v4, 0, 4);
		if (!ins.valid().empty()) { status << ins.valid() << "\n"; return false; }
		for (auto* operand : {&ins.a, &ins.b, &ins.c, &ins.d}) for (auto& element : operand->array) element.identifier = "%h" + std::to_string(element.reg);
		for (int bad : {0, 1, 2}) {
			PTXInstruction invalid = ins;
			if (bad == 0) invalid.a.type = PTXOperand::bf16;
			if (bad == 1) invalid.c.array.pop_back();
			if (bad == 2) invalid.modifier = PTXInstruction::satfinite;
			if (invalid.valid().empty()) { status << "invalid legacy f16 mma accepted\n"; return false; }
			bool rejected = false;
			try { cta->eval_Mma(cta->getActiveContext(), invalid); }
			catch (RuntimeException&) { rejected = true; }
			if (!rejected) { status << "invalid legacy f16 mma executed\n"; return false; }
		}
		std::stringstream source;
		source << ".version 8.0\n.target sm_86\n.entry fp16_884() { .reg .b32 %h<10>; " << ins.toString() << "; ret; }";
		Module parsed; parsed.load(source);
		std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
		cta->reset();
		for (int lane = 0; lane < 32; ++lane) {
			const int group = (lane / 4) % 4, row = lane % 4 + (lane / 16) * 4;
			for (int i = 0; i < 2; ++i) {
				const PTXU32 packed = f16Values[group + row + 2 * i + 1] | ((PTXU32)f16Values[group + row + 2 * i + 2] << 16);
				cta->setRegAsB32(lane, i, packed);
				cta->setRegAsB32(lane, 4 + i, f16Values[group + row + 4 * i] | ((PTXU32)f16Values[group + row + 4 * i + 2] << 16));
			}
			for (int i = 0; i < 4; ++i) cta->setRegAsB32(lane, 6 + i, f16Values[row + 2 * i] | ((PTXU32)f16Values[row + 2 * i + 1] << 16));
		}
		cta->eval_Mma(cta->getActiveContext(), ins);
		for (int lane = 0; lane < 32; ++lane) for (int col = 0; col < 8; ++col) {
			const int group = (lane / 4) % 4, row = lane % 4 + (lane / 16) * 4;
			int expected = row + col;
			for (int k = 0; k < 4; ++k) expected += (group + row + k + 1) * (group + col + 2 * k);
			const PTXU16 bits = (cta->getRegAsB32(lane, col / 2) >> (16 * (col & 1))) & 0xffff;
			if (std::ldexp(1024 + (bits & 1023), (bits >> 10) - 25) != expected) {
				status << "mma.m8n8k4.f16 mismatch lane " << lane << " column " << col << "\n"; return false;
			}
		}

		for (int layout = 0; layout < 4; ++layout) {
			ins.mmaAColumnMajor = layout & 1;
			ins.mmaBColumnMajor = layout & 2;
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry layouts() { .reg .b32 %h<10>; " << ins.toString() << "; ret; }";
			Module parsed; parsed.load(source);
			std::stringstream printed; parsed.write(printed); Module roundtrip; roundtrip.load(printed);
			if (printed.str().find(ins.toString()) == std::string::npos) { status << "mma layout lost in parsing\n"; return false; }
			cta->reset();
			for (int lane = 0; lane < 32; ++lane) {
				const int group = lane / 4 % 4, t = lane % 4, high = lane / 16 * 4;
				PTXU32 packedA[2] = {}, packedB[2] = {};
				for (int i = 0; i < 4; ++i) {
					const int aRow = ins.mmaAColumnMajor ? i + high : t + high;
					const int aK = ins.mmaAColumnMajor ? t : i;
					const int bK = ins.mmaBColumnMajor ? i : t;
					const int bCol = ins.mmaBColumnMajor ? t + high : i + high;
					packedA[i / 2] |= (PTXU32)f16Values[group + aRow + aK + 1] << (16 * (i % 2));
					packedB[i / 2] |= (PTXU32)f16Values[group + bCol + 2 * bK] << (16 * (i % 2));
				}
				for (int i = 0; i < 2; ++i) { cta->setRegAsB32(lane, i, packedA[i]); cta->setRegAsB32(lane, 4 + i, packedB[i]); }
				for (int i = 0; i < 4; ++i) cta->setRegAsB32(lane, 6 + i, f16Values[t + high + 2 * i] | ((PTXU32)f16Values[t + high + 2 * i + 1] << 16));
			}
			cta->eval_Mma(cta->getActiveContext(), ins);
			for (int lane = 0; lane < 32; ++lane) for (int col = 0; col < 8; ++col) {
				const int group = lane / 4 % 4, row = lane % 4 + lane / 16 * 4;
				int expected = row + col;
				for (int k = 0; k < 4; ++k) expected += (group + row + k + 1) * (group + col + 2 * k);
				const PTXU16 bits = cta->getRegAsB32(lane, col / 2) >> (16 * (col % 2));
				if (std::ldexp(1024 + (bits & 1023), (bits >> 10) - 25) != expected) { status << "mma layout mismatch " << layout << "\n"; return false; }
			}
		}

		ins.type = PTXOperand::f32;
		ins.mmaAColumnMajor = false; ins.mmaBColumnMajor = true;
		ins.a = vector(PTXOperand::f16, PTXOperand::b32, PTXOperand::v2, 8, 2);
		ins.b = vector(PTXOperand::f16, PTXOperand::b32, PTXOperand::v2, 10, 2);
		ins.c = ins.d = vector(PTXOperand::f32, PTXOperand::f32, PTXOperand::v8, 0, 8);
		for (auto* operand : {&ins.a, &ins.b, &ins.c, &ins.d}) for (auto& e : operand->array) e.identifier = "%f" + std::to_string(e.reg);
		std::stringstream fp32Source;
		fp32Source << ".version 8.0\n.target sm_86\n.entry fp32_884() { .reg .b32 %f<12>; " << ins.toString() << "; ret; }";
		Module fp32Parsed; fp32Parsed.load(fp32Source);
		std::stringstream fp32Printed; fp32Parsed.write(fp32Printed); Module fp32Roundtrip; fp32Roundtrip.load(fp32Printed);
		cta->reset();
		cta->functionCallStack.pushFrame(0, 12, 0, 0, 0, 0, 0);
		for (int lane = 0; lane < 32; ++lane) {
			const int group = lane / 4 % 4, t = lane % 4, high = lane / 16 * 4;
			for (int i = 0; i < 2; ++i) {
				cta->setRegAsB32(lane, 8 + i, f16Values[group + 2 * i + high + t + 1] | ((PTXU32)f16Values[group + 2 * i + high + t + 2] << 16));
				cta->setRegAsB32(lane, 10 + i, f16Values[group + t + high + 4 * i] | ((PTXU32)f16Values[group + t + high + 4 * i + 2] << 16));
			}
			for (int i = 0; i < 8; ++i) cta->setRegAsF32(lane, i, 100 * (lane % 2 + (i & 2) + high) + (i & 4) + (lane & 2) + (i & 1) + 0.25f);
		}
		cta->eval_Mma(cta->getActiveContext(), ins);
		for (int lane = 0; lane < 32; ++lane) for (int i = 0; i < 8; ++i) {
			const int group = lane / 4 % 4, row = lane % 2 + (i & 2) + lane / 16 * 4, col = (i & 4) + (lane & 2) + i % 2;
			float expected = 100 * row + col + 0.25f;
			for (int k = 0; k < 4; ++k) expected += (group + row + k + 1) * (group + col + 2 * k);
			if (cta->getRegAsF32(lane, i) != expected) { status << "legacy mma f32 mismatch lane " << lane << " register " << i << "\n"; return false; }
		}
		PTXInstruction illegalMixed = ins;
		illegalMixed.type = illegalMixed.d.type = PTXOperand::f16;
		illegalMixed.d = vector(PTXOperand::f16, PTXOperand::b32, PTXOperand::v4, 0, 4);
		if (illegalMixed.valid().empty()) { status << "mma accepted f32 C with f16 D\n"; return false; }
		ins.c = vector(PTXOperand::f16, PTXOperand::f16x2, PTXOperand::v4, 0, 4);
		for (auto& element : ins.c.array) element.identifier = "%f" + std::to_string(element.reg);
		for (int layout = 0; layout < 4; ++layout) {
			ins.mmaAColumnMajor = layout & 1;
			ins.mmaBColumnMajor = layout & 2;
			std::stringstream source;
			source << ".version 8.0\n.target sm_86\n.entry mixed() { .reg .b32 %f<12>; " << ins.toString() << "; ret; }";
			Module parsed; parsed.load(source);
			std::stringstream printed; parsed.write(printed);
			Module roundtrip; roundtrip.load(printed);
			for (int lane = 0; lane < 32; ++lane) {
				const int group = lane / 4 % 4, t = lane % 4, high = lane / 16 * 4;
				PTXU32 packedA[2] = {}, packedB[2] = {};
				for (int i = 0; i < 4; ++i) {
					const int aRow = ins.mmaAColumnMajor ? i + high : t + high;
					const int aK = ins.mmaAColumnMajor ? t : i;
					const int bK = ins.mmaBColumnMajor ? i : t;
					const int bCol = ins.mmaBColumnMajor ? t + high : i + high;
					packedA[i / 2] |= (PTXU32)f16Values[group + aRow + aK + 1] << (16 * (i % 2));
					packedB[i / 2] |= (PTXU32)f16Values[group + bCol + 2 * bK] << (16 * (i % 2));
				}
				for (int i = 0; i < 2; ++i) {
					cta->setRegAsB32(lane, 8 + i, packedA[i]);
					cta->setRegAsB32(lane, 10 + i, packedB[i]);
				}
				for (int i = 0; i < 4; ++i) {
					const PTXU32 packedC = f16Values[t + high + 2 * i] | ((PTXU32)f16Values[t + high + 2 * i + 1] << 16);
					cta->setRegAsB32(lane, i, packedC);
				}
			}
			cta->eval_Mma(cta->getActiveContext(), ins);
			for (int lane = 0; lane < 32; ++lane) {
				for (int i = 0; i < 8; ++i) {
					const int group = lane / 4 % 4;
					const int row = lane % 2 + (i & 2) + lane / 16 * 4;
					const int col = (i & 4) + (lane & 2) + i % 2;
					float expected = row + col;
					for (int k = 0; k < 4; ++k) expected += (group + row + k + 1) * (group + col + 2 * k);
					if (cta->getRegAsF32(lane, i) != expected) {
						status << "mixed mma mismatch layout " << layout << " lane " << lane << " register " << i << "\n";
						return false;
					}
				}
			}
		}
		// Keep a quarter-unit from FP16 C even when half-precision accumulation would lose it.
		for (int lane = 0; lane < 32; ++lane) {
			for (int i = 0; i < 4; ++i) cta->setRegAsB32(lane, i, 0x34003400u);
		}
		cta->eval_Mma(cta->getActiveContext(), ins);
		for (int lane = 0; lane < 32; ++lane) {
			for (int i = 0; i < 8; ++i) {
				const int group = lane / 4 % 4;
				const int row = lane % 2 + (i & 2) + lane / 16 * 4;
				const int col = (i & 4) + (lane & 2) + i % 2;
				float expected = 0.25f;
				for (int k = 0; k < 4; ++k) expected += (group + row + k + 1) * (group + col + 2 * k);
				if (cta->getRegAsF32(lane, i) != expected) { status << "mixed mma lost FP32 accumulation precision\n"; return false; }
			}
		}
		cta->functionCallStack.popFrame();

		return true;
	}

	// mma.m16n8k16.s32.s8/u8.s8/u8.s32: integer dot-product mma, PTX ISA 8.0
	// sec 9.7.15.5.9. Each case uses a constant-valued A/B fragment so the
	// expected accumulation (C + 16 * aVal * bVal) is a closed form -- still
	// exercises the real 16-term k-loop and the per-lane fragment layout.
	bool test_MmaInt8() {
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

		auto runCase = [&](PTXOperand::DataType aType, PTXOperand::DataType bType,
			int aVal, int bVal, PTXS32 cVal, bool satfinite,
			PTXS32 expected, const char *label, bool varyC = true) -> bool {
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Mma;
			ins.mmaShape = PTXInstruction::MmaM16N8K16;
			ins.type = PTXOperand::s32;
			ins.modifier = satfinite ? PTXInstruction::satfinite
				: PTXInstruction::Modifier_invalid;

			ins.d = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 0, 4);
			ins.c = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 6, 4);
			ins.a = vector(aType, PTXOperand::b32, PTXOperand::v2, 0, 2);
			ins.b = vector(bType, PTXOperand::b32, PTXOperand::v1, 4, 1);

			if (ins.valid() != "") {
				status << label << ": instruction rejected as invalid: "
					<< ins.valid() << "\n";
				return false;
			}

			PTXInstruction invalid = ins;
			invalid.modifier |= PTXInstruction::rn;
			if (invalid.valid().empty()) {
				status << "integer mma accepted a rounding modifier\n";
				return false;
			}
			const PTXU32 aByte = static_cast<PTXU32>(aVal) & 0xffu;
			const PTXU32 bByte = static_cast<PTXU32>(bVal) & 0xffu;
			const PTXU32 aPacked = aByte | (aByte << 8) | (aByte << 16) | (aByte << 24);
			const PTXU32 bPacked = bByte | (bByte << 8) | (bByte << 16) | (bByte << 24);

			cta->reset();
			for (int thread = 0; thread < threadCount; ++thread) {
				cta->setRegAsB32(thread, 0, aPacked);
				cta->setRegAsB32(thread, 1, aPacked);
				cta->setRegAsB32(thread, 4, bPacked);
				const int lane = thread & 31;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (int i = 0; i < 4; ++i) {
					const int row = groupID + (i >= 2 ? 8 : 0);
					const int col = threadInGroup * 2 + (i & 1);
					cta->setRegAsS32(thread, 6 + i,
						varyC ? cVal + row * 8 + col : cVal);
				}
			}

			cta->eval_Mma(cta->getActiveContext(), ins);

			for (int thread = 0; thread < threadCount; ++thread) {
				const int lane = thread & 31;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (int i = 0; i < 4; ++i) {
					const int row = groupID + (i >= 2 ? 8 : 0);
					const int col = threadInGroup * 2 + (i & 1);
					const PTXS32 got = cta->getRegAsS32(thread, i);
					const PTXS32 want = varyC ? expected + row * 8 + col
						: expected;
					if (got != want) {
						status << label << " incorrect [" << thread
							<< "]: got " << got << " want " << want << "\n";
						return false;
					}
				}
			}
			return true;
		};

		// s8 x s8: -3 * 5 * 16 terms = -240.
		if (!runCase(PTXOperand::s8, PTXOperand::s8, -3, 5, 0, false,
			-240, "mma s8xs8")) return false;

		// u8 x u8: byte 200 means 200 unsigned, -56 if wrongly sign-extended.
		// 200 * 200 * 16 = 640000.
		if (!runCase(PTXOperand::u8, PTXOperand::u8, 200, 200, 0, false,
			640000, "mma u8xu8")) return false;

		// mixed s8 x u8: -3 (signed) * 200 (unsigned) * 16 = -9600.
		if (!runCase(PTXOperand::s8, PTXOperand::u8, -3, 200, 0, false,
			-9600, "mma s8xu8")) return false;

		// satfinite: large positive C plus a large positive product overflows
		// s32 and must clamp to INT32_MAX rather than wrap.
		if (!runCase(PTXOperand::s8, PTXOperand::s8, 127, 127,
			(std::numeric_limits<PTXS32>::max)() - 100, true,
			(std::numeric_limits<PTXS32>::max)(), "mma satfinite positive",
			false))
			return false;

		// satfinite: large negative C plus a large negative product
		// underflows s32 and must clamp to INT32_MIN.
		if (!runCase(PTXOperand::s8, PTXOperand::s8, 127, -128,
			(std::numeric_limits<PTXS32>::min)() + 100, true,
			(std::numeric_limits<PTXS32>::min)(), "mma satfinite negative",
			false))
			return false;

		// Validation: integer mma requires an s32 accumulator, not f32.
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Mma;
			ins.mmaShape = PTXInstruction::MmaM16N8K16;
			ins.type = PTXOperand::f32;
			ins.d = vector(PTXOperand::f32, PTXOperand::f32, PTXOperand::v4, 0, 4);
			ins.c = vector(PTXOperand::f32, PTXOperand::f32, PTXOperand::v4, 6, 4);
			ins.a = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v2, 0, 2);
			ins.b = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v1, 4, 1);
			if (ins.valid() == "") {
				status << "mma with s8 inputs and f32 accumulator "
					"unexpectedly valid\n";
				return false;
			}
		}

		// Direct execution must not interpret unsupported shapes as m16n8k16.
		for (auto shape : {PTXInstruction::MmaShape_Invalid, PTXInstruction::MmaM16N8K8}) {
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Mma;
			ins.mmaShape = shape;
			ins.type = PTXOperand::s32;
			ins.d = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 0, 4);
			ins.c = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 6, 4);
			ins.a = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v2, 0, 2);
			ins.b = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v1, 4, 1);
			cta->reset();
			bool rejected = false;
			try { cta->eval_Mma(cta->getActiveContext(), ins); }
			catch (const RuntimeException&) { rejected = true; }
			if (!rejected) {
				status << "integer mma executed an unsupported shape\n";
				return false;
			}
		}

		for (auto aType : {PTXOperand::s4, PTXOperand::u4}) for (auto bType : {PTXOperand::s4, PTXOperand::u4}) {
			for (bool saturate : {false, true}) for (int sign : {-1, 1})
			for (auto shape : {PTXInstruction::MmaM8N8K32, PTXInstruction::MmaM16N8K32, PTXInstruction::MmaM16N8K64}) {
				const bool wide = shape != PTXInstruction::MmaM8N8K32;
				const bool k64 = shape == PTXInstruction::MmaM16N8K64;
				PTXInstruction ins;
				ins.opcode = PTXInstruction::Mma; ins.mmaShape = shape;
				ins.type = PTXOperand::s32; ins.modifier = saturate ? PTXInstruction::satfinite : 0;
				ins.a = vector(aType, PTXOperand::b32, k64 ? PTXOperand::v4 : wide ? PTXOperand::v2 : PTXOperand::v1, 4, k64 ? 4 : wide ? 2 : 1);
				ins.b = vector(bType, PTXOperand::b32, k64 ? PTXOperand::v2 : PTXOperand::v1, 8, k64 ? 2 : 1);
				ins.c = ins.d = vector(PTXOperand::s32, PTXOperand::s32, wide ? PTXOperand::v4 : PTXOperand::v2, 0, wide ? 4 : 2);
				if (k64) for (auto invalidType : {PTXOperand::s8, PTXOperand::u8, PTXOperand::f16, PTXOperand::tf32}) {
					PTXInstruction invalid = ins;
					invalid.a.type = invalid.b.type = invalidType;
					if (invalid.valid().empty()) { status << "m16n8k64 accepted non-int4 inputs\n"; return false; }
				}
				const PTXS32 base = sign > 0 ? INT32_MAX - 200 : INT32_MIN + 200;
				auto decode = [](int value, PTXOperand::DataType type) { return type == PTXOperand::s4 && value >= 8 ? value - 16 : value; };
				cta->reset();
				for (int lane = 0; lane < 32; ++lane) {
					const int group = lane / 4, t = lane % 4;
					PTXU32 a = 0, b = 0;
					PTXU32 upperA = 0, upperB = 0;
					for (int i = 0; i < 8; ++i) {
						a |= (PTXU32)((group + 8 * t + i + (t >= 2 ? 3 : 0)) % 16) << (4 * i);
						b |= (PTXU32)((3 * (8 * t + i) + group + (t >= 2 ? 5 : 0)) % 16) << (4 * i);
						upperA |= (PTXU32)((group + 8 * t + i + (t >= 2 ? 3 : 0) + 6) % 16) << (4 * i);
						upperB |= (PTXU32)((3 * (8 * t + i) + group + (t >= 2 ? 5 : 0) + 10) % 16) << (4 * i);
					}
					cta->setRegAsB32(lane, 4, a); cta->setRegAsB32(lane, 8, b);
					if (wide) cta->setRegAsB32(lane, 5, a ^ 0x88888888u);
					if (k64) {
						cta->setRegAsB32(lane, 6, upperA); cta->setRegAsB32(lane, 7, upperA ^ 0x88888888u);
						cta->setRegAsB32(lane, 9, upperB);
					}
					for (int i = 0; i < (wide ? 4 : 2); ++i) cta->setRegAsS32(lane, i,
						base + (group + (i >= 2 ? 8 : 0)) * 8 + 2 * t + (i & 1));
				}
				cta->eval_Mma(cta->getActiveContext(), ins);
				for (int lane = 0; lane < 32; ++lane) for (int i = 0; i < (wide ? 4 : 2); ++i) {
					const int row = lane / 4 + (i >= 2 ? 8 : 0), col = 2 * (lane % 4) + (i & 1);
					int64_t expected = (int64_t)base + row * 8 + col;
					for (int k = 0; k < (k64 ? 64 : 32); ++k) expected +=
						decode((row + k + (k / 16) * 3) % 16, aType) *
						decode((3 * k + col + (k / 16) * 5) % 16, bType);
					if (saturate) expected = std::max<int64_t>(INT32_MIN, std::min<int64_t>(INT32_MAX, expected));
					const PTXS32 want = hydrazine::bit_cast<PTXS32>((PTXU32)expected);
					if (cta->getRegAsS32(lane, i) != want) { status << "int4 mma shape " << shape << " mismatch lane " << lane << " register " << i << "\n"; return false; }
				}
			}
		}
		return true;
	}

	// mma.m8n8k16/m16n8k32.s32.s8/u8.s8/u8.s32: the two remaining integer mma
	// shapes beyond m16n8k16 (PTX ISA 8.0 sec 9.7.15.5.3 / 9.7.15.5.10).
	// Same closed-form approach as test_MmaInt8: constant-valued A/B
	// fragments make the expected accumulation (C + k * aVal * bVal) a
	// closed form while a row/col-varying C still exercises the real
	// per-lane C/D fragment layout and the shape's k-loop bound.
	bool test_MmaInt8WideShapes() {
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

		// cdRowCol: m8n8k16 has a 8x8 C/D with 2 registers per lane;
		// m16n8k16/m16n8k32 share the same 16x8 C/D with 4 registers per
		// lane -- reuse that mapping for m16n8k32 directly.
		auto runCase = [&](PTXInstruction::MmaShape shape, int k,
			PTXOperand::Vec aVec, int aCount, PTXOperand::Vec bVec, int bCount,
			PTXOperand::Vec cdVec, int cdCount,
			auto cdRowCol,
			PTXOperand::DataType aType, PTXOperand::DataType bType,
			int aVal, int bVal, PTXS32 cVal, bool satfinite,
			PTXS32 expected, const char *label, bool varyC = true) -> bool {
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Mma;
			ins.mmaShape = shape;
			ins.type = PTXOperand::s32;
			ins.modifier = satfinite ? PTXInstruction::satfinite
				: PTXInstruction::Modifier_invalid;

			// Register numbering mirrors test_MmaInt8: D and A share the
			// low registers (A is fully read into local matrices before D
			// is written, so reuse is safe), keeping everything within the
			// small register file the embedded test kernel declares.
			ins.d = vector(PTXOperand::s32, PTXOperand::s32, cdVec, 0, cdCount);
			ins.a = vector(aType, PTXOperand::b32, aVec, 0, aCount);
			ins.b = vector(bType, PTXOperand::b32, bVec, 4, bCount);
			ins.c = vector(PTXOperand::s32, PTXOperand::s32, cdVec, 6, cdCount);

			if (ins.valid() != "") {
				status << label << ": instruction rejected as invalid: "
					<< ins.valid() << "\n";
				return false;
			}

			const PTXU32 aByte = static_cast<PTXU32>(aVal) & 0xffu;
			const PTXU32 bByte = static_cast<PTXU32>(bVal) & 0xffu;
			const PTXU32 aPacked = aByte | (aByte << 8) | (aByte << 16) | (aByte << 24);
			const PTXU32 bPacked = bByte | (bByte << 8) | (bByte << 16) | (bByte << 24);

			cta->reset();
			for (int thread = 0; thread < threadCount; ++thread) {
				for (int i = 0; i < aCount; ++i) {
					cta->setRegAsB32(thread, i, aPacked);
				}
				for (int i = 0; i < bCount; ++i) {
					cta->setRegAsB32(thread, 4 + i, bPacked);
				}
				const int lane = thread & 31;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (int i = 0; i < cdCount; ++i) {
					int row, col;
					cdRowCol(groupID, threadInGroup, i, row, col);
					cta->setRegAsS32(thread, 6 + i,
						varyC ? cVal + row * 8 + col : cVal);
				}
			}

			cta->eval_Mma(cta->getActiveContext(), ins);

			for (int thread = 0; thread < threadCount; ++thread) {
				const int lane = thread & 31;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (int i = 0; i < cdCount; ++i) {
					int row, col;
					cdRowCol(groupID, threadInGroup, i, row, col);
					const PTXS32 got = cta->getRegAsS32(thread, i);
					const PTXS32 want = varyC ? expected + row * 8 + col
						: expected;
					if (got != want) {
						status << label << " incorrect [" << thread
							<< "]: got " << got << " want " << want << "\n";
						return false;
					}
				}
			}
			return true;
		};

		auto cdRowCol8x8 = [](int groupID, int threadInGroup, int i,
			int &row, int &col) {
			row = groupID;
			col = threadInGroup * 2 + i;
		};
		auto cdRowCol16x8 = [](int groupID, int threadInGroup, int i,
			int &row, int &col) {
			row = groupID + (i >= 2 ? 8 : 0);
			col = threadInGroup * 2 + (i & 1);
		};

		// m8n8k16: 1 A register, 1 B register, 2 C/D registers, k = 16.
		if (!runCase(PTXInstruction::MmaM8N8K16, 16,
			PTXOperand::v1, 1, PTXOperand::v1, 1, PTXOperand::v2, 2, cdRowCol8x8,
			PTXOperand::s8, PTXOperand::s8, -3, 5, 0, false,
			-240, "mma.m8n8k16 s8xs8")) return false;
		if (!runCase(PTXInstruction::MmaM8N8K16, 16,
			PTXOperand::v1, 1, PTXOperand::v1, 1, PTXOperand::v2, 2, cdRowCol8x8,
			PTXOperand::u8, PTXOperand::u8, 200, 200, 0, false,
			640000, "mma.m8n8k16 u8xu8")) return false;
		if (!runCase(PTXInstruction::MmaM8N8K16, 16,
			PTXOperand::v1, 1, PTXOperand::v1, 1, PTXOperand::v2, 2, cdRowCol8x8,
			PTXOperand::s8, PTXOperand::u8, -3, 200, 0, false,
			-9600, "mma.m8n8k16 s8xu8")) return false;
		if (!runCase(PTXInstruction::MmaM8N8K16, 16,
			PTXOperand::v1, 1, PTXOperand::v1, 1, PTXOperand::v2, 2, cdRowCol8x8,
			PTXOperand::s8, PTXOperand::s8, 127, 127,
			(std::numeric_limits<PTXS32>::max)() - 100, true,
			(std::numeric_limits<PTXS32>::max)(), "mma.m8n8k16 satfinite positive",
			false)) return false;
		if (!runCase(PTXInstruction::MmaM8N8K16, 16,
			PTXOperand::v1, 1, PTXOperand::v1, 1, PTXOperand::v2, 2, cdRowCol8x8,
			PTXOperand::s8, PTXOperand::s8, 127, -128,
			(std::numeric_limits<PTXS32>::min)() + 100, true,
			(std::numeric_limits<PTXS32>::min)(), "mma.m8n8k16 satfinite negative",
			false)) return false;

		// m16n8k32: 4 A registers, 2 B registers, 4 C/D registers, k = 32.
		if (!runCase(PTXInstruction::MmaM16N8K32, 32,
			PTXOperand::v4, 4, PTXOperand::v2, 2, PTXOperand::v4, 4, cdRowCol16x8,
			PTXOperand::s8, PTXOperand::s8, -3, 5, 0, false,
			-480, "mma.m16n8k32 s8xs8")) return false;
		if (!runCase(PTXInstruction::MmaM16N8K32, 32,
			PTXOperand::v4, 4, PTXOperand::v2, 2, PTXOperand::v4, 4, cdRowCol16x8,
			PTXOperand::u8, PTXOperand::u8, 200, 200, 0, false,
			1280000, "mma.m16n8k32 u8xu8")) return false;
		if (!runCase(PTXInstruction::MmaM16N8K32, 32,
			PTXOperand::v4, 4, PTXOperand::v2, 2, PTXOperand::v4, 4, cdRowCol16x8,
			PTXOperand::s8, PTXOperand::u8, -3, 200, 0, false,
			-19200, "mma.m16n8k32 s8xu8")) return false;
		if (!runCase(PTXInstruction::MmaM16N8K32, 32,
			PTXOperand::v4, 4, PTXOperand::v2, 2, PTXOperand::v4, 4, cdRowCol16x8,
			PTXOperand::s8, PTXOperand::s8, 127, 127,
			(std::numeric_limits<PTXS32>::max)() - 100, true,
			(std::numeric_limits<PTXS32>::max)(), "mma.m16n8k32 satfinite positive",
			false)) return false;
		if (!runCase(PTXInstruction::MmaM16N8K32, 32,
			PTXOperand::v4, 4, PTXOperand::v2, 2, PTXOperand::v4, 4, cdRowCol16x8,
			PTXOperand::s8, PTXOperand::s8, 127, -128,
			(std::numeric_limits<PTXS32>::min)() + 100, true,
			(std::numeric_limits<PTXS32>::min)(), "mma.m16n8k32 satfinite negative",
			false)) return false;

		// Validation: m8n8k16 rejects m16n8k16-sized fragments (wrong vec).
		{
			PTXInstruction ins;
			ins.opcode = PTXInstruction::Mma;
			ins.mmaShape = PTXInstruction::MmaM8N8K16;
			ins.type = PTXOperand::s32;
			ins.d = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 0, 4);
			ins.c = vector(PTXOperand::s32, PTXOperand::s32, PTXOperand::v4, 6, 4);
			ins.a = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v2, 0, 2);
			ins.b = vector(PTXOperand::s8, PTXOperand::b32, PTXOperand::v1, 4, 1);
			if (ins.valid() == "") {
				status << "mma.m8n8k16 with m16n8k16-sized fragments "
					"unexpectedly valid\n";
				return false;
			}
		}

		return true;
	}

	bool test_Lg2() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_lg2() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  lg2.approx.f32 f0, f1;\n"
			<< "  lg2.approx.ftz.f32 f0, f1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 lg2 forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Lg2;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx;

		// f32
		//
		if (result) {
			for (int i = 0; i < threadCount; i++) {
				const PTXF32 value = 0.51f
					+ (PTXF32)i / (PTXF32)threadCount * 1.48f;
				cta->setRegAsF32(i, 0, value);
				cta->setRegAsF32(i, 2, 0);
			}
			cta->eval_Lg2(cta->getActiveContext(), ins);
			for (int i = 0; i < threadCount; i++) {
				const PTXF32 value = 0.51f
					+ (PTXF32)i / (PTXF32)threadCount * 1.48f;
				const double expected = std::log2((double)value);
				if (std::fabs(cta->getRegAsF32(i, 2) - expected)
					> std::ldexp(1.0, -22)) {
					result = false;
					status << "lg2.f32 incorrect [" << i
						<< "] - expected: " << expected
						<< ", got " << cta->getRegAsF32(i, 2) << "\n";
					break;
				}
			}
		}

		cta->setRegAsF32(0, 0, -std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(1, 0, -1.0f);
		cta->setRegAsF32(2, 0, -0.0f);
		cta->setRegAsF32(3, 0, 0.0f);
		cta->setRegAsF32(4, 0, std::numeric_limits<PTXF32>::infinity());
		cta->setRegAsF32(5, 0, std::numeric_limits<PTXF32>::quiet_NaN());
		cta->eval_Lg2(cta->getActiveContext(), ins);
		if (!hydrazine::isnan(cta->getRegAsF32(0, 2))
			|| !hydrazine::isnan(cta->getRegAsF32(1, 2))
			|| cta->getRegAsF32(2, 2) != -std::numeric_limits<PTXF32>::infinity()
			|| cta->getRegAsF32(3, 2) != -std::numeric_limits<PTXF32>::infinity()
			|| cta->getRegAsF32(4, 2) != std::numeric_limits<PTXF32>::infinity()
			|| !hydrazine::isnan(cta->getRegAsF32(5, 2)))
			return false;

		const PTXF32 subnormal = std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 0, subnormal);
		cta->eval_Lg2(cta->getActiveContext(), ins);
		if (std::fabs((cta->getRegAsF32(0, 2) + 149.0) / 149.0)
			> std::ldexp(1.0, -22)) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->eval_Lg2(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2)
			!= -std::numeric_limits<PTXF32>::infinity()) return false;

		return result;
	}

	bool test_Sqrt() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_sqrt() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  .reg .f64 d0, d1;\n"
			<< "  sqrt.approx.f32 f0, f1;\n"
			<< "  sqrt.rz.ftz.f32 f0, f1;\n"
			<< "  sqrt.rp.f64 d0, d1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 sqrt forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Sqrt;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::f64;
		ins.a.type = ins.d.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::rp;
		if (!ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		if (ins.valid().empty()) return false;

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

		const int modes[] = {PTXInstruction::rn, PTXInstruction::rz,
			PTXInstruction::rm, PTXInstruction::rp};
		const PTXU32 expected32[] = {
			0x3fb504f3u, 0x3fb504f3u, 0x3fb504f3u, 0x3fb504f4u};
		const PTXU64 expected64[] = {0x3ff6a09e667f3bcdull,
			0x3ff6a09e667f3bccull, 0x3ff6a09e667f3bccull,
			0x3ff6a09e667f3bcdull};
		for (unsigned int i = 0; i < 4; ++i) {
			ins.modifier = modes[i];
			ins.type = PTXOperand::f32;
			ins.a.type = ins.d.type = PTXOperand::f32;
			cta->setRegAsF32(0, 0, 2.0f);
			cta->eval_Sqrt(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2))
				!= expected32[i]) return false;

			ins.type = PTXOperand::f64;
			ins.a.type = ins.d.type = PTXOperand::f64;
			cta->setRegAsF64(0, 0, 2.0);
			cta->eval_Sqrt(cta->getActiveContext(), ins);
			if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2))
				!= expected64[i]) return false;
		}

		ins.type = PTXOperand::f32;
		ins.a.type = ins.d.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		const PTXF32 subnormal = std::numeric_limits<PTXF32>::min()
			- std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 0, subnormal);
		cta->setRegAsF32(1, 0, -subnormal);
		cta->eval_Sqrt(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(0, 2)) != 0
			|| hydrazine::bit_cast<PTXU32>(cta->getRegAsF32(1, 2))
				!= 0x80000000u) return false;

		ins.modifier = PTXInstruction::approx;
		cta->setRegAsF32(0, 0, subnormal);
		cta->eval_Sqrt(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2) == 0.0f) return false;

		return result;
	}

	bool test_Rsqrt() {
		bool result = true;
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_rsqrt() {\n"
			<< "  .reg .f32 f0, f1;\n"
			<< "  .reg .f64 d0, d1;\n"
			<< "  rsqrt.approx.f32 f0, f1;\n"
			<< "  rsqrt.approx.ftz.f32 f0, f1;\n"
			<< "  rsqrt.approx.f64 d0, d1;\n"
			<< "  rsqrt.approx.ftz.f64 d0, d1;\n"
			<< "  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const std::exception& error) {
			status << "failed to parse PTX 8.0 rsqrt forms: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Rsqrt;
		ins.type = PTXOperand::f32;
		ins.a = reg("r1", PTXOperand::f32, 0);
		ins.d = reg("r3", PTXOperand::f32, 2);
		ins.modifier = PTXInstruction::approx;
		if (!ins.valid().empty()) return false;
		ins.modifier = 0;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::approx | PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::f64;
		ins.a.type = ins.d.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		if (!ins.valid().empty()) return false;

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

		ins.type = PTXOperand::f64;
		ins.a.type = ins.d.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::approx | PTXInstruction::ftz;
		cta->setRegAsF64(0, 0, 3.0);
		cta->setRegAsF64(1, 0, std::numeric_limits<PTXF64>::quiet_NaN());
		cta->setRegAsF64(2, 0, -0.0);
		cta->setRegAsF64(3, 0, std::numeric_limits<PTXF64>::denorm_min());
		cta->eval_Rsqrt(cta->getActiveContext(), ins);
		if (hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(0, 2))
				!= 0x3fe279a700000000ull
			|| hydrazine::bit_cast<PTXU64>(cta->getRegAsF64(1, 2))
				!= 0x7fffffff00000000ull
			|| cta->getRegAsF64(2, 2)
				!= -std::numeric_limits<PTXF64>::infinity()
			|| cta->getRegAsF64(3, 2)
				!= std::numeric_limits<PTXF64>::infinity()) return false;

		ins.type = PTXOperand::f32;
		ins.a.type = ins.d.type = PTXOperand::f32;
		const PTXF32 subnormal = std::numeric_limits<PTXF32>::min()
			- std::numeric_limits<PTXF32>::denorm_min();
		cta->setRegAsF32(0, 0, subnormal);
		ins.modifier = PTXInstruction::approx;
		cta->eval_Rsqrt(cta->getActiveContext(), ins);
		if (!std::isfinite(cta->getRegAsF32(0, 2))) return false;
		ins.modifier |= PTXInstruction::ftz;
		cta->eval_Rsqrt(cta->getActiveContext(), ins);
		if (cta->getRegAsF32(0, 2)
			!= std::numeric_limits<PTXF32>::infinity()) return false;

		return result;
	}

	bool test_Dp() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_dp() {\n"
			<< "  .reg .b32 d0, a0, b0, c0;\n"
			<< "  dp4a.u32.u32 d0, a0, b0, c0;\n"
			<< "  dp4a.u32.s32 d0, a0, b0, c0;\n"
			<< "  dp2a.lo.u32.u32 d0, a0, b0, c0;\n"
			<< "  dp2a.hi.u32.s32 d0, a0, b0, c0;\n  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse dp2a/dp4a examples: " << error.what() << "\n";
			return false;
		}

		struct Case {
			PTXInstruction::Opcode opcode;
			PTXOperand::DataType aType, bType;
			unsigned int mode;
			PTXU32 a, b, c, expected;
		};
		const Case cases[] = {
			{PTXInstruction::Dp4a, PTXOperand::u32, PTXOperand::u32, 0,
				0x04030201u, 0x08070605u, 10, 80},
			{PTXInstruction::Dp4a, PTXOperand::u32, PTXOperand::s32, 0,
				0x04030201u, 0xfcfdfeffu, 10, 0xffffffecu},
			{PTXInstruction::Dp2a, PTXOperand::u32, PTXOperand::u32, PTXInstruction::lo,
				0x00030002u, 0x64640504u, 1, 24},
			{PTXInstruction::Dp2a, PTXOperand::s32, PTXOperand::u32, PTXInstruction::hi,
				0x0003fffeu, 0x05040000u, 1, 8}
		};
		PTXInstruction ins;
		ins.d = reg("d", PTXOperand::b32, 0);
		for (const Case& test : cases) {
			ins.opcode = test.opcode;
			ins.type = test.aType;
			ins.bType = test.bType;
			ins.modifier = test.mode;
			ins.a = imm_uint("a", test.aType, test.a);
			ins.b = imm_uint("b", test.bType, test.b);
			ins.c = imm_uint("c", PTXOperand::b32, test.c);
			cta->eval_Dp(cta->getActiveContext(), ins);
			for (int thread = 0; thread < threadCount; ++thread) {
				if (cta->getRegAsU32(thread, 0) != test.expected) return false;
			}
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//
	// logical and shift instructions
	//
	/////////////////////////////////////////////////////////////////////////////////////////////////

	bool test_Fns() {
		struct Case {
			PTXU32 base;
			PTXS32 offset;
			PTXU32 expected;
		};
		const Case cases[] = {
			{3,  1, 3},
			{3, -1, 3},
			{2,  1, 3},
			{2, -1, 1}
		};

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Fns;
		ins.type = PTXOperand::b32;
		ins.d = reg("d", PTXOperand::b32, 0);
		ins.a = imm_uint("mask", PTXOperand::b32, 0xaaaaaaaau);

		for (const Case& test : cases) {
			ins.b = imm_uint("base", PTXOperand::b32, test.base);
			ins.c = imm_int("offset", PTXOperand::s32, test.offset);
			cta->eval_Fns(cta->getActiveContext(), ins);

			for (int thread = 0; thread < threadCount; ++thread) {
				PTXU32 actual = cta->getRegAsB32(thread, ins.d.reg);
				if (actual != test.expected) {
					status << "fns.b32 failed for base " << test.base
						<< ", offset " << test.offset << ": expected "
						<< test.expected << ", got " << actual << "\n";
					return false;
				}
			}
		}
		return true;
	}

	bool test_Szext() {
		std::stringstream ptx;
		ptx << ".version 8.0\n"
			<< ".target sm_86\n"
			<< ".address_size 64\n"
			<< ".visible .entry test_szext() {\n"
			<< "  .reg .b32 rd;\n"
			<< "  .reg .s32 ra;\n"
			<< "  .reg .u32 rb;\n"
			<< "  szext.clamp.s32 rd, ra, rb;\n"
			<< "  szext.wrap.u32 rd, 0xffffffff, 0;\n"
			<< "  ret;\n"
			<< "}\n";
		Module parsed;
		try {
			parsed.load(ptx);
		}
		catch (const hydrazine::Exception& error) {
			status << "failed to parse PTX 8.0 szext examples: "
				<< error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Szext;
		ins.type = PTXOperand::s32;
		ins.shiftMode = PTXInstruction::ShiftMode::Clamp;
		ins.d = reg("rd", PTXOperand::b32, 0);
		ins.a = reg("ra", PTXOperand::s32, 1);
		ins.b = reg("rb", PTXOperand::u32, 2);
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU32(thread, 1, 0x80);
			cta->setRegAsU32(thread, 2, 8);
		}
		cta->eval_Szext(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0xffffff80u) return false;
		}

		ins.type = PTXOperand::u32;
		ins.shiftMode = PTXInstruction::ShiftMode::Wrap;
		ins.a = imm_uint("a", PTXOperand::u32, 0xffffffffu);
		ins.b = imm_uint("b", PTXOperand::u32, 0);
		cta->eval_Szext(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0) return false;
		}

		return true;
	}

	bool test_Bmsk() {
		std::stringstream ptx;
		ptx << ".version 8.0\n"
			<< ".target sm_86\n"
			<< ".address_size 64\n"
			<< ".visible .entry test_bmsk() {\n"
			<< "  .reg .b32 rd, ra, rb;\n"
			<< "  bmsk.clamp.b32 rd, ra, rb;\n"
			<< "  bmsk.wrap.b32 rd, 1, 2;\n"
			<< "  ret;\n"
			<< "}\n";
		Module parsed;
		try {
			parsed.load(ptx);
		}
		catch (const hydrazine::Exception& error) {
			status << "failed to parse PTX 8.0 bmsk examples: "
				<< error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Bmsk;
		ins.type = PTXOperand::b32;
		ins.shiftMode = PTXInstruction::ShiftMode::Clamp;
		ins.d = reg("rd", PTXOperand::b32, 0);
		ins.a = reg("ra", PTXOperand::b32, 1);
		ins.b = reg("rb", PTXOperand::b32, 2);
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU32(thread, 1, 1);
			cta->setRegAsU32(thread, 2, 2);
		}
		cta->eval_Bmsk(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0x00000006u) return false;
		}

		ins.shiftMode = PTXInstruction::ShiftMode::Wrap;
		ins.a = imm_uint("a", PTXOperand::b32, 1);
		ins.b = imm_uint("b", PTXOperand::b32, 2);
		cta->eval_Bmsk(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0x00000006u) return false;
		}

		return true;
	}

	bool test_Bfe() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Bfe;
		ins.b = imm_uint("pos", PTXOperand::u32, 0);
		ins.c = imm_uint("len", PTXOperand::u32, 0);
		ins.d = reg("d", PTXOperand::u32, 0);
		ins.a = reg("a", PTXOperand::u32, 1);
		ins.type = PTXOperand::u32;
		if (!ins.valid().empty()) return false;
		PTXInstruction invalid = ins;
		invalid.a.addressMode = PTXOperand::Immediate;
		invalid.b = reg("pos", PTXOperand::f32, 2);
		if (invalid.valid().empty()) return false;

		auto check32 = [&](PTXOperand::DataType type, PTXU32 value,
			PTXU32 pos, PTXU32 len, PTXU32 expected) {
			ins.type = type;
			ins.d.type = type;
			ins.a.type = type;
			ins.b.imm_uint = pos;
			ins.c.imm_uint = len;
			cta->reset();
			for (int t = 0; t < threadCount; ++t) cta->setRegAsU32(t, 1, value);
			cta->eval_Bfe(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; ++t)
				if (cta->getRegAsU32(t, 0) != expected) return false;
			return true;
		};
		if (!check32(PTXOperand::u32, 0xD2, 1, 4, 0x9)) return false;
		if (!check32(PTXOperand::u32, 0x12345678, 0, 32, 0x12345678)) return false;
		if (!check32(PTXOperand::u32, 0xffffffff, 0, 0, 0)) return false;
		if (!check32(PTXOperand::u32, 0xD2, 0x101, 0x104, 0x9)) return false;
		if (!check32(PTXOperand::s32, 0x000000f0, 4, 4, 0xffffffffu)) return false;
		if (!check32(PTXOperand::s32, 0x80000000, 30, 4, 0xfffffffeu)) return false;
		if (!check32(PTXOperand::s32, 0x80000000, 40, 3, 0xffffffffu)) return false;

		ins.type = PTXOperand::u64;
		ins.d.type = PTXOperand::u64;
		ins.a.type = PTXOperand::u64;
		ins.b.imm_uint = 4;
		ins.c.imm_uint = 8;
		cta->reset();
		for (int t = 0; t < threadCount; ++t) cta->setRegAsU64(t, 1, 0x123456789abcdef0ULL);
		cta->eval_Bfe(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0xef) return false;
		ins.b.imm_uint = 0;
		ins.c.imm_uint = 64;
		cta->eval_Bfe(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0x123456789abcdef0ULL) return false;
		ins.type = PTXOperand::s64;
		ins.d.type = PTXOperand::s64;
		ins.a.type = PTXOperand::s64;
		ins.b.imm_uint = 62;
		ins.c.imm_uint = 4;
		cta->reset();
		for (int t = 0; t < threadCount; ++t) cta->setRegAsU64(t, 1, 0x8000000000000000ULL);
		cta->eval_Bfe(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0xfffffffffffffffeULL) return false;

		ins.type = PTXOperand::u32;
		ins.d.type = PTXOperand::u32;
		ins.a.type = PTXOperand::u32;
		ins.b.imm_uint = 0;
		ins.c.imm_uint = 4;
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 3;
		cta->reset();
		for (int t = 0; t < threadCount; ++t) {
			cta->setRegAsU32(t, 1, 0xf0);
			cta->setRegAsU32(t, 0, 0xdeadbeef);
			cta->setRegAsPredicate(t, 3, false);
		}
		cta->eval_Bfe(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 0) != 0xdeadbeef) return false;
		return true;
	}

	bool test_Bfi() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Bfi;
		ins.type = PTXOperand::b32;
		ins.d = reg("d", PTXOperand::b32, 0);
		ins.pq = imm_uint("pq", PTXOperand::b32, 0xf);
		ins.a = imm_uint("a", PTXOperand::b32, 0xa5);
		ins.b = imm_uint("pos", PTXOperand::u32, 0x104);
		ins.c = imm_uint("len", PTXOperand::u32, 4);
		if (!ins.valid().empty()) return false;

		auto check32 = [&](PTXU32 pq, PTXU32 a, PTXU32 pos,
			PTXU32 len, PTXU32 expected) {
			ins.pq.imm_uint = pq;
			ins.a.imm_uint = a;
			ins.b.imm_uint = pos;
			ins.c.imm_uint = len;
			cta->reset();
			cta->eval_Bfi(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; ++t)
				if (cta->getRegAsU32(t, 0) != expected) return false;
			return true;
		};
		if (!check32(0xf, 0xa5, 0x104, 4, 0xf5)) return false;
		if (!check32(0xf, 0x12345678, 4, 0x104, 0x123456f8)) return false;
		if (!check32(0xf, 0x12345678, 4, 0x100, 0x12345678)) return false;
		if (!check32(0xffffffff, 0, 30, 4, 0xc0000000)) return false;
		if (!check32(0xffffffff, 0x12345678, 32, 4, 0x12345678)) return false;

		PTXInstruction invalid = ins;
		invalid.c = reg("len", PTXOperand::u64, 1);
		if (invalid.valid().empty()) return false;
		invalid.c = reg("len", PTXOperand::f32, 1);
		if (invalid.valid().empty()) return false;

		ins.type = PTXOperand::b64;
		ins.d.type = PTXOperand::b64;
		ins.pq = imm_uint("pq", PTXOperand::b64, 0xf);
		ins.a = imm_uint("a", PTXOperand::b64, 0x123456789abcde00ULL);
		ins.b.imm_uint = 0x104;
		ins.c.imm_uint = 4;
		cta->reset();
		cta->eval_Bfi(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0x123456789abcdef0ULL) return false;

		ins.type = PTXOperand::b32;
		ins.d.type = PTXOperand::b32;
		ins.pq = imm_uint("pq", PTXOperand::b32, 0xf);
		ins.a = imm_uint("a", PTXOperand::b32, 0xa5);
		ins.b = imm_uint("pos", PTXOperand::u32, 0);
		ins.c = imm_uint("len", PTXOperand::u32, 4);
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 3;
		cta->reset();
		for (int t = 0; t < threadCount; ++t) {
			cta->setRegAsU32(t, 0, 0xdeadbeef);
			cta->setRegAsPredicate(t, 3, false);
		}
		cta->eval_Bfi(cta->getActiveContext(), ins);
		return cta->getRegAsU32(0, 0) == 0xdeadbeef;
	}

	bool test_Bfind() {
		struct Case { PTXOperand::DataType type; PTXU64 value;
			bool shift; PTXU32 expected; };
		const Case cases[] = {
			{ PTXOperand::u32, 0, false, 0xffffffffu },
			{ PTXOperand::u32, 1, true, 31 },
			{ PTXOperand::u32, 0xffffffffu, false, 31 },
			{ PTXOperand::s32, 0xffffffffu, false, 0xffffffffu },
			{ PTXOperand::s32, 0xfffffffeu, true, 31 },
			{ PTXOperand::s32, 0x80000000u, false, 30 },
			{ PTXOperand::u64, 0, false, 0xffffffffu },
			{ PTXOperand::u64, 1, true, 63 },
			{ PTXOperand::u64, 0xffffffffffffffffULL, false, 63 },
			{ PTXOperand::s64, 0xffffffffffffffffULL, false, 0xffffffffu },
			{ PTXOperand::s64, 0xfffffffffffffffeULL, true, 63 },
			{ PTXOperand::s64, 0x8000000000000000ULL, false, 62 }
		};
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Bfind;
		ins.d = reg("d", PTXOperand::u32, 0);
		ins.a = reg("a", PTXOperand::u32, 1);
		for (const Case& test : cases) {
			ins.type = test.type;
			ins.a.type = test.type;
			ins.shiftAmount = test.shift;
			cta->reset();
			if (test.type == PTXOperand::u32 || test.type == PTXOperand::s32)
				cta->setRegAsU32(0, 1, static_cast<PTXU32>(test.value));
			else cta->setRegAsU64(0, 1, test.value);
			cta->eval_Bfind(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != test.expected) return false;
		}
		ins.type = PTXOperand::s32;
		ins.a.type = PTXOperand::s32;
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 3;
		cta->reset();
		cta->setRegAsU32(0, 0, 0xdeadbeefu);
		cta->setRegAsU32(0, 1, 0xffffffffu);
		cta->setRegAsPredicate(0, 3, false);
		cta->eval_Bfind(cta->getActiveContext(), ins);
		return cta->getRegAsU32(0, 0) == 0xdeadbeefu;
	}

	bool test_Popc() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Popc;
		ins.type = PTXOperand::b32;
		ins.a = reg("a", PTXOperand::b32, 1);
		ins.d = reg("d", PTXOperand::b32, 0);
		if (!ins.valid().empty()) return false;
		ins.d.type = PTXOperand::u32;
		if (!ins.valid().empty()) return false;
		ins.d.type = PTXOperand::b32;
		ins.modifier = PTXInstruction::rn;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::ftz;
		if (ins.valid().empty()) return false;
		ins.modifier = PTXInstruction::sat;
		if (ins.valid().empty()) return false;
		ins.modifier = 0;
		ins.carry = PTXInstruction::CC;
		if (ins.valid().empty()) return false;
		ins.carry = PTXInstruction::None;
		ins.type = PTXOperand::u32;
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::b32;
		ins.a.type = PTXOperand::b64;
		if (ins.valid().empty()) return false;
		ins.type = PTXOperand::b64;
		ins.a.type = PTXOperand::b64;
		ins.d.type = PTXOperand::b64;
		if (ins.valid().empty()) return false;
		ins.d.type = PTXOperand::b32;
		if (!ins.valid().empty()) return false;
		ins.type = PTXOperand::b32;
		ins.a.type = PTXOperand::b32;
		cta->reset();
		const PTXU32 values[] = {0, 0xffffffffU, 0x55, 0xaa, 1, 0x80000000U};
		const PTXU32 expected[] = {0, 32, 4, 4, 1, 1};
		for (unsigned i = 0; i < sizeof(values) / sizeof(*values); ++i) {
			cta->setRegAsU32(0, 1, values[i]);
			cta->eval_Popc(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != expected[i]) return false;
		}
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 3;
		cta->setRegAsPredicate(0, 3, false);
		cta->setRegAsU32(0, 0, 0xdeadbeef);
		cta->eval_Popc(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 0) != 0xdeadbeef) return false;
		ins.pg.condition = PTXOperand::PT;
		ins.type = PTXOperand::b64;
		ins.a.type = PTXOperand::b64;
		const PTXU64 values64[] = {0, 0xffffffffffffffffULL, 0x55, 0xaa, 1,
			0x8000000000000000ULL};
		const PTXU32 expected64[] = {0, 64, 4, 4, 1, 1};
		for (unsigned i = 0; i < sizeof(values64) / sizeof(*values64); ++i) {
			cta->setRegAsU64(0, 1, values64[i]);
			cta->eval_Popc(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != expected64[i]) return false;
		}
		ins.pg.condition = PTXOperand::Pred;
		cta->setRegAsPredicate(0, 3, false);
		cta->setRegAsU32(0, 0, 0xcafebabe);
		cta->eval_Popc(cta->getActiveContext(), ins);
		return cta->getRegAsU32(0, 0) == 0xcafebabe;
	}

	bool test_Lop3() {
		std::stringstream ptx;
		ptx << ".version 8.2\n"
			<< ".target sm_86\n"
			<< ".address_size 64\n"
			<< ".visible .entry test_lop3() {\n"
			<< "  .reg .b32 d, a, b, c;\n"
			<< "  .reg .pred p, q;\n"
			<< "  lop3.b32 d, a, b, c, 0x40;\n"
			<< "  lop3.or.b32 d|p, a, b, c, 0x3f, q;\n"
			<< "  lop3.and.b32 _|p, a, b, c, 0x3f, q;\n"
			<< "  ret;\n"
			<< "}\n";
		Module parsed;
		try {
			parsed.load(ptx);
		}
		catch (const hydrazine::Exception& error) {
			status << "failed to parse lop3 examples: " << error.what() << "\n";
			return false;
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Lop3;
		ins.type = PTXOperand::b32;
		ins.d = reg("d", PTXOperand::b32, 0);
		ins.a = reg("a", PTXOperand::b32, 1);
		ins.b = reg("b", PTXOperand::b32, 2);
		ins.c = reg("c", PTXOperand::b32, 3);
		ins.immLut = imm_uint("immLut", PTXOperand::b32, 0x40);
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU32(thread, 1, 0xffffffffu);
			cta->setRegAsU32(thread, 2, 0x0f0f0f0fu);
			cta->setRegAsU32(thread, 3, 0x00ff00ffu);
		}
		cta->eval_Lop3(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0x0f000f00u) return false;
		}

		ins.booleanOperator = PTXInstruction::BoolOr;
		ins.pq = reg("p", PTXOperand::pred, 4);
		ins.q = reg("q", PTXOperand::pred, 5);
		ins.immLut = imm_uint("immLut", PTXOperand::b32, 0x3f);
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU32(thread, 1, 0xffffffffu);
			cta->setRegAsU32(thread, 2, 0xffffffffu);
			cta->setRegAsPredicate(thread, 5, true);
		}
		cta->eval_Lop3(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU32(thread, 0) != 0
				|| !cta->getRegAsPredicate(thread, 4)) return false;
		}

		ins.booleanOperator = PTXInstruction::BoolAnd;
		ins.d.addressMode = PTXOperand::BitBucket;
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU32(thread, 1, 0);
			cta->setRegAsPredicate(thread, 5, false);
		}
		cta->eval_Lop3(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsPredicate(thread, 4)) return false;
		}

		return true;
	}

	// The lop3 predicate source `q` must be walked by assignRegisters like
	// every other register operand, or it keeps whatever register id the
	// parser left it with instead of the id its defining instruction got.
	bool test_Lop3RegisterAllocation() {
		std::stringstream ptx;
		ptx << ".version 8.2\n"
			<< ".target sm_86\n"
			<< ".address_size 64\n"
			<< ".visible .entry test_lop3_regalloc() {\n"
			<< "  .reg .b32 dd, aa, bb, cc;\n"
			<< "  .reg .pred pp, qq;\n"
			<< "  lop3.b32 dd, aa, bb, cc, 0x40;\n"
			<< "  setp.eq.b32 qq, aa, bb;\n"
			<< "  lop3.or.b32 dd|pp, aa, bb, cc, 0x3f, qq;\n"
			<< "  ret;\n"
			<< "}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse lop3 register allocation example: "
				<< error.what() << "\n";
			return false;
		}
		PTXKernel* kernel = parsed.kernels().begin()->second;
		PTXKernel::assignRegisters(*kernel->cfg());
		PTXOperand::RegisterType qqDef = 0, qqUse = 0;
		bool sawDef = false, sawUse = false;
		for (auto block = kernel->cfg()->begin(); block != kernel->cfg()->end(); ++block)
			for (auto instruction = block->instructions.begin();
				instruction != block->instructions.end(); ++instruction) {
				const PTXInstruction* instr =
					dynamic_cast<const PTXInstruction*>(*instruction);
				if (!instr) continue;
				if (instr->opcode == PTXInstruction::SetP) {
					qqDef = instr->d.reg;
					sawDef = true;
				} else if (instr->opcode == PTXInstruction::Lop3
					&& instr->q.addressMode == PTXOperand::Register) {
					qqUse = instr->q.reg;
					sawUse = true;
				}
			}
		if (!sawDef || !sawUse || qqDef != qqUse) {
			status << "lop3 q operand register id " << qqUse
				<< " does not match its definition's id " << qqDef << "\n";
			return false;
		}

		return true;
	}

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
						<< ", got " << cta->getRegAsB16(t, 0) << "\n";
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
						<< ", got " << cta->getRegAsB32(t, 0) << "\n";
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
						<< ", got " << cta->getRegAsB64(t, 0) << "\n";
				}
			}
		}

		// pred
		//
		if (result) {
			ins.type = PTXOperand::pred;
			ins.d = reg("p3", PTXOperand::pred, 0);
			ins.a = reg("p1", PTXOperand::pred, 1);
			ins.b = reg("p2", PTXOperand::pred, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsPredicate(t, 1, t & 1);
				cta->setRegAsPredicate(t, 2, t & 2);
			}
			cta->eval_And(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				const bool expected = (t & 1) && (t & 2);
				if (cta->getRegAsPredicate(t, 0) != expected) {
					result = false;
					status << "and.pred failed\n";
					break;
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
						<< ", got " << cta->getRegAsB16(t, 0) << "\n";
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
						<< ", got " << cta->getRegAsB32(t, 0) << "\n";
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
						<< ", got " << cta->getRegAsB64(t, 0) << "\n";
				}
			}
		}

		// pred
		//
		if (result) {
			ins.type = PTXOperand::pred;
			ins.d = reg("p3", PTXOperand::pred, 0);
			ins.a = reg("p1", PTXOperand::pred, 1);
			ins.b = reg("p2", PTXOperand::pred, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsPredicate(t, 1, t & 1);
				cta->setRegAsPredicate(t, 2, t & 2);
			}
			cta->eval_Or(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				const bool expected = (t & 1) || (t & 2);
				if (cta->getRegAsPredicate(t, 0) != expected) {
					result = false;
					status << "or.pred failed\n";
					break;
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
						<< ", got " << cta->getRegAsB16(t, 0) << "\n";
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
						<< ", got " << cta->getRegAsB32(t, 0) << "\n";
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

		// pred
		//
		if (result) {
			ins.type = PTXOperand::pred;
			ins.d = reg("p3", PTXOperand::pred, 0);
			ins.a = reg("p1", PTXOperand::pred, 1);
			ins.b = reg("p2", PTXOperand::pred, 2);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsPredicate(t, 1, t & 1);
				cta->setRegAsPredicate(t, 2, t & 2);
			}
			cta->eval_Xor(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				const bool a = t & 1;
				const bool b = t & 2;
				const bool expected = a != b;
				if (cta->getRegAsPredicate(t, 0) != expected) {
					result = false;
					status << "xor.pred failed\n";
					break;
				}
			}
		}
		return result;
	}

	/*!
		Tests several forms of the not instruction
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
					status << "not.b16 failed (thread " << t << "): expected " << expected
						<< ", got " << cta->getRegAsB16(t, 0) << "\n";
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
					status << "not.b32 failed (thread " << t << "): expected " << expected
						<< ", got " << cta->getRegAsB32(t, 0) << "\n";
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
					status << "not.b64 failed (thread " << t << "): expected " << expected
						<< ", got " << got << "\n";
				}
			}
		}

		// pred
		//
		if (result) {
			ins.type = PTXOperand::pred;
			ins.d = reg("p3", PTXOperand::pred, 0);
			ins.a = reg("p1", PTXOperand::pred, 1);
			for (int t = 0; t < threadCount; t++) {
				cta->setRegAsPredicate(t, 1, t & 1);
			}
			cta->eval_Not(cta->getActiveContext(), ins);
			for (int t = 0; t < threadCount; t++) {
				const bool expected = !(t & 1);
				if (cta->getRegAsPredicate(t, 0) != expected) {
					result = false;
					status << "not.pred failed\n";
					break;
				}
			}
		}
		return result;
	}

	bool test_Shl() {
		bool result = true;

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Shl;
		ins.type = PTXOperand::b32;
		ins.d = reg("r3", PTXOperand::b32, 0);
		ins.a = reg("r1", PTXOperand::b32, 1);
		ins.b = reg("r2", PTXOperand::u32, 2);
		if (!ins.valid().empty()) {
			status << "valid shl.b32 rejected: " << ins.valid() << "\n";
			return false;
		}

		PTXInstruction invalid = ins;
		invalid.d = reg("rd", PTXOperand::b64, 0);
		invalid.a = reg("ra", PTXOperand::b64, 1);
		if (invalid.valid().empty()) {
			status << "shl.b32 accepted 64-bit operands\n";
			return false;
		}
		invalid = ins;
		invalid.b = reg("rf", PTXOperand::f32, 2);
		if (invalid.valid().empty()) {
			status << "shl.b32 accepted an f32 shift operand\n";
			return false;
		}

		cta->reset();
		for (int t = 0; t < threadCount; t++) {
			cta->setRegAsB32(t, 1, 1);
			cta->setRegAsU32(t, 2, 31 + t % 3);
		}
		cta->eval_Shl(cta->getActiveContext(), ins);
		for (int t = 0; t < threadCount; t++) {
			const PTXU32 shift = 31 + t % 3;
			const PTXB32 expected = shift < 32 ? 1u << shift : 0;
			const PTXB32 got = cta->getRegAsB32(t, 0);
			if (got != expected) {
				result = false;
				status << "shl.b32 failed (thread " << t << "): expected "
					<< expected << ", got " << got << "\n";
				break;
			}
		}
		return result;
	}

	bool test_Shf() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Shf;
		ins.type = PTXOperand::b32;
		ins.d = reg("r4", PTXOperand::b32, 0);
		ins.a = reg("r1", PTXOperand::b32, 1);
		ins.b = reg("r2", PTXOperand::b32, 2);
		ins.c = reg("r3", PTXOperand::u32, 3);
		if (!ins.valid().empty()) {
			status << "valid shf rejected: " << ins.valid() << "\n";
			return false;
		}
		PTXInstruction invalid = ins;
		invalid.c = reg("f1", PTXOperand::f32, 3);
		if (invalid.valid().empty()) {
			status << "shf accepted an f32 shift operand\n";
			return false;
		}

		const PTXB32 a = 0x12345678u;
		const PTXB32 b = 0xabcdef01u;
		for (int mode = 0; mode < 2; ++mode) {
			ins.shiftMode = mode == 0 ? PTXInstruction::ShiftMode::Wrap
				: PTXInstruction::ShiftMode::Clamp;
			for (int direction = 0; direction < 2; ++direction) {
				ins.shiftDirection = direction == 0
					? PTXInstruction::ShiftLeft : PTXInstruction::ShiftRight;
				cta->reset();
				for (int t = 0; t < threadCount; ++t) {
					cta->setRegAsB32(t, 1, a);
					cta->setRegAsB32(t, 2, b);
					cta->setRegAsU32(t, 3, 31 + t % 4);
				}
				cta->eval_Shf(cta->getActiveContext(), ins);
				for (int t = 0; t < threadCount; ++t) {
					const PTXU32 c = 31 + t % 4;
					const PTXU32 n = mode == 0 ? c & 31 : std::min(c, 32u);
					PTXB32 expected;
					if (direction == 0) {
						expected = n == 0 ? b : n == 32 ? a
							: (b << n) | (a >> (32 - n));
					} else {
						expected = n == 0 ? a : n == 32 ? b
							: (b << (32 - n)) | (a >> n);
					}
					const PTXB32 got = cta->getRegAsB32(t, 0);
					if (got != expected) {
						status << "shf failed for count " << c << ": expected "
							<< expected << ", got " << got << "\n";
						return false;
					}
				}
			}
		}
		return true;
	}

	bool test_Shr() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Shr;
		ins.d = reg("r3", PTXOperand::u32, 0);
		ins.a = reg("r1", PTXOperand::u32, 1);
		ins.b = reg("r2", PTXOperand::u32, 2);
		if (!ins.valid().empty()) {
			status << "valid shr.u32 rejected: " << ins.valid() << "\n";
			return false;
		}

		PTXInstruction invalid = ins;
		invalid.d = reg("rd", PTXOperand::u64, 0);
		invalid.a = reg("ra", PTXOperand::u64, 1);
		if (invalid.valid().empty()) {
			status << "shr.u32 accepted 64-bit operands\n";
			return false;
		}
		invalid = ins;
		invalid.b = reg("rf", PTXOperand::f32, 2);
		if (invalid.valid().empty()) {
			status << "shr.u32 accepted an f32 shift operand\n";
			return false;
		}

		cta->reset();
		ins.type = PTXOperand::u32;
		for (int t = 0; t < threadCount; t++) {
			cta->setRegAsU32(t, 1, 0x80000000u);
			cta->setRegAsU32(t, 2, 31 + t % 3);
		}
		cta->eval_Shr(cta->getActiveContext(), ins);
		for (int t = 0; t < threadCount; t++) {
			const PTXU32 shift = 31 + t % 3;
			const PTXU32 expected = shift == 31 ? 1 : 0;
			const PTXU32 got = cta->getRegAsU32(t, 0);
			if (got != expected) {
				status << "shr.u32 failed (thread " << t << "): expected "
					<< expected << ", got " << got << "\n";
				return false;
			}
		}

		ins.type = PTXOperand::s32;
		ins.d.type = ins.a.type = PTXOperand::s32;
		for (int t = 0; t < threadCount; t++) {
			cta->setRegAsS32(t, 1, t & 1 ? INT_MIN : INT_MAX);
		}
		cta->eval_Shr(cta->getActiveContext(), ins);
		for (int t = 0; t < threadCount; t++) {
			const PTXS32 expected = t & 1 ? -1 : 0;
			const PTXS32 got = cta->getRegAsS32(t, 0);
			if (got != expected) {
				status << "shr.s32 failed (thread " << t << "): expected "
					<< expected << ", got " << got << "\n";
				return false;
			}
		}
		return true;
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
		std::stringstream cvtPtx;
		cvtPtx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_cvt() {\n"
			<< "  .reg .f16x2 d; .reg .f32 a, b;\n"
			<< "  cvt.rn.f16x2.f32 d, a, b; ret; }\n";
		Module parsed;
		try { parsed.load(cvtPtx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse .reg .f16x2 CVT: "
				<< error.what() << "\n";
			return false;
		}
		std::stringstream packedImmediatePtx;
		packedImmediatePtx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry packed_immediate() {\n"
			<< "  .reg .b32 d, td; .reg .b16 bd;\n"
			<< "  cvt.rn.f16x2.f32 d, 0f3f800000, 0f40000000;\n"
			<< "  cvt.rn.bf16x2.f32 d, 0f3f800000, 0f40000000;\n"
			<< "  cvt.rn.bf16.f32 bd, 0f3f800000;\n"
			<< "  cvt.rna.tf32.f32 td, 0f3f800000; ret; }\n";
		try {
			Module immediate;
			immediate.load(packedImmediatePtx);
			Module roundTrip;
			std::stringstream serialized(immediate.toString());
			roundTrip.load(serialized);
			unsigned int foundA = 0;
			unsigned int foundB = 0;
			for (Module::StatementVector::const_iterator i =
				roundTrip.statements().begin(); i != roundTrip.statements().end(); ++i) {
				if (i->directive == PTXStatement::Instr
					&& i->instruction.opcode == PTXInstruction::Cvt) {
					if (i->instruction.a.type == PTXOperand::f32) ++foundA;
					if (i->instruction.b.type == PTXOperand::f32) ++foundB;
				}
			}
			if (foundA != 4 || foundB != 2) {
				status << "packed cvt immediate lost its f32 source type\n";
				return false;
			}
		}
		catch (const std::exception& error) {
			status << "packed cvt immediate parse/round-trip failed: "
				<< error.what() << "\n";
			return false;
		}
		std::stringstream invalidBf16Register;
		invalidBf16Register << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry invalid_bf16() {\n"
			<< "  .reg .f32 d; .reg .b32 a;\n"
			<< "  cvt.f32.bf16 d, a; ret; }\n";
		try {
			Module invalid;
			invalid.load(invalidBf16Register);
			status << "cvt.f32.bf16 accepted a b32 register\n";
			return false;
		}
		catch (const std::exception&) {
		}
		std::stringstream validBf16Register;
		validBf16Register << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry valid_bf16() {\n"
			<< "  .reg .f32 d; .reg .b16 a;\n"
			<< "  cvt.f32.bf16 d, a; ret; }\n";
		try {
			Module valid;
			valid.load(validBf16Register);
		}
		catch (const std::exception& error) {
			status << "cvt.f32.bf16 rejected a b16 register: "
				<< error.what() << "\n";
			return false;
		}
		std::stringstream invalidScalarCvtPtx;
		invalidScalarCvtPtx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry invalid_cvt() {\n"
			<< "  .reg .u32 d, a, b;\n"
			<< "  cvt.u32.u32 d, a, b; ret; }\n";
		try {
			Module invalid;
			invalid.load(invalidScalarCvtPtx);
			status << "scalar CVT with an extra operand was accepted\n";
			return false;
		}
		catch (const std::exception&) {
		}

		PTXInstruction ins;
		ins.opcode = PTXInstruction::Cvt;
		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::s32, 1);
		if (!ins.valid().empty()) {
			status << "valid cvt.rn.ftz.f32.s32 rejected\n";
			return false;
		}
		ins.type = PTXOperand::f64;
		ins.d = reg("d", PTXOperand::f64, 0);
		if (ins.valid().empty()) {
			status << "invalid cvt.rn.ftz.f64.s32 accepted\n";
			return false;
		}
		ins.modifier = 0;
		ins.type = PTXOperand::b32;
		ins.d = reg("d", PTXOperand::b32, 0);
		ins.a = reg("a", PTXOperand::s32, 1);
		if (ins.valid().empty()) {
			status << "invalid cvt.b32.s32 accepted\n";
			return false;
		}
		ins.type = PTXOperand::s32;
		ins.d = reg("d", PTXOperand::s32, 0);
		ins.a = reg("a", PTXOperand::b32, 1);
		if (ins.valid().empty()) {
			status << "invalid cvt.s32.b32 accepted\n";
			return false;
		}
		ins.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::ftz;
		ins.d = reg("d", PTXOperand::f64, 0);
		ins.a = reg("a", PTXOperand::b32, 1);
		ins.a.relaxedType = PTXOperand::f32;
		if (!ins.valid().empty()) {
			status << "valid cvt.ftz.f64.f32 with b32 source rejected\n";
			return false;
		}
		ins.modifier = 0;
		ins.type = PTXOperand::f32;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::s32, 1);
		if (ins.valid().empty()) {
			status << "cvt.f32.s32 accepted without rounding\n";
			return false;
		}
		ins.type = PTXOperand::s32;
		ins.d = reg("d", PTXOperand::s32, 0);
		ins.a = reg("a", PTXOperand::f32, 1);
		if (ins.valid().empty()) {
			status << "cvt.s32.f32 accepted without rounding\n";
			return false;
		}
		ins.type = PTXOperand::f32;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::f64, 1);
		if (ins.valid().empty()) {
			status << "cvt.f32.f64 accepted without rounding\n";
			return false;
		}
		ins.type = PTXOperand::s64;
		ins.modifier = PTXInstruction::rn;
		ins.d = reg("d", PTXOperand::s64, 0);
		ins.a = reg("a", PTXOperand::s32, 1);
		if (ins.valid().empty()) {
			status << "cvt.rn.s64.s32 accepted\n";
			return false;
		}
		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rni;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::f32, 1);
		if (!ins.valid().empty()) {
			status << "valid cvt.rni.f32.f32 rejected\n";
			return false;
		}
		ins.type = PTXOperand::s64;
		ins.modifier = PTXInstruction::sat;
		ins.d = reg("d", PTXOperand::s64, 0);
		ins.a = reg("a", PTXOperand::s32, 1);
		if (ins.valid().empty()) {
			status << "cvt.sat.s64.s32 accepted although saturation is impossible\n";
			return false;
		}

		cta->reset();
		ins.type = PTXOperand::f64;
		ins.modifier = PTXInstruction::ftz;
		ins.d = reg("d", PTXOperand::f64, 0);
		ins.a = reg("a", PTXOperand::f32, 1);
		cta->setRegAsU32(0, 1, 0x80000001);
		cta->eval_Cvt(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0x8000000000000000ull) {
			status << "cvt.ftz.f64.f32 did not flush its f32 input\n";
			return false;
		}

		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn | PTXInstruction::ftz;
		ins.d = reg("d", PTXOperand::f32, 0);
		ins.a = reg("a", PTXOperand::f64, 1);
		cta->setRegAsF64(0, 1,
			-static_cast<PTXF64>(std::numeric_limits<PTXF32>::denorm_min()));
		cta->eval_Cvt(cta->getActiveContext(), ins);
		if (cta->getRegAsU32(0, 0) != 0x80000000) {
			status << "cvt.rn.ftz.f32.f64 did not flush its f32 result\n";
			return false;
		}

		// f64-to-f32 CVT clears stale upper bits in a wider destination.
		ins.type = PTXOperand::f32;
		ins.modifier = PTXInstruction::rn;
		ins.d = reg("d", PTXOperand::b64, 0);
		ins.a = reg("a", PTXOperand::f64, 1);
		cta->setRegAsU64(0, 0, ~0ull);
		cta->setRegAsF64(0, 1, 1.0);
		cta->eval_Cvt(cta->getActiveContext(), ins);
		if (cta->getRegAsU64(0, 0) != 0x000000003f800000ull) {
			status << "cvt.rn.f32.f64 did not zero-extend its result\n";
			return false;
		}

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
			cta->setRegAsU32(0, 1, 0x3f808001);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 0) != 0x3f80) {
				status << "cvt.rz.bf16.f32 failed\n";
				result = false;
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
			// Same-size floating conversions honor integer rounding modifiers.
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::f16, 1);
			cta->setRegAsU16(0, 1, 0x3f00); // 1.75
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 0) != 0x3c00) {
				status << "cvt.rzi.f16.f16 failed\n";
				result = false;
			}
		}

		if (result) {
			ins.modifier = PTXInstruction::rni;
			cta->setRegAsU16(0, 1, 0x4100); // 2.5, ties to even 2
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU16(0, 0) != 0x4000) {
				status << "cvt.rni.f16.f16 failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::f64;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::f64, 0);
			ins.a = reg("a", PTXOperand::f64, 1);
			cta->setRegAsF64(0, 1, 1.75);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsF64(0, 0) != 1.0) {
				status << "cvt.rzi.f64.f64 failed\n";
				result = false;
			}
		}

		if (result) {
			ins.modifier = PTXInstruction::rni;
			cta->setRegAsF64(0, 1, 2.5);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsF64(0, 0) != 2.0) {
				status << "cvt.rni.f64.f64 failed\n";
				result = false;
			}
		}

		if (result) {
			// .rni is nearest-even regardless of the host mode, then restores it.
			ins.type = PTXOperand::f32;
			ins.modifier = PTXInstruction::rni;
			ins.d = reg("d", PTXOperand::f32, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			const int previous = hydrazine::fegetround();
			hydrazine::fesetround(FE_UPWARD);
			cta->setRegAsF32(0, 1, 2.5f);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			const bool rounded = cta->getRegAsU32(0, 0) == 0x40000000u;
			const bool restored = hydrazine::fegetround() == FE_UPWARD;
			hydrazine::fesetround(previous);
			if (!rounded || !restored) {
				status << "cvt.rni did not force nearest-even or restore host mode\n";
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

		if (result) {
			// Integer saturation clamps both ends of the destination range.
			ins.type = PTXOperand::u8;
			ins.modifier = PTXInstruction::sat;
			ins.d = reg("d", PTXOperand::u8, 0);
			ins.a = reg("a", PTXOperand::s32, 1);
			cta->setRegAsS32(0, 1, 300);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU8(0, 0) != 255) {
				status << "cvt.sat.u8.s32 failed\n";
				result = false;
			}
		}

		if (result) {
			// Floating saturation clamps to [0, 1].
			ins.type = PTXOperand::f32;
			ins.modifier = PTXInstruction::rn | PTXInstruction::sat;
			ins.d = reg("d", PTXOperand::f32, 0);
			ins.a = reg("a", PTXOperand::s32, 1);
			cta->setRegAsS32(0, 1, -2);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != 0) {
				status << "cvt.rn.sat.f32.s32 failed\n";
				result = false;
			}
		}

		if (result) {
			// Unsigned and floating results zero-extend in wider registers.
			ins.type = PTXOperand::u32;
			ins.modifier = 0;
			ins.d = reg("d", PTXOperand::u64, 0);
			ins.a = reg("a", PTXOperand::s32, 1);
			cta->setRegAsS32(0, 1, -1);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU64(0, 0) != 0x00000000ffffffffull) {
				status << "cvt.u32.s32 did not zero-extend its result\n";
				result = false;
			}
		}

		if (result) {
			// Exact exclusive upper bounds must not reach an integer cast.
			ins.type = PTXOperand::s32;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::s32, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			cta->setRegAsF32(0, 1, std::ldexp(1.0f, 31));
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsS32(0, 0) != INT_MAX) {
				status << "cvt.rzi.s32.f32 upper boundary failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::u32;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::u32, 0);
			ins.a = reg("a", PTXOperand::f64, 1);
			cta->setRegAsF64(0, 1, std::ldexp(1.0, 32));
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != UINT_MAX) {
				status << "cvt.rzi.u32.f64 upper boundary failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::s64;
			ins.modifier = PTXInstruction::rzi;
			ins.d = reg("d", PTXOperand::s64, 0);
			ins.a = reg("a", PTXOperand::f64, 1);
			cta->setRegAsF64(0, 1, std::ldexp(1.0, 63));
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsS64(0, 0) != LLONG_MAX) {
				status << "cvt.rzi.s64.f64 upper boundary failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::f32;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("d", PTXOperand::b64, 0);
			ins.a = reg("a", PTXOperand::s32, 1);
			if (!ins.valid().empty()) {
				status << "cvt.rn.f32.s32 rejected a wider destination register\n";
				result = false;
			} else {
				cta->setRegAsU64(0, 0, ~0ull);
				cta->setRegAsS32(0, 1, 1);
				cta->eval_Cvt(cta->getActiveContext(), ins);
				if (cta->getRegAsU64(0, 0) != 0x000000003f800000ull) {
					status << "cvt.rn.f32.s32 did not zero-extend its result\n";
					result = false;
				}
			}
		}

		if (result) {
			// ReLU and packed conversions supported by sm_86.
			ins.type = PTXOperand::f16x2;
			ins.modifier = PTXInstruction::rn | PTXInstruction::relu;
			ins.d = reg("d", PTXOperand::b32, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = reg("b", PTXOperand::f32, 2);
			cta->setRegAsF32(0, 1, 1.0f);
			cta->setRegAsF32(0, 2, -2.0f);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != 0x3c000000) {
				status << "cvt.rn.relu.f16x2.f32 failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::bf16x2;
			ins.modifier = PTXInstruction::rn;
			cta->setRegAsF32(0, 1, 1.0f);
			cta->setRegAsF32(0, 2, 2.0f);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != 0x3f804000) {
				status << "cvt.rn.bf16x2.f32 failed\n";
				result = false;
			}
		}

		if (result) {
			ins.type = PTXOperand::tf32;
			ins.modifier = PTXInstruction::rna;
			ins.b = PTXOperand();
			cta->setRegAsU32(0, 1, 0x3f801000);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != 0x3f802000) {
				status << "cvt.rna.tf32.f32 failed\n";
				result = false;
			}
		}

		if (result) {
			// CVT sources are read before writing an aliased destination.
			ins.type = PTXOperand::u32;
			ins.modifier = 0;
			ins.d = reg("r0", PTXOperand::b64, 0);
			ins.a = reg("r0", PTXOperand::s32, 0);
			cta->setRegAsS32(0, 0, -1);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU64(0, 0) != 0xffffffffull) {
				status << "aliased scalar cvt lost its source\n";
				result = false;
			}
		}

		if (result) {
			// Both packed sources may alias the packed destination register.
			ins.type = PTXOperand::f16x2;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("r0", PTXOperand::b32, 0);
			ins.a = reg("r0", PTXOperand::f32, 0);
			ins.b = reg("r0", PTXOperand::f32, 0);
			cta->setRegAsF32(0, 0, 1.0f);
			cta->eval_Cvt(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 0) != 0x3c003c00u) {
				status << "aliased packed cvt lost its sources\n";
				result = false;
			}
		}

		if (result) {
			// ReLU maps NaN to the canonical FP16 NaN encoding.
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn | PTXInstruction::relu;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = PTXOperand();
			if (!ins.valid().empty()) {
				status << "valid scalar ReLU CVT rejected before evaluation\n";
				result = false;
			}
			cta->setRegAsU32(0, 1, 0xffc12345u);
			if (result) cta->eval_Cvt(cta->getActiveContext(), ins);
			if (result && cta->getRegAsU16(0, 0) != 0x7fffu) {
				status << "cvt.rn.relu.f16.f32 did not canonicalize NaN\n";
				result = false;
			}
		}

		if (result) {
			// Packed f16x2 accepts relaxed b32 sources and wider destinations.
			ins.type = PTXOperand::f16x2;
			ins.modifier = PTXInstruction::rn;
			ins.d = reg("d", PTXOperand::b64, 0);
			ins.a = reg("a", PTXOperand::b32, 1);
			ins.a.relaxedType = PTXOperand::f32;
			ins.b = reg("b", PTXOperand::b32, 2);
			ins.b.relaxedType = PTXOperand::f32;
			if (!ins.valid().empty()) {
				status << "packed cvt rejected relaxed sources or destination\n";
				result = false;
			}
		}

		if (result) {
			// Scalar CVT rejects invalid ReLU modifiers.
			ins.type = PTXOperand::f16;
			ins.modifier = PTXInstruction::rn | PTXInstruction::relu
				| PTXInstruction::sat;
			ins.d = reg("d", PTXOperand::b16, 0);
			ins.a = reg("a", PTXOperand::f32, 1);
			ins.b = PTXOperand();
			if (ins.valid().empty()) {
				status << "cvt.rn.sat.relu.f16.f32 accepted\n";
				result = false;
			}
			if (result) {
				ins.modifier = PTXInstruction::rn | PTXInstruction::relu
					| PTXInstruction::ftz;
				if (ins.valid().empty()) {
					status << "cvt.relu.ftz.f16.f32 accepted\n";
					result = false;
				}
			}
		}
	
		return result;
	}

	bool test_Cvta() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Cvta;
		ins.addressSpace = PTXInstruction::Param;
		ins.type = PTXOperand::u64;
		ins.d = reg("d", PTXOperand::u64, 1);
		ins.a = reg("a", PTXOperand::u64, 0);
		if (!ins.valid().empty()) return false;

		cta->reset();
		const PTXU64 expected = (PTXU64)kernel->ArgumentMemory + 1;
		for (int thread = 0; thread < threadCount; ++thread) {
			cta->setRegAsU64(thread, 0, 1);
		}
		cta->eval_Cvta(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU64(thread, 1) != expected) return false;
			cta->setRegAsU64(thread, 0, expected);
		}

		ins.toAddrSpace = true;
		cta->eval_Cvta(cta->getActiveContext(), ins);
		for (int thread = 0; thread < threadCount; ++thread) {
			if (cta->getRegAsU64(thread, 1) != 1) return false;
		}
		cta->reset();
		cta->functionCallStack.pushFrame(0, kernel->registerCount(), 16, 0, 0, 0, 0);
		ins.pg.condition = PTXOperand::PT;
		ins.addressSpace = PTXInstruction::Local;
		for (PTXOperand::DataType type : {PTXOperand::u32, PTXOperand::u64}) {
			ins.type = type;
			for (PTXOperand::RegisterType source : {0u, 1u}) {
				for (PTXU64 offset : {PTXU64(0), PTXU64(15)}) {
					ins.toAddrSpace = false;
					ins.a = reg("a", type, source);
					ins.d = reg("d", type, 2);
					for (int thread = 0; thread < 2; ++thread) {
						if (type == PTXOperand::u32)
							cta->setRegAsU32(thread, source, static_cast<PTXU32>(offset));
						else cta->setRegAsU64(thread, source, offset);
					}
					cta->eval_Cvta(cta->getActiveContext(), ins);
					for (int thread = 0; thread < 2; ++thread) {
						const PTXU64 expected = reinterpret_cast<PTXU64>(
							cta->functionCallStack.localMemoryPointer(thread)) + offset;
						if ((type == PTXOperand::u32
							? cta->getRegAsU32(thread, 2) : cta->getRegAsU64(thread, 2))
							!= (type == PTXOperand::u32 ? static_cast<PTXU32>(expected) : expected))
							return false;
					}
					ins.toAddrSpace = true;
					ins.a = reg("g", type, 2);
					ins.d = reg("o", type, 3);
					cta->eval_Cvta(cta->getActiveContext(), ins);
					for (int thread = 0; thread < 2; ++thread)
						if ((type == PTXOperand::u32
							? cta->getRegAsU32(thread, 3) : cta->getRegAsU64(thread, 3))
							!= (type == PTXOperand::u32 ? static_cast<PTXU32>(offset) : offset))
							return false;
				}
			}
		}
		ins.type = PTXOperand::u64;
		ins.toAddrSpace = false;
		ins.addressSpace = PTXInstruction::Local;
		ins.a = reg("a", PTXOperand::u64, 1);
		ins.d = reg("d", PTXOperand::u64, 2);
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 4;
		for (int thread = 0; thread < 2; ++thread) {
			cta->setRegAsPredicate(thread, 4, false);
			cta->setRegAsU64(thread, 2, 0xdeadbeefULL);
		}
		cta->eval_Cvta(cta->getActiveContext(), ins);
		for (int thread = 0; thread < 2; ++thread)
			if (cta->getRegAsU64(thread, 2) != 0xdeadbeefULL) return false;
		cta->functionCallStack.popFrame();
		return true;
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

		if (result) {
			// set.num.u32.f64 is false when either input is NaN.
			ins = PTXInstruction();
			ins.opcode = PTXInstruction::Set;
			ins.type = PTXOperand::u32;
			ins.d = reg("d", PTXOperand::u32, 3);
			ins.a = reg("a", PTXOperand::f64, 1);
			ins.b = reg("b", PTXOperand::f64, 2);
			ins.comparisonOperator = PTXInstruction::Num;

			cta->setRegAsF64(0, 1,
				std::numeric_limits<PTXF64>::quiet_NaN());
			cta->setRegAsF64(0, 2, 1.0);
			cta->eval_Set(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 3) != 0) {
				status << "[set.num.u32.f64 NaN test] failed\n";
				result = false;
			}

			ins.comparisonOperator = PTXInstruction::Ne;
			cta->eval_Set(cta->getActiveContext(), ins);
			if (cta->getRegAsU32(0, 3) != 0) {
				status << "[set.ne.u32.f64 NaN test] failed\n";
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

		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_setp_sink() {\n"
			<< "  .reg .pred p;\n  .reg .s32 a, b;\n"
			<< "  setp.eq.s32 _|p, a, b;\n"
			<< "  setp.ne.s32 p|_, a, b;\n  ret;\n}\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (const hydrazine::Exception& error) {
			status << "failed to parse setp sink forms: " << error.what() << "\n";
			return false;
		}

		PTXOperand sink;
		sink.identifier = "_";
		sink.type = PTXOperand::b64;
		sink.addressMode = PTXOperand::BitBucket;
		sink.reg = 0;
		ins.type = PTXOperand::s32;
		ins.a = reg("a", PTXOperand::s32, 1);
		ins.b = reg("b", PTXOperand::s32, 2);
		ins.comparisonOperator = PTXInstruction::Eq;
		cta->setRegAsS32(0, 1, 1);
		cta->setRegAsS32(0, 2, 2);
		cta->setRegAsPredicate(0, 0, true);

		ins.d = sink;
		ins.pq = reg("q", PTXOperand::pred, 3);
		cta->eval_SetP(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)
			|| !cta->getRegAsPredicate(0, 3)) return false;

		ins.d = reg("p", PTXOperand::pred, 3);
		ins.pq = sink;
		cta->setRegAsS32(0, 2, 1);
		cta->eval_SetP(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)
			|| !cta->getRegAsPredicate(0, 3)) return false;

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

		// .ftz is controlled by the f32 comparison type, not the result type.
		ins.type = PTXOperand::u64;
		ins.modifier = PTXInstruction::ftz;
		ins.d = reg("d", PTXOperand::u64, 0);
		ins.a = reg("a", PTXOperand::u64, 1);
		ins.b = reg("b", PTXOperand::u64, 2);
		ins.c = reg("c", PTXOperand::f32, 3);
		if (!ins.valid().empty()) {
			status << "slct.ftz.u64.f32 should be valid\n";
			result = false;
		}
		PTXInstruction invalid = ins;
		invalid.type = PTXOperand::f32;
		invalid.d = reg("d", PTXOperand::f32, 0);
		invalid.a = reg("a", PTXOperand::f32, 1);
		invalid.b = reg("b", PTXOperand::f32, 2);
		invalid.c = reg("c", PTXOperand::s32, 3);
		if (invalid.valid().empty()) {
			status << "slct.ftz.f32.s32 should be invalid\n";
			result = false;
		}

		if (result) {
			cta->setRegAsU64(0, 1, 11);
			cta->setRegAsU64(0, 2, 22);
			cta->setRegAsF32(0, 3,
				-std::numeric_limits<PTXF32>::denorm_min());
			cta->eval_SlCt(cta->getActiveContext(), ins);
			if (cta->getRegAsU64(0, 0) != 11) {
				status << "slct.ftz.u64.f32 did not select a\n";
				result = false;
			}
		}

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
		PTXInstruction ins;
		ins.opcode = PTXInstruction::TestP;
		ins.d = reg("p", PTXOperand::pred, 0);
		cta->reset();
		for (int typeIndex = 0; typeIndex < 2; ++typeIndex) {
			ins.type = typeIndex == 0 ? PTXOperand::f32 : PTXOperand::f64;
			ins.a = reg("a", ins.type, 1);
			if (ins.type == PTXOperand::f32) {
				cta->setRegAsF32(0, 1, 0.0f);
				cta->setRegAsF32(1, 1, -0.0f);
				cta->setRegAsF32(2, 1, std::numeric_limits<PTXF32>::denorm_min());
				cta->setRegAsF32(3, 1, 1.0f);
			} else {
				cta->setRegAsF64(0, 1, 0.0);
				cta->setRegAsF64(1, 1, -0.0);
				cta->setRegAsF64(2, 1, std::numeric_limits<PTXF64>::denorm_min());
				cta->setRegAsF64(3, 1, 1.0);
			}
			for (int mode = 0; mode < 2; ++mode) {
				const bool normal = mode == 0;
				ins.floatingPointMode = normal
					? PTXInstruction::Normal : PTXInstruction::SubNormal;
				cta->eval_TestP(cta->getActiveContext(), ins);
				if (cta->getRegAsPredicate(0, 0)
					|| cta->getRegAsPredicate(1, 0)
					|| cta->getRegAsPredicate(2, 0) != !normal
					|| cta->getRegAsPredicate(3, 0) != normal) {
					status << "testp normal/subnormal classification failed\n";
					return false;
				}
			}
		}
		return true;
	}

	bool test_Isspacep() {
		std::stringstream ptx;
		ptx << ".version 8.0\n.target sm_86\n.address_size 64\n"
			<< ".visible .entry test_isspacep() {\n"
			<< "  .reg .pred %p; .reg .u32 %r; .reg .u64 %rd;\n"
			<< "  isspacep.shared %p, %r;\n"
			<< "  isspacep.shared::cta %p, %rd; ret; }\n";
		Module parsed;
		try { parsed.load(ptx); }
		catch (...) {
			status << "isspacep.shared::cta failed to parse\n";
			return false;
		}
		PTXInstruction parsed32, parsed64;
		unsigned int parsedIsspacep = 0;
		for (Module::StatementVector::const_iterator i = parsed.statements().begin();
			i != parsed.statements().end(); ++i) {
			if (i->directive != PTXStatement::Instr
				|| i->instruction.opcode != PTXInstruction::Isspacep) continue;
			if (parsedIsspacep++ == 0) parsed32 = i->instruction;
			else parsed64 = i->instruction;
		}
		if (parsedIsspacep != 2 || parsed32.a.type != PTXOperand::u32
			|| parsed64.a.type != PTXOperand::u64
			|| parsed32.addressSpace != PTXInstruction::Shared
			|| parsed64.addressSpace != PTXInstruction::Shared) return false;
		std::stringstream serialized(parsed.toString());
		try { Module reparsed; reparsed.load(serialized); }
		catch (...) { return false; }
		if (parsed.toString().find("shared::cta") != std::string::npos) return false;
		for (const char* space : {"global::cta", "local::cta", "shared.cta",
			"shared::cluster"}) {
			std::stringstream invalid;
			invalid << ".version 8.0\n.target sm_86\n.address_size 64\n"
				<< ".visible .entry invalid() { .reg .pred %p; .reg .u64 %rd;\n"
				<< "isspacep." << space << " %p, %rd; ret; }\n";
			try { Module rejected; rejected.load(invalid); return false; }
			catch (...) {}
		}
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Isspacep;
		ins.addressSpace = PTXInstruction::Shared;
		ins.d = reg("p", PTXOperand::pred, 0);
		ins.a = reg("a", PTXOperand::u64, 1);
		if (!ins.valid().empty()) return false;
		ins.d.type = PTXOperand::u32;
		if (ins.valid().empty()) return false;
		ins.d.type = PTXOperand::pred;
		for (PTXOperand::DataType type : {PTXOperand::s32, PTXOperand::f32,
			PTXOperand::u16, PTXOperand::b64}) {
			ins.a.type = type;
			if (ins.valid().empty()) return false;
		}
		ins.a.type = PTXOperand::u64;
		ins.a.vec = PTXOperand::v2;
		if (ins.valid().empty()) return false;
		ins.a.vec = PTXOperand::v1;

		cta->reset();
		cta->functionCallStack.pushFrame(0, kernel->registerCount(), 16, 64,
			0, 0, 0);
		PTXU64 shared64, local64;
		hydrazine::bit_cast(shared64, cta->functionCallStack.sharedMemoryPointer());
		hydrazine::bit_cast(local64, cta->functionCallStack.localMemoryPointer(0));
		for (const PTXInstruction& qualified : {parsed32, parsed64}) {
			const PTXU64 base = shared64;
			const PTXU64 size = cta->functionCallStack.sharedMemorySize();
			for (PTXU64 offset : {PTXU64(0), size - 1, size}) {
				if (qualified.a.type == PTXOperand::u32)
					cta->setRegAsU32(0, qualified.a.reg, static_cast<PTXU32>(base + offset));
				else cta->setRegAsU64(0, qualified.a.reg, base + offset);
				cta->eval_Isspacep(cta->getActiveContext(), qualified);
				if (cta->getRegAsPredicate(0, qualified.d.reg) != (offset < size)) return false;
			}
		}
		ins.a.type = PTXOperand::u64;
		ins.addressSpace = PTXInstruction::Shared;
		cta->setRegAsU64(0, 1, shared64);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		cta->setRegAsU64(0, 1, local64);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Local;
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Global;
		cta->setRegAsU64(0, 1, shared64);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (cta->getRegAsPredicate(0, 0)) return false;
		cta->setRegAsU64(0, 1, 0x123456789ULL);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		PTXU32 shared32;
		hydrazine::bit_cast(shared32, cta->functionCallStack.sharedMemoryPointer());
		ins.a.type = PTXOperand::u32;
		ins.addressSpace = PTXInstruction::Shared;
		cta->setRegAsU32(0, 1, shared32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		PTXU32 local32;
		hydrazine::bit_cast(local32, cta->functionCallStack.localMemoryPointer(0));
		ins.addressSpace = PTXInstruction::Local;
		cta->setRegAsU32(0, 1, local32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Global;
		cta->setRegAsU32(0, 1, local32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (cta->getRegAsPredicate(0, 0)) return false;
		cta->setRegAsPredicate(0, 0, true);
		cta->setRegAsPredicate(0, 2, false);
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 2;
		ins.addressSpace = PTXInstruction::Shared;
		cta->setRegAsU32(0, 1, local32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		cta->setRegAsPredicate(0, 0, false);
		cta->setRegAsU32(0, 1, shared32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		bool predicatedOff = !cta->getRegAsPredicate(0, 0);

		*reinterpret_cast<PTXU32*>(local64) = 0x12345678;
		cta->functionCallStack.pushFrame(0, kernel->registerCount(), 16, 64,
			0, 0, 0);
		PTXInstruction load;
		load.opcode = PTXInstruction::Ld;
		load.addressSpace = PTXInstruction::Generic;
		load.type = PTXOperand::u32;
		load.d = reg("d", PTXOperand::u32, 3);
		load.a = reg("a", PTXOperand::u64, 1);
		load.a.addressMode = PTXOperand::Indirect;
		load.pg.condition = PTXOperand::Pred;
		load.pg.reg = 4;
		for (int thread = 0; thread < threadCount; ++thread)
			cta->setRegAsPredicate(thread, 4, thread == 0);
		cta->setRegAsU64(0, 1, local64);
		cta->eval_Ld(cta->getActiveContext(), load);
		bool callerLocal = cta->getRegAsU32(0, 3) == 0x12345678;
		ins.a.type = PTXOperand::u64;
		ins.pg.condition = PTXOperand::PT;
		ins.addressSpace = PTXInstruction::Local;
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		callerLocal &= cta->getRegAsPredicate(0, 0);
		ins.addressSpace = PTXInstruction::Global;
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		callerLocal &= !cta->getRegAsPredicate(0, 0);
		ins.a.type = PTXOperand::u32;
		cta->setRegAsU32(0, 1, static_cast<PTXU32>(local64));
		ins.addressSpace = PTXInstruction::Local;
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		callerLocal &= cta->getRegAsPredicate(0, 0);
		ins.addressSpace = PTXInstruction::Global;
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		callerLocal &= !cta->getRegAsPredicate(0, 0);
		cta->functionCallStack.popFrame();
		cta->functionCallStack.popFrame();
		return predicatedOff && callerLocal;
	}

	bool test_IsspacepConst() {
		ConstMemoryTestKernel constKernel(
			module.getKernel("_Z17k_simple_sequencePi"));
		constKernel.initialize();
		constKernel.setKernelShape(1, 1, 1);
		constKernel.setConstMemory(16);
		CooperativeThreadArray constCta(&constKernel, ir::Dim3(), false);
		constCta.functionCallStack.pushFrame(0, constKernel.registerCount(),
			16, 64, 0, 0, 0);
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Isspacep;
		ins.addressSpace = PTXInstruction::Const;
		ins.d = reg("p", PTXOperand::pred, 0);
		ins.a = reg("a", PTXOperand::u64, 1);
		if (!ins.valid().empty()) return false;
		PTXU64 base;
		hydrazine::bit_cast(base, constKernel.ConstMemory);
		constCta.setRegAsU64(0, 1, base + 15);
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		if (!constCta.getRegAsPredicate(0, 0)) return false;
		constCta.setRegAsU64(0, 1, base + 16);
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		if (constCta.getRegAsPredicate(0, 0)) return false;
		constCta.setRegAsU64(0, 1, base - 1);
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		if (constCta.getRegAsPredicate(0, 0)) return false;
		constCta.setRegAsPredicate(0, 2, false);
		constCta.setRegAsPredicate(0, 0, true);
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 2;
		constCta.setRegAsU64(0, 1, base + 16);
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		if (!constCta.getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Global;
		ins.pg.condition = PTXOperand::PT;
		const PTXU64 addresses[] = {base - 1, base, base + 15, base + 16};
		for (PTXOperand::DataType type : {PTXOperand::u64, PTXOperand::u32}) {
			ins.a.type = type;
			for (int i = 0; i < 4; ++i) {
				if (type == PTXOperand::u32)
					constCta.setRegAsU32(0, 1, static_cast<PTXU32>(addresses[i]));
				else constCta.setRegAsU64(0, 1, addresses[i]);
				constCta.eval_Isspacep(constCta.getActiveContext(), ins);
				if (constCta.getRegAsPredicate(0, 0) != (i == 0 || i == 3)) return false;
			}
		}
		constCta.setRegAsPredicate(0, 0, true);
		constCta.setRegAsPredicate(0, 2, false);
		ins.a.type = PTXOperand::u64;
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 2;
		constCta.setRegAsU64(0, 1, base);
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		if (!constCta.getRegAsPredicate(0, 0)) return false;
		constCta.setRegAsPredicate(0, 2, true);
		ins.addressSpace = PTXInstruction::Const;
		constCta.setRegAsU32(0, 1, static_cast<PTXU32>(base));
		ins.a.type = PTXOperand::u32;
		constCta.eval_Isspacep(constCta.getActiveContext(), ins);
		return constCta.getRegAsPredicate(0, 0);
	}

	bool test_IsspacepParam() {
		PTXInstruction ins;
		ins.opcode = PTXInstruction::Isspacep;
		ins.addressSpace = PTXInstruction::Param;
		ins.d = reg("p", PTXOperand::pred, 0);
		ins.a = reg("a", PTXOperand::u64, 1);
		if (!ins.valid().empty() || kernel->argumentMemorySize() == 0)
			return false;
		PTXU64 base = reinterpret_cast<PTXU64>(kernel->ArgumentMemory);
		PTXU64 size = kernel->argumentMemorySize();
		cta->reset();
		for (PTXU64 address : {base, base + size - 1, base - 1, base + size}) {
			cta->setRegAsU64(0, 1, address);
			cta->eval_Isspacep(cta->getActiveContext(), ins);
			if (cta->getRegAsPredicate(0, 0) !=
				(address >= base && address - base < size)) return false;
		}
		ins.addressSpace = PTXInstruction::Global;
		cta->setRegAsU64(0, 1, base);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Param;
		ins.a.type = PTXOperand::u32;
		const PTXU32 base32 = static_cast<PTXU32>(base);
		cta->setRegAsU32(0, 1, base32 + static_cast<PTXU32>(size - 1));
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Global;
		cta->setRegAsU32(0, 1, base32);
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		if (!cta->getRegAsPredicate(0, 0)) return false;
		ins.addressSpace = PTXInstruction::Param;
		cta->setRegAsPredicate(0, 0, true);
		cta->setRegAsPredicate(0, 2, false);
		ins.pg.condition = PTXOperand::Pred;
		ins.pg.reg = 2;
		cta->setRegAsU32(0, 1, base32 + static_cast<PTXU32>(size));
		cta->eval_Isspacep(cta->getActiveContext(), ins);
		return cta->getRegAsPredicate(0, 0);
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
			result = (result && test_Cvta());
			result = (result && test_Isspacep());
			result = (result && test_IsspacepConst());
			result = (result && test_IsspacepParam());
			result = (result && test_LdLu());
			result = (result && test_SuldSustCacheOperator());
			result = (result && test_StCacheOperator());
			result = (result && test_GrammarSharedActionFix());
			result = (result && test_PrefetchOpcode());
			result = (result && test_Red());
			result = (result && test_Fence());
			result = (result && test_AtomicRedSemantics());
			result = (result && test_LdStOrdering());
			result = (result && test_LdStMmio());
			result = (result && test_AtomicRedTypeTable());

			// arithmetic instructions
			result = (result && test_Abs());
			result = (result && test_Add());
			result = (result && test_Sub());
			result = (result && test_AddSubRounding());
			result = (result && test_Div());
			result = (result && test_Neg());
			result = (result && test_Rem());
			result = (result && test_Sad());
			result = (result && test_Min());
			result = (result && test_Max());
			if (prolix && result) {
				status << "pass: arithmetic instructions\n";
			}

			// difficult arithmetic instructions
			result = (result && test_Mad());
			result = (result && test_Mul());
			result = (result && test_MulRounding());
			result = (result && test_AddC());
			result = (result && test_SubC());
			result = (result && test_Dp());
			if (prolix && result) {
				status << "pass: exotic arithmetic instructions\n";
			}

			// floating-point instructions
			result = (result && test_Cos());
			result = (result && test_Sin());
			result = (result && test_Tanh());
			result = (result && test_CopySign());
			result = (result && test_Ex2());
			result = (result && test_Fma());
			result = (result && test_F16Fma());
			result = (result && test_Bf16Fma());
			result = (result && test_MmaWarpParticipation());
			result = (result && test_ShflSyncParser());
			result = (result && test_ShflSyncExecution());
			result = (result && test_ShflSyncSynchronization());
			result = (result && test_WarpRegisters());
			result = (result && test_BarWarpParser());
			result = (result && test_BarWarpSynchronization());
			result = (result && test_MatchParser());
			result = (result && test_MatchExecution());
			result = (result && test_MatchSynchronization());
			result = (result && test_ReduxParser());
			result = (result && test_ReduxExecution());
			result = (result && test_ReduxSynchronization());
			result = (result && test_VoteSyncParser());
			result = (result && test_VoteSyncDivergentArrival());
			result = (result && test_VoteSync());
			result = (result && test_VoteSyncDifferentMasks());
			result = (result && test_Activemask());
			result = (result && test_MovmatrixParser());
			result = (result && test_Movmatrix());
			result = (result && test_MovmatrixControls());
			result = (result && test_LdmatrixIR());
			result = (result && test_LdmatrixParser());
			result = (result && test_LdmatrixAddress());
			result = (result && test_Ldmatrix());
			result = (result && test_LdmatrixControls());
			result = (result && test_SparseMmaParser());
			result = (result && test_SparseHalfMmaParser());
			result = (result && test_SparseHalfMma());
			result = (result && test_SparseHalfMma(true));
			result = (result && test_SparseTf32Mma());
			result = (result && test_SparseTf32Mma(true));
			result = (result && test_SparseIntegerMmaParser());
			result = (result && test_SparseIntegerMmaParser(true));
			result = (result && test_SparseIntegerMma(false) && test_SparseIntegerMma(true));
			result = (result && test_SparseIntegerMma(false, true) && test_SparseIntegerMma(true, true));
			result = (result && test_SparseVectorFragments());
			result = (result && test_MmaNamedDestination());
			result = (result && test_MmaFloatShapes());
			result = (result && test_Mma());
			result = (result && test_MmaInt8());
			result = (result && test_MmaInt8WideShapes());
			result = (result && test_Lg2());
			result = (result && test_Sqrt());
			result = (result && test_Rsqrt());
			result = (result && test_Rcp());
			result = (result && test_TestP());
			if (prolix && result) {
				status << "pass: floating-point instructions\n";
			}

			// logical and shift instructions
			result = (result && test_Fns());
			result = (result && test_Szext());
			result = (result && test_Bfe());
			result = (result && test_Bfi());
			result = (result && test_Bfind());
			result = (result && test_Popc());
			result = (result && test_Bmsk());
			result = (result && test_Lop3());
			result = (result && test_Lop3RegisterAllocation());
			result = (result && test_And());
			result = (result && test_Or());
			result = (result && test_Xor());
			result = (result && test_Not());
			result = (result && test_Shf());
			result = (result && test_Shl());
			result = (result && test_Shr());
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
