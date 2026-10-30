/*! \file CooperativeThreadArray.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 5 February 2009
	\brief defines the state of each cooperative thread array
		with associated code for emulating its execution
*/

// Ocelot Includes
#include <ocelot/ir/PTXOperand.h>
#include <ocelot/ir/PTXInstruction.h>
#include <ocelot/ir/Module.h>

#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/EmulatedKernelScheduler.h>
#include <ocelot/executive/CTAContext.h>
#include <ocelot/executive/TextureOperations.h>

#include <ocelot/api/OcelotConfiguration.h>

// Hydrazine Includes
#include <hydrazine/Casts.h>
#include <hydrazine/FloatingPoint.h>
#include <hydrazine/debug.h>
#include <hydrazine/math.h>

// Standard Library Includes
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <climits>
#include <algorithm>
#include <limits>
#include <sstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

// global control for enabling reporting within the emulator
#define REPORT_BASE 0

// reporting for kernel instructions
#define REPORT_STATIC_INSTRUCTIONS 0
#define REPORT_DYNAMIC_INSTRUCTIONS 1

// reporting for register accesses
#define REPORT_NTH_THREAD_ONLY 1
#define NTH_THREAD 0
#define REPORT_REGISTER_READS 1
#define REPORT_REGISTER_WRITES 1
#define REPORT_PREDICATE_READS 0

// individually turn on or off reporting for particular instructions
#define REPORT_ABS        1
#define REPORT_ADD        1
#define REPORT_ADDC       1
#define REPORT_AND        1
#define REPORT_ATOM       1
#define REPORT_BAR        1
#define REPORT_BRA        1
#define REPORT_BRKPT      1
#define REPORT_CALL       1
#define REPORT_CNOT       1
#define REPORT_COS        1
#define REPORT_CVT        1
#define REPORT_DIV        1
#define REPORT_EX2        1
#define REPORT_EXIT       1
#define REPORT_LD         0
#define REPORT_LG2        1
#define REPORT_MAD24      1
#define REPORT_MAD        1
#define REPORT_MAX        1
#define REPORT_MIN        1
#define REPORT_MOV        1
#define REPORT_MUL24      1
#define REPORT_MUL        1
#define REPORT_NEG        1
#define REPORT_NOT        1
#define REPORT_OR         1
#define REPORT_RCP        1
#define REPORT_RECONVERGE 0
#define REPORT_RED        1
#define REPORT_REM        1
#define REPORT_RET        1
#define REPORT_RSQRT      1
#define REPORT_SAD        1
#define REPORT_SELP       0
#define REPORT_SETP       0
#define REPORT_SET        1
#define REPORT_SHL        1
#define REPORT_SHR        1
#define REPORT_SIN        1
#define REPORT_SLCT       1
#define REPORT_SQRT       1
#define REPORT_ST         0
#define REPORT_SUB        1
#define REPORT_SUBC       1
#define REPORT_TEX        1
#define REPORT_TRAP       1
#define REPORT_VOTE       1
#define REPORT_XOR        1

template <typename T>
static T CTAAbs(T a) {
	if (a < 0) {
		return -a;
	}
	return a;
}

template<typename T>
static bool addressInRange(T address, T base, T size) {
	return address >= base && address - base < size;
}

template <typename T>
static T CTARemainder(T a, T b) {
	if (b == 0) {
		report("warning: rem by zero is unspecified by PTX; Ocelot returns 0");
		return 0;
	}
	return a % b;
}

template<typename T>
bool issubnormal_(T r0)
{
	return !hydrazine::isnormal(r0) && !hydrazine::isnan(r0)
		&& !hydrazine::isinf(r0) && r0 != (T)0;
}

static int setRoundingMode(int modifier)
{
	const int previous = hydrazine::fegetround();
	int rounding = FE_TONEAREST;
	if (modifier & ir::PTXInstruction::rz) rounding = FE_TOWARDZERO;
	else if (modifier & ir::PTXInstruction::rm) rounding = FE_DOWNWARD;
	else if (modifier & ir::PTXInstruction::rp) rounding = FE_UPWARD;
	hydrazine::fesetround(rounding);
	return previous;
}

template<typename T>
static T roundedAdd(T a, T b, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = a + b;
	hydrazine::fesetround(previous);
	return d;
}

template<typename T>
static T roundedSub(T a, T b, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = a - b;
	hydrazine::fesetround(previous);
	return d;
}

template<typename T>
static T roundedMul(T a, T b, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = a * b;
	hydrazine::fesetround(previous);
	return d;
}

template<typename T>
static T roundedDiv(T a, T b, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = a / b;
	hydrazine::fesetround(previous);
	return d;
}

template<typename T>
static T roundedSqrt(T a, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = std::sqrt(a);
	hydrazine::fesetround(previous);
	return d;
}

template<typename T>
static T roundedFma(T a, T b, T c, int modifier)
{
	const int previous = setRoundingMode(modifier);
	T d = std::fma(a, b, c);
	hydrazine::fesetround(previous);
	return d;
}

// Truncate a * b + c to f32 and force the last bit to 1 if any bits were lost
// (round-to-odd), so the later RNE to bf16 cannot see a false tie.
static ir::PTXF32 fmaF32OneRound(ir::PTXF32 a, ir::PTXF32 b, ir::PTXF32 c)
{
	std::feclearexcept(FE_INEXACT);
	const int previous = setRoundingMode(ir::PTXInstruction::rz);
	volatile ir::PTXF32 d = static_cast<ir::PTXF32>(
		static_cast<double>(a) * b + c); // bf16 product is exact in double
	const ir::PTXU32 lost = std::fetestexcept(FE_INEXACT) ? 1U : 0U;
	hydrazine::fesetround(previous);
	return hydrazine::bit_cast<ir::PTXF32>(
		hydrazine::bit_cast<ir::PTXU32>(static_cast<ir::PTXF32>(d)) | lost);
}

static executive::ReconvergenceMechanism*
	getReconvergenceMechanism(executive::CooperativeThreadArray* cta) {

	typedef api::OcelotConfiguration config;
	auto reconvergenceMechanism = config::get().executive.reconvergenceMechanism;

	if (reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_IPDOM) {
		return new executive::ReconvergenceIPDOM(cta);
	}
	else if (reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_Barrier) {
		return new executive::ReconvergenceBarrier(cta);
	}
	else if (reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFGen6) {
		return new executive::ReconvergenceTFGen6(cta);
	}
	else if (reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFSortedStack) {
		return new executive::ReconvergenceTFSortedStack(cta);
	}
	else if (reconvergenceMechanism
		== executive::ReconvergenceMechanism::Reconverge_TFSoftware) {
		return new executive::ReconvergenceTFSoftware(cta);
	}
	else {
		assertM(false, "unknown thread reconvergence mechanism - "
			<< config::get().executive.reconvergenceMechanism);
	}

	return 0;
}

#define max(a, b) ((a) < (b) ? (b) : (a))
#define min(a, b) ((a) > (b) ? (b) : (a))

//////////////////////////////////////////////////////////////////////////////////////////////////
/*!
	Constructs a cooperative thread array from an EmulatedKernel instance

	\param kernel pointer to EmulatedKernel to which this CTA belongs
*/
executive::CooperativeThreadArray::CooperativeThreadArray(
	EmulatedKernel *k, const ir::Dim3& grid, bool trace):
	blockDim(k->blockDim()),
	gridDim(grid),
	threadCount(blockDim.x*blockDim.y*blockDim.z),
	kernel(k),
	functionCallStack(blockDim.x*blockDim.y*blockDim.z, k->argumentMemorySize(),
		k->parameterMemorySize(), k->registerCount(), k->localMemorySize(),
		k->globalLocalMemorySize(), k->totalSharedMemorySize()),
	clock(0),
	traceEvents(trace) {

	reconvergenceMechanism = getReconvergenceMechanism(this);

	initialize(blockDim);
}

executive::CooperativeThreadArray::CooperativeThreadArray(
	EmulatedKernel *k, const ir::Dim3& grid, const ir::Dim3& cta,
	unsigned int argumentMemorySize, unsigned int parameterMemorySize,
	unsigned int registerCount, unsigned int localMemorySize,
	unsigned int globalLocalMemorySize, unsigned int totalSharedMemorySize,
	bool trace):
	blockDim(cta),
	gridDim(grid),
	threadCount(blockDim.x*blockDim.y*blockDim.z),
	kernel(k),
	functionCallStack(blockDim.x*blockDim.y*blockDim.z, argumentMemorySize,
		parameterMemorySize, registerCount, localMemorySize,
		globalLocalMemorySize, totalSharedMemorySize),
	clock(0),
	traceEvents(trace) {

	reconvergenceMechanism = getReconvergenceMechanism(this);

	initialize(blockDim);
}

executive::CooperativeThreadArray::CooperativeThreadArray() : kernel(0),
	reconvergenceMechanism(0) {
}

/*!
	Destroys state associated with CTA
*/
executive::CooperativeThreadArray::~CooperativeThreadArray() {
	delete reconvergenceMechanism;
}

/*!
	Gets current instruction
*/
const ir::PTXInstruction&
	executive::CooperativeThreadArray::currentInstruction(
	CTAContext & context) {
	return kernel->instructions[context.PC];
}

ir::PTXU32 executive::CooperativeThreadArray::getSpecialValue(
	const int threadId, const ir::PTXOperand::SpecialRegister reg,
	const ir::PTXOperand::VectorIndex index ) const
{
	assert( reg != ir::PTXOperand::SpecialRegister_invalid );
	assert( reg != ir::PTXOperand::pm0 );
	assert( reg != ir::PTXOperand::pm1 );
	assert( reg != ir::PTXOperand::pm2 );
	assert( reg != ir::PTXOperand::pm3 );
	assert( reg != ir::PTXOperand::smId );
	assert( reg != ir::PTXOperand::nsmId );
	assert( reg != ir::PTXOperand::gridId );

	// PTX warp identity and lane masks describe 32 physical lane positions.
	const int lane = threadId % 32;
	const ir::PTXU32 laneBit = ir::PTXU32(1) << lane;

	switch( reg ) {
		case ir::PTXOperand::laneId: return lane;
		case ir::PTXOperand::lanemask_eq: return laneBit;
		case ir::PTXOperand::lanemask_lt: return laneBit - 1;
		case ir::PTXOperand::lanemask_le: return (laneBit - 1) | laneBit;
		case ir::PTXOperand::lanemask_gt: return ~((laneBit - 1) | laneBit);
		case ir::PTXOperand::lanemask_ge: return ~(laneBit - 1);
		case ir::PTXOperand::tid: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return ( threadId % blockDim.x );
				}
				case ir::PTXOperand::iy: {
					return ( ( threadId / blockDim.x ) % blockDim.y );
				}
				case ir::PTXOperand::iz: {
					return ( ( threadId / ( blockDim.x * blockDim.y ) ) );
				}
				case ir::PTXOperand::iw: assertM(false, "%tid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%tid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::ntid: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return blockDim.x;
				}
				case ir::PTXOperand::iy: {
					return blockDim.y;
				}
				case ir::PTXOperand::iz: {
					return blockDim.z;
				}
				case ir::PTXOperand::iw: assertM(false, "%ntid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%ntid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::warpId: return 0;
		case ir::PTXOperand::warpSize: return 32;
		case ir::PTXOperand::ctaId: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return blockId.x;
				}
				case ir::PTXOperand::iy: {
					return blockId.y;
				}
				case ir::PTXOperand::iz: {
					return blockId.z;
				}
				case ir::PTXOperand::iw: assertM(false, "%ctaid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false, "%ctaid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::nctaId: {
			switch( index ) {
				case ir::PTXOperand::ix: {
					return gridDim.x;
				}
				case ir::PTXOperand::iy: {
					return gridDim.y;
				}
				case ir::PTXOperand::iz: {
					return gridDim.z;
				}
				case ir::PTXOperand::iw: assertM(false,"%nctaid.w is invalid.");
				case ir::PTXOperand::iAll: assertM(false,"%nctaid is invalid.");
			}
			break;
		}
		case ir::PTXOperand::clock: return clock; break;
		default: break;
	}

	return 0;
}

ir::PTXF32 executive::CooperativeThreadArray::sat(int modifier, ir::PTXF32 f) {
	if (modifier & ir::PTXInstruction::sat) {
		return (f <= 0 || hydrazine::isnan(f) ? 0 : (f >= 1.0f ? 1.0f : f));
	}
	return f;
}

ir::PTXF64 executive::CooperativeThreadArray::sat(int modifier, ir::PTXF64 f) {
	if (modifier & ir::PTXInstruction::sat) {
		return (f <= 0 || hydrazine::isnan(f) ? 0 : (f >= 1.0 ? 1.0 : f));
	}
	return f;
}

static ir::PTXF32 ftz(int modifier, ir::PTXF32 f) {
	if (modifier & ir::PTXInstruction::ftz) {
		return (!hydrazine::isnormal(f) &&
			!hydrazine::isnan(f) && !hydrazine::isinf(f))
			? hydrazine::copysign(0.0f, f) : f;
	}
	return f;
}

static ir::PTXF32 minMaxF32(int modifier, ir::PTXF32 a,
	ir::PTXF32 b, bool maximum) {
	const bool xorSign = std::signbit(a) != std::signbit(b);
	if (modifier & ir::PTXInstruction::xorsign) {
		a = std::fabs(a);
		b = std::fabs(b);
	}
	ir::PTXF32 d;
	if ((modifier & ir::PTXInstruction::nan)
		&& (hydrazine::isnan(a) || hydrazine::isnan(b))) {
		d = hydrazine::bit_cast<ir::PTXF32>(0x7fffffffU);
	} else if (hydrazine::isnan(a)) {
		d = b;
	} else if (hydrazine::isnan(b)) {
		d = a;
	} else {
		d = maximum ? (a > b ? a : b) : (a < b ? a : b);
	}
	d = ftz(modifier, d);
	return (modifier & ir::PTXInstruction::xorsign) && !hydrazine::isnan(d)
		? hydrazine::copysign(d, xorSign ? -1.0f : 1.0f) : d;
}

static ir::PTXU16 ftzF16(int modifier, ir::PTXU16 bits)
{
	const bool subnormal = (bits & 0x7c00u) == 0 && (bits & 0x03ffu) != 0;
	if ((modifier & ir::PTXInstruction::ftz) && subnormal) {
		return bits & 0x8000u; // preserve sign: +0 or -0
	}
	return bits;
}

static ir::PTXF32 f16ToF32(ir::PTXU16 bits) {
	ir::PTXF16 half;
	std::memcpy(&half, &bits, sizeof(half));
	return static_cast<ir::PTXF32>(half);
}

static ir::PTXF32 bf16ToF32(ir::PTXU16 bits) {
	return hydrazine::bit_cast<ir::PTXF32>(static_cast<ir::PTXU32>(bits) << 16);
}

// PTX half formats use sign/magnitude masks; CUDA defines 0x7fff as canonical NaN.
static const ir::PTXU16 halfSignMask = 0x8000u;
static const ir::PTXU16 halfMagnitudeMask = 0x7fffu;
static const ir::PTXU16 halfCanonicalNan = 0x7fffu;
static const ir::PTXU16 f16ExponentMask = 0x7c00u;
static const ir::PTXU16 f16MantissaMask = 0x03ffu;
static const ir::PTXU16 bf16ExponentMask = 0x7f80u;
static const ir::PTXU16 bf16MantissaMask = 0x007fu;

static bool isNanHalf(ir::PTXU16 value, ir::PTXOperand::DataType type) {
	const bool bfloat = type == ir::PTXOperand::bf16 || type == ir::PTXOperand::bf16x2;
	const ir::PTXU16 exponentMask = bfloat ? bf16ExponentMask : f16ExponentMask;
	const ir::PTXU16 mantissaMask = bfloat ? bf16MantissaMask : f16MantissaMask;
	return (value & exponentMask) == exponentMask && (value & mantissaMask) != 0;
}

static ir::PTXU16 minMaxHalf(ir::PTXOperand::DataType type, int modifier,
	ir::PTXU16 a, ir::PTXU16 b, bool maximum) {
	const bool isF16 = type == ir::PTXOperand::f16 || type == ir::PTXOperand::f16x2;
	if (isF16) {
		a = ftzF16(modifier, a);
		b = ftzF16(modifier, b);
	}
	ir::PTXU16 xorSign = 0;
	if (modifier & ir::PTXInstruction::xorsign) {
		xorSign = (a ^ b) & halfSignMask;
		a &= halfMagnitudeMask;
		b &= halfMagnitudeMask;
	}
	const bool aNan = isNanHalf(a, type);
	const bool bNan = isNanHalf(b, type);
	const bool bothZero = ((a | b) & halfMagnitudeMask) == 0;
	if (aNan && bNan) {
		return halfCanonicalNan;
	}
	if ((modifier & ir::PTXInstruction::nan) && (aNan || bNan)) {
		return halfCanonicalNan;
	}
	ir::PTXU16 d;
	if (aNan) {
		d = b;
	}
	else if (bNan) {
		d = a;
	}
	else if (bothZero) {
		d = maximum ? (a & b) : (a | b);
	}
	else {
		const ir::PTXF32 aValue = isF16 ? f16ToF32(a) : bf16ToF32(a);
		const ir::PTXF32 bValue = isF16 ? f16ToF32(b) : bf16ToF32(b);
		d = maximum ? (aValue > bValue ? a : b) : (aValue < bValue ? a : b);
	}
	if (modifier & ir::PTXInstruction::xorsign) {
		return (d & halfMagnitudeMask) | xorSign;
	}
	return d;
}

static ir::PTXF32 tf32FromF32(ir::PTXF32 value) {
	ir::PTXU32 bits = hydrazine::bit_cast<ir::PTXU32>(value);
	if ((bits & 0x7f800000u) == 0x7f800000u) return value;
	const ir::PTXU32 discarded = bits & 0x1fffu;
	bits &= ~0x1fffu;
	if (discarded > 0x1000u ||
		(discarded == 0x1000u && (bits & 0x2000u))) {
		bits += 0x2000u;
	}
	return hydrazine::bit_cast<ir::PTXF32>(bits);
}

static ir::PTXU32 tf32FromF32Rna(ir::PTXF32 value) {
	// Model TF32 by retaining the upper 10 f32 fraction bits.
	const ir::PTXU32 exponentMask = 0x7f800000u;
	const ir::PTXU32 discardedMask = (1u << 13) - 1;
	const ir::PTXU32 halfway = 1u << 12;
	const ir::PTXU32 retainedUnit = 1u << 13;
	ir::PTXU32 bits = hydrazine::bit_cast<ir::PTXU32>(value);
	if ((bits & exponentMask) == exponentMask) return bits;
	const ir::PTXU32 discarded = bits & discardedMask;
	bits &= ~discardedMask;
	if (discarded >= halfway) bits += retainedUnit;
	return bits;
}

template< typename Source >
static ir::PTXU16 toF16(Source value, int modifier);

static ir::PTXU16 mmaHalfBits(executive::CooperativeThreadArray& cta,
	int threadID, const ir::PTXOperand& operand, unsigned int half)
{
	if (operand.bytes() == 4) {
		return static_cast<ir::PTXU16>(
			(cta.operandAsB32(threadID, operand) >> (half * 16)) & 0xffffu);
	}
	return cta.operandAsU16(threadID, operand);
}

static ir::PTXF32 mmaHalf(executive::CooperativeThreadArray& cta,
	int threadID, const ir::PTXOperand& operand, unsigned int half,
	ir::PTXOperand::DataType type)
{
	ir::PTXU16 bits = mmaHalfBits(cta, threadID, operand, half);
	return type == ir::PTXOperand::bf16 ? bf16ToF32(bits) : f16ToF32(bits);
}

void executive::CooperativeThreadArray::trace() {
	if (traceEvents) {
		currentEvent.contextStackSize =
			(ir::PTXU32)reconvergenceMechanism->stackSize();
		kernel->traceEvent(currentEvent);
	}
}

void executive::CooperativeThreadArray::postTrace() {
	if (traceEvents) {
		currentEvent.contextStackSize =
			(ir::PTXU32)reconvergenceMechanism->stackSize();

		kernel->tracePostEvent(currentEvent);
	}
}

////////////////////////////////////////////////////////////////////////////////

void executive::CooperativeThreadArray::reset() {

	reconvergenceMechanism->initialize();

	barriers.clear();
	barriers.resize(16);	// ir::PTX2.1

	size_t threads = getActiveContext().active.size();
	for (BarrierVector::iterator bar_it = barriers.begin();
		bar_it != barriers.end(); ++bar_it) {
		bar_it->initialize(threads);
	}
}

/*! initializes elements of the CTA */
void executive::CooperativeThreadArray::initialize(const ir::Dim3 & block) {

	counter = 0;
	blockId = block;

	currentEvent.blockId  = blockId;
	currentEvent.gridDim  = gridDim;
	currentEvent.blockDim = blockDim;

	reset();
}

/*! finishes execution of the CTA */
void executive::CooperativeThreadArray::finalize() {

}

/*!
	Called by the worker thread to evaluate a block
*/
void executive::CooperativeThreadArray::execute(int PC) {
	using namespace ir;

	jumpToPC(PC);

	bool running = true;
	assert(reconvergenceMechanism->stackSize());

	report("CooperativeThreadArray::execute called");
	report("  block is " << blockId.x << ", "
		<< blockId.y << ", " << blockId.z);
	reportE(REPORT_STATIC_INSTRUCTIONS, "Running " << kernel->toString());

	do {
		assert(reconvergenceMechanism->stackSize());

		// get the context and advance the program counter
		CTAContext& context = getActiveContext();
		const ir::PTXInstruction& instr  = currentInstruction(context);
		const ir::PTXInstruction::Opcode opcode = instr.opcode;

		reconvergenceMechanism->evalPredicate(context);

		reportE(REPORT_DYNAMIC_INSTRUCTIONS, " [PC: " << context.PC
			<< ", counter: " << counter
			<< "] " << instr.toString() << " [stack "
			<< reconvergenceMechanism->stackSize() << "] [active "
			<< context.active.count() << "]" );

		if (traceEvents) {
			currentEvent.reset();
			currentEvent.PC = context.PC;
			currentEvent.instruction = &instr;
			currentEvent.active = context.predicateMask(instr);
		}

		switch (instr.opcode) {
			case ir::PTXInstruction::Abs:
				eval_Abs(context, instr); break;
			case ir::PTXInstruction::Add:
				eval_Add(context, instr); break;
			case ir::PTXInstruction::AddC:
				eval_AddC(context, instr); break;
			case ir::PTXInstruction::And:
				eval_And(context, instr); break;
			case ir::PTXInstruction::Atom:
				eval_Atom(context, instr); break;
			case ir::PTXInstruction::Bar:
				eval_Bar(context, instr); break;
			case ir::PTXInstruction::BarWarp:
				eval_WarpSync(context, instr); break;
			case ir::PTXInstruction::Bfi:
				eval_Bfi(context, instr); break;
			case ir::PTXInstruction::Bfind:
				eval_Bfind(context, instr); break;
			case ir::PTXInstruction::Bmsk:
				eval_Bmsk(context, instr); break;
			case ir::PTXInstruction::Bfe:
				eval_Bfe(context, instr); break;
			case ir::PTXInstruction::Bra:
				eval_Bra(context, instr); break;
			case ir::PTXInstruction::Brev:
				eval_Brev(context, instr); break;
			case ir::PTXInstruction::Brkpt:
				eval_Brkpt(context, instr); break;
			case ir::PTXInstruction::Call:
				eval_Call(context, instr); break;
			case ir::PTXInstruction::Clz:
				eval_Clz(context, instr); break;
			case ir::PTXInstruction::CNot:
				eval_CNot(context, instr); break;
			case ir::PTXInstruction::CopySign:
				eval_CopySign(context, instr); break;
			case ir::PTXInstruction::Cos:
				eval_Cos(context, instr); break;
			case ir::PTXInstruction::Cvt:
				eval_Cvt(context, instr); break;
			case ir::PTXInstruction::Cvta:
				eval_Cvta(context, instr); break;
			case ir::PTXInstruction::Div:
				eval_Div(context, instr); break;
			case ir::PTXInstruction::Dp2a:
			case ir::PTXInstruction::Dp4a:
				eval_Dp(context, instr); break;
			case ir::PTXInstruction::Ex2:
				eval_Ex2(context, instr); break;
			case ir::PTXInstruction::Exit:
				eval_Exit(context, instr); break;
			case ir::PTXInstruction::Fma:
				eval_Fma(context, instr); break;
			case ir::PTXInstruction::Fns:
				eval_Fns(context, instr); break;
			case ir::PTXInstruction::Mma:
				eval_Mma(context, instr); break;
			case ir::PTXInstruction::Isspacep:
				eval_Isspacep(context, instr); break;
			case ir::PTXInstruction::Ld:
				eval_Ld(context, instr); break;
			case ir::PTXInstruction::Lg2:
				eval_Lg2(context, instr); break;
			case ir::PTXInstruction::Lop3:
				eval_Lop3(context, instr); break;
			case ir::PTXInstruction::Ldmatrix:
				eval_Ldmatrix(context, instr); break;
			case ir::PTXInstruction::Ldu:
				eval_Ldu(context, instr); break;
			case ir::PTXInstruction::Mad24:
				eval_Mad24(context, instr); break;
			case ir::PTXInstruction::Mad:
				eval_Mad(context, instr); break;
			case ir::PTXInstruction::Max:
				eval_Max(context, instr); break;
			case ir::PTXInstruction::Membar:
				eval_Membar(context, instr); break;
			case ir::PTXInstruction::Fence:
				eval_Fence(context, instr); break;
			case ir::PTXInstruction::Min:
				eval_Min(context, instr); break;
			case ir::PTXInstruction::Activemask:
				eval_Activemask(context, instr); break;
			case ir::PTXInstruction::Movmatrix:
				eval_Movmatrix(context, instr); break;
			case ir::PTXInstruction::Mov:
				eval_Mov(context, instr); break;
			case ir::PTXInstruction::Mul24:
				eval_Mul24(context, instr); break;
			case ir::PTXInstruction::Mul:
				eval_Mul(context, instr); break;
			case ir::PTXInstruction::Neg:
				eval_Neg(context, instr); break;
			case ir::PTXInstruction::Not:
				eval_Not(context, instr); break;
			case ir::PTXInstruction::Or:
				eval_Or(context, instr); break;
			case ir::PTXInstruction::Pmevent:
				eval_Pmevent(context, instr); break;
			case ir::PTXInstruction::Popc:
				eval_Popc(context, instr); break;
			case ir::PTXInstruction::Prefetch:
				eval_Prefetch(context, instr); break;
			case ir::PTXInstruction::MbarrierPendingCount:
				eval_MbarrierPendingCount(context, instr); break;
			case ir::PTXInstruction::MbarrierInit:
			case ir::PTXInstruction::MbarrierInval:
			case ir::PTXInstruction::CpAsyncMbarrierArrive:
			case ir::PTXInstruction::CpAsyncMbarrierArriveNoInc:
			case ir::PTXInstruction::MbarrierTestWait:
			case ir::PTXInstruction::MbarrierTestWaitParity:
			case ir::PTXInstruction::MbarrierArrive:
			case ir::PTXInstruction::MbarrierArriveDrop:
				eval_Mbarrier(context, instr); break;
			case ir::PTXInstruction::CpAsync:
				eval_CpAsync(context, instr); break;
			case ir::PTXInstruction::CpAsyncCommitGroup:
			case ir::PTXInstruction::CpAsyncWaitGroup:
			case ir::PTXInstruction::CpAsyncWaitAll:
				// Copies complete immediately in the CTA emulator.
				trace();
				break;
			case ir::PTXInstruction::Createpolicy:
				// Cache hints are not modeled by the CTA emulator; no policy is needed.
				trace();
				break;
			case ir::PTXInstruction::Prefetchu:
				eval_Prefetchu(context, instr); break;
			case ir::PTXInstruction::Prmt:
				eval_Prmt(context, instr); break;
			case ir::PTXInstruction::Rcp:
				eval_Rcp(context, instr); break;
			case ir::PTXInstruction::Red:
				eval_Red(context, instr); break;
			case ir::PTXInstruction::Rem:
				eval_Rem(context, instr); break;
			case ir::PTXInstruction::Ret:
				eval_Ret(context, instr); break;
			case ir::PTXInstruction::Rsqrt:
				eval_Rsqrt(context, instr); break;
			case ir::PTXInstruction::Sad:
				eval_Sad(context, instr); break;
			case ir::PTXInstruction::SelP:
				eval_SelP(context, instr); break;
			case ir::PTXInstruction::Set:
				eval_Set(context, instr); break;
			case ir::PTXInstruction::SetP:
				eval_SetP(context, instr); break;
			case ir::PTXInstruction::Shf:
				eval_Shf(context, instr); break;
			case ir::PTXInstruction::Shr:
				eval_Shr(context, instr); break;
			case ir::PTXInstruction::Shl:
				eval_Shl(context, instr); break;
			case ir::PTXInstruction::Sin:
				eval_Sin(context, instr); break;
			case ir::PTXInstruction::SlCt:
				eval_SlCt(context, instr); break;
			case ir::PTXInstruction::Sqrt:
				eval_Sqrt(context, instr); break;
			case ir::PTXInstruction::St:
				eval_St(context, instr); break;
			case ir::PTXInstruction::Sub:
				eval_Sub(context, instr); break;
			case ir::PTXInstruction::SubC:
				eval_SubC(context, instr); break;
			case ir::PTXInstruction::Szext:
				eval_Szext(context, instr); break;
			case ir::PTXInstruction::Tanh:
				eval_Tanh(context, instr); break;
			case ir::PTXInstruction::TestP:
				eval_TestP(context, instr); break;
			case ir::PTXInstruction::Tex:
				eval_Tex(context, instr); break;
			case ir::PTXInstruction::Trap:
				eval_Trap(context, instr); break;
			case ir::PTXInstruction::ShflSync:
				eval_WarpSync(context, instr); break;
			case ir::PTXInstruction::Match:
			case ir::PTXInstruction::Redux:
			case ir::PTXInstruction::Vote: {
				const bool warpSync = instr.voteSync || instr.opcode == ir::PTXInstruction::Redux || instr.opcode == ir::PTXInstruction::Match;
				if (warpSync) eval_WarpSync(context, instr);
				else reconvergenceMechanism->eval_Vote(context, instr);
				break;
			}
			case ir::PTXInstruction::Xor:
				eval_Xor(context, instr); break;
			case ir::PTXInstruction::Reconverge:
				eval_Reconverge(context, instr); break;
			default:
				assertM(false, "Opcode at pc "
					<< context.PC << " - "
					<< ir::PTXInstruction::toString(instr.opcode)
					<< " not supported.");
				break;
		}

		postTrace();

		running = reconvergenceMechanism->nextInstruction(
			getActiveContext(), instr, opcode);

		clock += 4;
		++counter;

	} while (running);

	if(getActiveContext().executionState == CTAContext::Exit)
	{
		finalize();
		report("kernel finished in " << counter << " instructions");
	}
}

void executive::CooperativeThreadArray::jumpToPC(int PC) {
	assert(reconvergenceMechanism->stackSize() != 0);
	assert(PC < (int)kernel->instructions.size());

	getActiveContext().PC = PC;
}

int executive::CooperativeThreadArray::getPC() const {
	assert(reconvergenceMechanism->stackSize() != 0);

	return getActiveContext().PC;
}

executive::CooperativeThreadArray::RegisterFile
	executive::CooperativeThreadArray::getCurrentRegisterFile() const {
	RegisterFile file(threadCount * functionCallStack.registerCount());
	RegisterFile::iterator ri = file.begin();
	for (int thread = 0; thread != threadCount; ++thread) {
		for (unsigned int reg = 0;
			reg != functionCallStack.registerCount(); ++reg, ++ri) {
			*ri = functionCallStack.registerFilePointer(thread)[reg];
		}
	}
	return file;
}

/*!
	gets the active context of the cooperative thread array
*/
executive::CTAContext & executive::CooperativeThreadArray::getActiveContext() {
	return reconvergenceMechanism->getContext();
}

const executive::CTAContext & executive::CooperativeThreadArray::getActiveContext() const {
	return reconvergenceMechanism->getContext();
}

executive::CTAContext::ExecutionState
	executive::CooperativeThreadArray::getExecutionState() const {
	return reconvergenceMechanism->getContext().executionState;
}

void executive::CooperativeThreadArray::setExecutionState(
	executive::CTAContext::ExecutionState state) {

	reconvergenceMechanism->getContext().executionState = state;
}

////////////////////////////////////////////////////////////////////////////////
//
// Execution support


// Register accessors

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU8 executive::CooperativeThreadArray::getRegAsU8(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU8 r = *( (ir::PTXU8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU16 executive::CooperativeThreadArray::getRegAsU16(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU16 r = *( (ir::PTXU16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU32 executive::CooperativeThreadArray::getRegAsU32(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU32 r = *( (ir::PTXU32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXU64 executive::CooperativeThreadArray::getRegAsU64(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXU64 r = *( (ir::PTXU64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << (void*)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << (void*)r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS8 executive::CooperativeThreadArray::getRegAsS8(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS8 r = *( (ir::PTXS8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS16 executive::CooperativeThreadArray::getRegAsS16(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS16 r = *( (ir::PTXS16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS32 executive::CooperativeThreadArray::getRegAsS32(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS32 r = *( (ir::PTXS32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXS64 executive::CooperativeThreadArray::getRegAsS64(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXS64 r = *( (ir::PTXS64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXF32 executive::CooperativeThreadArray::getRegAsF32(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXF32 r = *( (ir::PTXF32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(r))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXF64 executive::CooperativeThreadArray::getRegAsF64(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXF64 r = *( (ir::PTXF64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(r))
			<< std::dec << ")");
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB8 executive::CooperativeThreadArray::getRegAsB8(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB8 r = *( (ir::PTXB8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << (int)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << (int)r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB16 executive::CooperativeThreadArray::getRegAsB16(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB16 r = *( (ir::PTXB16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB32 executive::CooperativeThreadArray::getRegAsB32(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB32 r = *( (ir::PTXB32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\reg register index
*/
ir::PTXB64 executive::CooperativeThreadArray::getRegAsB64(int threadID,
	ir::PTXOperand::RegisterType reg) {
	ir::PTXB64 r = *( (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]));
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << (void*)r);
	}
	#else
	reportE(REPORT_REGISTER_READS, "   thread " << threadID
		<< " reg " << reg << " <= " << (void*)r);
	#endif
	return r;
}

/*!
	Gets a register value

	\param threadID ID of the active thread
	\param reg index of register
*/
bool executive::CooperativeThreadArray::getRegAsPredicate(int threadID,
	ir::PTXOperand::RegisterType reg) {
	bool r = (bool)getRegAsU64(threadID, reg);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_READS, "   thread " << threadID
			<< " reg " << reg << " <= " << r);
	}
	#else
	reportE(REPORT_REGISTER_READS && REPORT_PREDICATE_READS,
		"   thread " << threadID
		<< " reg " << reg << " <= " << r);
	#endif
	return r;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU8(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXU8 value) {
	ir::PTXU8* r = (ir::PTXU8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU16(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXU16 value) {
	ir::PTXU16* r = (ir::PTXU16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU32(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXU32 value) {
	ir::PTXU32* r = (ir::PTXU32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsU64(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXU64 value) {
	ir::PTXU64* r = (ir::PTXU64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << (void*)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << (void*)value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS8(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXS8 value) {
	ir::PTXS8* r = (ir::PTXS8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS16(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXS16 value) {
	ir::PTXS16* r = (ir::PTXS16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS32(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXS32 value) {
	ir::PTXS32* r = (ir::PTXS32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsS64(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXS64 value) {
	ir::PTXS64* r = (ir::PTXS64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsF32(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXF32 value) {
	ir::PTXF32* r = (ir::PTXF32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(value))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU32>(value))
			<< std::dec << ")");
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsF64(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXF64 value) {
	ir::PTXF64* r = (ir::PTXF64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(value))
			<< std::dec << ")");
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value
			<< " (0x" << std::hex << (hydrazine::bit_cast<ir::PTXU64>(value))
			<< std::dec << ")");
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB8(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXB8 value) {
	ir::PTXB8* r = (ir::PTXB8*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << (int)value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << (int)value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB16(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXB16 value) {
	ir::PTXB16* r = (ir::PTXB16*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB32(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXB32 value) {
	ir::PTXB32* r = (ir::PTXB32*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\reg register index
*/
void  executive::CooperativeThreadArray::setRegAsB64(int threadID,
	ir::PTXOperand::RegisterType reg, ir::PTXB64 value) {
	ir::PTXB64* r = (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

/*!
	Sets a register value

	\param threadID ID of the active thread
	\param reg index of register
	\param value value of predicate register
*/
void executive::CooperativeThreadArray::setRegAsPredicate(int threadID,
	ir::PTXOperand::RegisterType reg, bool value) {
	ir::PTXB64* r = (ir::PTXB64*)(
		&functionCallStack.registerFilePointer(threadID)[reg]);
	#if REPORT_NTH_THREAD_ONLY == 1
	if (threadID == NTH_THREAD) {
		reportE(REPORT_REGISTER_WRITES, "   thread " << threadID
			<< " reg " << reg << " value " << " => " << value );
	}
	#else
	reportE(REPORT_REGISTER_WRITES,
		"   thread " << threadID
		<< " reg " << reg << " value " << " => " << value );
	#endif
	*r = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
ir::PTXU8 executive::CooperativeThreadArray::operandAsU8(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU8(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU8)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU16 executive::CooperativeThreadArray::operandAsU16(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU16(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU16)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU32 executive::CooperativeThreadArray::operandAsU32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU32(threadID, op.reg);
		case ir::PTXOperand::Address:
			return (ir::PTXU32)(op.imm_uint) + op.offset;
		case ir::PTXOperand::Immediate:
			return (ir::PTXU32)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXU32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXU64 executive::CooperativeThreadArray::operandAsU64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsU64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsU64(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXU64)(op.imm_uint);
		case ir::PTXOperand::Address:
			return (ir::PTXU64)(op.imm_uint) + op.offset;
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS8 executive::CooperativeThreadArray::operandAsS8(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS8(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXS8)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS16 executive::CooperativeThreadArray::operandAsS16(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS16(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXS16)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS32 executive::CooperativeThreadArray::operandAsS32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS32(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXS32)(op.imm_int);
		case ir::PTXOperand::Special:
			return (ir::PTXS32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXS64 executive::CooperativeThreadArray::operandAsS64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsS64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsS64(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXS64)(op.imm_int);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXF32 executive::CooperativeThreadArray::operandAsF32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			return getRegAsF32(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXF32)(op.imm_single);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXF64 executive::CooperativeThreadArray::operandAsF64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			return getRegAsF64(threadID, op.reg);
		case ir::PTXOperand::Immediate:
			return (ir::PTXF64)(op.imm_float);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB8 executive::CooperativeThreadArray::operandAsB8(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB8(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB8(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB8)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB8)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB16 executive::CooperativeThreadArray::operandAsB16(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB16(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB16(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB16)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB16)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB32 executive::CooperativeThreadArray::operandAsB32(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB32(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB32(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB32)(op.imm_uint);
		case ir::PTXOperand::Special:
			return (ir::PTXB32)getSpecialValue(threadID, op.special, op.vIndex);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

ir::PTXB64 executive::CooperativeThreadArray::operandAsB64(int threadID,
	const ir::PTXOperand &op) {
	switch (op.addressMode) {
		case ir::PTXOperand::Indirect:
			return getRegAsB64(threadID, op.reg) + op.offset;
		case ir::PTXOperand::Register:
			return getRegAsB64(threadID, op.reg);
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			return (ir::PTXB64)(op.imm_uint);
		default:
			assert(0 == "invalid address mode of operand");
	}
	return 0;
}

bool executive::CooperativeThreadArray::operandAsPredicate(int threadID,
	const ir::PTXOperand &op) {
	bool result = false;
	switch (op.addressMode) {
		case ir::PTXOperand::Register:
			result = getRegAsPredicate(threadID, op.reg);
			break;
		case ir::PTXOperand::Immediate:
			return (bool)(op.imm_uint);
		default:
			assert(0 == "invalid address mode of operand");
	}

	switch (op.condition) {
		case ir::PTXOperand::PT:
			result = true;
			break;
		case ir::PTXOperand::nPT:
			result = false;
			break;
		case ir::PTXOperand::Pred:
			break;
		case ir::PTXOperand::InvPred:
			result = !result;
			break;
	}

	return result;
}

template<typename T>
T executive::CooperativeThreadArray::getFunctionParameter(int threadID,
	const ir::PTXInstruction& i, int index)
{
	assert((size_t)index < i.b.array.size());

	T value;

	char* base    = (char*) functionCallStack.stackFramePointer(threadID);
	void* address = base + i.b.array[index].offset;

	std::memcpy(&value, address, sizeof(T));

	return value;
}

template<typename T>
void executive::CooperativeThreadArray::setFunctionParameter(int threadID,
	const ir::PTXInstruction& i, int index, T value)
{
	assert((size_t)index < i.d.array.size());

	char* base    = (char*) functionCallStack.stackFramePointer(threadID);
	void* address = base + i.d.array[index].offset;

	std::memcpy(address, &value, sizeof(T));
}
////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Abs(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16 || instr.type == ir::PTXOperand::bf16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 a = operandAsU16(threadID, instr.a);
			if (instr.type == ir::PTXOperand::f16) a = ftzF16(instr.modifier, a);
			setRegAsB16(threadID, instr.d.reg, a & 0x7fff);
		}
	}
	else if (instr.type == ir::PTXOperand::f16x2 || instr.type == ir::PTXOperand::bf16x2) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			if (instr.type == ir::PTXOperand::f16x2)
				a = static_cast<ir::PTXU32>(ftzF16(instr.modifier, a)) |
					(static_cast<ir::PTXU32>(ftzF16(instr.modifier, a >> 16)) << 16);
			setRegAsU32(threadID, instr.d.reg, a & 0x7fff7fffu);
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {

			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, CTAAbs(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s8) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS8 d, a = operandAsS8(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS8(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a);
			d = CTAAbs(a);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Add(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16x2) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU16 al = static_cast<ir::PTXU16>(a), ah = a >> 16;
			ir::PTXU16 bl = static_cast<ir::PTXU16>(b), bh = b >> 16;
			ir::PTXU16 dl = ftzF16(effectiveModifier, toF16(
				roundedAdd(f16ToF32(ftzF16(effectiveModifier, al)),
					f16ToF32(ftzF16(effectiveModifier, bl)), effectiveModifier),
				effectiveModifier));
			ir::PTXU16 dh = ftzF16(effectiveModifier, toF16(
				roundedAdd(f16ToF32(ftzF16(effectiveModifier, ah)),
					f16ToF32(ftzF16(effectiveModifier, bh)), effectiveModifier),
				effectiveModifier));
			setRegAsU32(threadID, instr.d.reg, static_cast<ir::PTXU32>(dl) |
				(static_cast<ir::PTXU32>(dh) << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.a)));
			ir::PTXF32 b = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.b)));
			ir::PTXU16 d = toF16(roundedAdd(a, b, effectiveModifier), effectiveModifier);
			setRegAsB16(threadID, instr.d.reg, ftzF16(effectiveModifier, d));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier,
				roundedAdd(a, b, instr.modifier)));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);
			d = roundedAdd(a, b, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = a + b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::sat) {
				ir::PTXS64 temp = (ir::PTXS64)a + (ir::PTXS64)b;
				temp = min(temp, (ir::PTXS64)INT_MAX);
				temp = max(temp, (ir::PTXS64)INT_MIN);
				d = (ir::PTXS32)temp;
			}
			else if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXS32 carry = 0;
				hydrazine::add(d, carry, a, b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a + b;
			}
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = a + b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = a + b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXU32 carry = 0;
				hydrazine::add(d, carry, a, b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a + b;
			}
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = a + b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_AddC(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {

	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = 0,
				a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);

			ir::PTXU32 carry = getRegAsU32(threadID, instr.c.reg);
			hydrazine::add(d, carry, a, b, carry);

			setRegAsU32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = 0,
				a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);

			ir::PTXS32 carry = getRegAsS32(threadID, instr.c.reg);
			hydrazine::add(d, carry, a, b, carry);

			setRegAsS32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_And(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			bool d, a = operandAsPredicate(threadID, instr.a),
				b = operandAsPredicate(threadID, instr.b);
			d = b && a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a),
				b = operandAsB16(threadID, instr.b);
			d = a & b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 d, a = operandAsB32(threadID, instr.a),
				b = operandAsB32(threadID, instr.b);
			d = a & b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 d, a = operandAsB64(threadID, instr.a),
				b = operandAsB64(threadID, instr.b);
			d = a & b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Atom(CTAContext &context, const ir::PTXInstruction &instr) {
	evalAtomicRMW(context, instr, instr.atomicOperation, instr.d.addressMode != ir::PTXOperand::BitBucket);
}

// atom writes the pre-update value back to instr.d; red performs the same
// read-modify-write but discards it (writeback == false). red carries its
// operation in instr.reductionOperation (a distinct enum/op set from atom's
// instr.atomicOperation), so the caller resolves and passes the operation
// explicitly rather than this function reading instr.atomicOperation itself.
void executive::CooperativeThreadArray::evalAtomicRMW(CTAContext &context,
	const ir::PTXInstruction &instr, ir::PTXInstruction::AtomicOperation operation,
	bool writeback) {
	size_t elementSize = 0;
	const auto genericSource = [&](const char* source) {
		const bool address32 = instr.a.bytes() == sizeof(ir::PTXU32);
		if (address32) {
			const ir::PTXU32 address = (ir::PTXU64)source;
			const auto* shared = (const char*)functionCallStack.sharedMemoryPointer();
			const ir::PTXU32 offset = address - ir::PTXU32((ir::PTXU64)shared);
			if (offset < functionCallStack.sharedMemorySize()) return shared + offset;
			return (const char*)(ir::PTXU64)address;
		}
		return source;
	};

	switch (instr.type) {
		case ir::PTXOperand::b16:
		case ir::PTXOperand::f16:
			elementSize = sizeof(ir::PTXU16);
			break;
		case ir::PTXOperand::f16x2:
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::u64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::f64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Indirect:
					source += instr.a.offset;
				case ir::PTXOperand::Register:
					source += getRegAsU64(threadID, instr.a.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					source += instr.a.offset;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand",
						context.PC, instr);
			}

			if (instr.addressSpace == ir::PTXInstruction::Generic) source = genericSource(source);
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Indirect:
				source += instr.a.offset;
			case ir::PTXOperand::Register:
				source += getRegAsU64(threadID, instr.a.reg);
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				source += instr.a.offset;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand",
					context.PC, instr);
		}

		switch (instr.addressSpace) {
			case ir::PTXInstruction::Generic:
				source = genericSource(source);
				break;
			case ir::PTXInstruction::Global:
				{

				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			default:
				throw RuntimeException("unsupported address space",
					context.PC, instr);
		}

		switch (operation) {
			case ir::PTXInstruction::AtomicAnd:
				{
					if (instr.type == ir::PTXOperand::b32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						if (writeback) setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						*((ir::PTXB32*)source) = d & b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB32*)source) );
					}
					else if (instr.type == ir::PTXOperand::b64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						if (writeback) setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						*((ir::PTXB64*)source) = d & b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicOr:
				{
					if (instr.type == ir::PTXOperand::b32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						if (writeback) setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						*((ir::PTXB32*)source) = d | b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB32*)source) );
					}
					else if (instr.type == ir::PTXOperand::b64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						if (writeback) setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						*((ir::PTXB64*)source) = d | b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicXor:
				{
					if (instr.type == ir::PTXOperand::b32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						if (writeback) setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						*((ir::PTXB32*)source) = d ^ b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB32*)source) );
					}
					else if (instr.type == ir::PTXOperand::b64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						if (writeback) setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						*((ir::PTXB64*)source) = d ^ b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicCas:
				{
					if (instr.type == ir::PTXOperand::b16) {
						ir::PTXU16 old;
						std::memcpy(&old, source, sizeof(old));
						const ir::PTXU16 compare = operandAsU16(threadID, instr.b), replacement = operandAsU16(threadID, instr.c);
						const ir::PTXU16 result = old == compare ? replacement : old;
						std::memcpy((char*)source, &result, sizeof(result));
						if (writeback) setRegAsB16(threadID, instr.d.reg, old);
					}
					else if (instr.type == ir::PTXOperand::b32
						|| instr.type == ir::PTXOperand::s32
						|| instr.type == ir::PTXOperand::u32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						ir::PTXB32 c = operandAsB32(threadID, instr.c);
						if (writeback) setRegAsB32(threadID, instr.d.reg, d);
						*((ir::PTXB32*)source) = (d==b) ? c : d;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB32*)source) );
					}
					else if (instr.type == ir::PTXOperand::b64
						|| instr.type == ir::PTXOperand::s64
						|| instr.type == ir::PTXOperand::u64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						ir::PTXB64 c = operandAsB64(threadID, instr.c);
						if (writeback) setRegAsB64(threadID, instr.d.reg, d);
						*((ir::PTXB64*)source) = (d==b) ? c : d;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicExch:
				{
					if (instr.type == ir::PTXOperand::b32
						|| instr.type == ir::PTXOperand::s32
						|| instr.type == ir::PTXOperand::u32) {
						ir::PTXB32 d = *((ir::PTXB32*)source);
						if (writeback) setRegAsB32(threadID, instr.d.reg, d);
						ir::PTXB32 b = operandAsB32(threadID, instr.b);
						*((ir::PTXB32*)source) = b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB32*)source) );
					}
					else if (instr.type == ir::PTXOperand::b64
						|| instr.type == ir::PTXOperand::s64
						|| instr.type == ir::PTXOperand::u64) {
						ir::PTXB64 d = *((ir::PTXB64*)source);
						if (writeback) setRegAsB64(threadID, instr.d.reg, d);
						ir::PTXB64 b = operandAsB64(threadID, instr.b);
						*((ir::PTXB64*)source) = b;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXB64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicAdd:
				{
					if (instr.type == ir::PTXOperand::f16 || instr.type == ir::PTXOperand::f16x2) {
						const ir::PTXU32 value = elementSize == 2 ? operandAsU16(threadID, instr.b) : operandAsU32(threadID, instr.b);
						ir::PTXU32 original = 0;
						for (size_t half = 0; half < elementSize / 2; ++half) {
							ir::PTXU16 old;
							std::memcpy(&old, source + 2 * half, sizeof(old));
							original |= ir::PTXU32(old) << (16 * half);
							const ir::PTXU16 result = toF16(roundedAdd<double>(f16ToF32(old),
								f16ToF32(ir::PTXU16(value >> (16 * half))), ir::PTXInstruction::rn), ir::PTXInstruction::rn);
							std::memcpy((char*)source + 2 * half, &result, sizeof(result));
						}
						if (writeback) {
							if (elementSize == 2) setRegAsB16(threadID, instr.d.reg, original);
							else setRegAsB32(threadID, instr.d.reg, original);
						}
					}
					else if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						if (writeback) setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU32*)source) );
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						if (writeback) setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXS32*)source) );
					}
					else if (instr.type == ir::PTXOperand::f32) {
						ir::PTXF32 d;
						std::memcpy(&d, source, sizeof(d));
						const ir::PTXF32 b = operandAsF32(threadID, instr.b);
						const bool shared = instr.addressSpace == ir::PTXInstruction::Shared ||
							(instr.addressSpace == ir::PTXInstruction::Generic && addressInRange<ir::PTXU64>(
								(ir::PTXU64)source, (ir::PTXU64)functionCallStack.sharedMemoryPointer(), functionCallStack.sharedMemorySize()));
						const int modifiers = ir::PTXInstruction::rn | (shared ? 0 : ir::PTXInstruction::ftz);
						const ir::PTXF32 result = ftz(modifiers, roundedAdd(ftz(modifiers, d), ftz(modifiers, b), modifiers));
						std::memcpy((char*)source, &result, sizeof(result));
						if (writeback) setRegAsF32(threadID, instr.d.reg, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << result );
					}
					else if (instr.type == ir::PTXOperand::u64) {
						ir::PTXU64 d = *((ir::PTXU64*)source);
						if (writeback) setRegAsU64(threadID, instr.d.reg, d);
						ir::PTXU64 b = operandAsU64(threadID, instr.b);
						*((ir::PTXU64*)source) = b + d;
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU64*)source) );
					}
					else if (instr.type == ir::PTXOperand::f64) {
						ir::PTXF64 d;
						std::memcpy(&d, source, sizeof(d));
						const ir::PTXF64 b = operandAsF64(threadID, instr.b);
						const ir::PTXF64 result = roundedAdd(d, b, ir::PTXInstruction::rn);
						std::memcpy((char*)source, &result, sizeof(result));
						if (writeback) setRegAsF64(threadID, instr.d.reg, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << result );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicInc:
				{
					if(instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
					ir::PTXU32 d = *((ir::PTXU32*)source);
					if (writeback) setRegAsU32(threadID, instr.d.reg, d);
					ir::PTXU32 b = operandAsU32(threadID, instr.b);
					*((ir::PTXU32*)source) = (d >= b) ? 0 : d + 1;
					reportE(REPORT_ATOM, "Atomically updated " << d
						<< " to " << *((ir::PTXU32*)source) );
				}
				break;
			case ir::PTXInstruction::AtomicDec:
				{
					if(instr.type != ir::PTXOperand::u32) {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
					ir::PTXU32 d = *((ir::PTXU32*)source);
					if (writeback) setRegAsU32(threadID, instr.d.reg, d);
					ir::PTXU32 b = operandAsU32(threadID, instr.b);
					*((ir::PTXU32*)source) = ((d == 0) || (d > b)) ? b : d - 1;
					reportE(REPORT_ATOM, "Atomically updated " << d
						<< " to " << *((ir::PTXU32*)source) );
				}
				break;
			case ir::PTXInstruction::AtomicMin:
				{
					if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						if (writeback) setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU32*)source) );
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						if (writeback) setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXS32*)source) );
					}
					else if (instr.type == ir::PTXOperand::u64) {
						ir::PTXU64 d = *((ir::PTXU64*)source);
						if (writeback) setRegAsU64(threadID, instr.d.reg, d);
						ir::PTXU64 b = operandAsU64(threadID, instr.b);
						*((ir::PTXU64*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU64*)source) );
					}
					else if (instr.type == ir::PTXOperand::s64) {
						ir::PTXS64 d = *((ir::PTXS64*)source);
						if (writeback) setRegAsS64(threadID, instr.d.reg, d);
						ir::PTXS64 b = operandAsS64(threadID, instr.b);
						*((ir::PTXS64*)source) = min(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXS64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			case ir::PTXInstruction::AtomicMax:
				{
					if (instr.type == ir::PTXOperand::u32) {
						ir::PTXU32 d = *((ir::PTXU32*)source);
						if (writeback) setRegAsU32(threadID, instr.d.reg, d);
						ir::PTXU32 b = operandAsU32(threadID, instr.b);
						*((ir::PTXU32*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU32*)source) );
					}
					else if (instr.type == ir::PTXOperand::s32) {
						ir::PTXS32 d = *((ir::PTXS32*)source);
						if (writeback) setRegAsS32(threadID, instr.d.reg, d);
						ir::PTXS32 b = operandAsS32(threadID, instr.b);
						*((ir::PTXS32*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXS32*)source) );
					}
					else if (instr.type == ir::PTXOperand::u64) {
						ir::PTXU64 d = *((ir::PTXU64*)source);
						if (writeback) setRegAsU64(threadID, instr.d.reg, d);
						ir::PTXU64 b = operandAsU64(threadID, instr.b);
						*((ir::PTXU64*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXU64*)source) );
					}
					else if (instr.type == ir::PTXOperand::s64) {
						ir::PTXS64 d = *((ir::PTXS64*)source);
						if (writeback) setRegAsS64(threadID, instr.d.reg, d);
						ir::PTXS64 b = operandAsS64(threadID, instr.b);
						*((ir::PTXS64*)source) = max(b, d);
						reportE(REPORT_ATOM, "Atomically updated " << d
							<< " to " << *((ir::PTXS64*)source) );
					}
					else {
						throw RuntimeException("invalid data type",
							context.PC, instr);
					}
				}
				break;
			default:
				throw RuntimeException("Invalid atomic operation",
					context.PC, instr);
		}
	}
}

void executive::CooperativeThreadArray::eval_Bfi(CTAContext &context, const
	ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::b32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 pq = operandAsB32(threadID, instr.pq);
			ir::PTXB32 a = operandAsB32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU32 c = operandAsU32(threadID, instr.c);
			ir::PTXB32 d = hydrazine::bitFieldInsert(pq, a, b & 0xff, c & 0xff);
			setRegAsB32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::b64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 pq = operandAsB64(threadID, instr.pq);
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU32 c = operandAsU32(threadID, instr.c);
			ir::PTXB64 d = hydrazine::bitFieldInsert(pq, a, b & 0xff, c & 0xff);
			setRegAsB64(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Bfind(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::u32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a);
			d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::s32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			const ir::PTXS32 signedA = operandAsS32(threadID, instr.a);
			const ir::PTXU32 a = signedA < 0
				? ~static_cast<ir::PTXU32>(signedA)
				: static_cast<ir::PTXU32>(signedA);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::s64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			const ir::PTXS64 signedA = operandAsS64(threadID, instr.a);
			const ir::PTXU64 a = signedA < 0
				? ~static_cast<ir::PTXU64>(signedA)
				: static_cast<ir::PTXU64>(signedA);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::u64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 a = operandAsU64(threadID, instr.a);
			ir::PTXU32 d = hydrazine::bfind(a, instr.shiftAmount);
			setRegAsU32(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

void executive::CooperativeThreadArray::eval_Bmsk(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;

		const ir::PTXU32 a = operandAsU32(threadID, instr.a);
		const ir::PTXU32 b = operandAsU32(threadID, instr.b);
		const ir::PTXU32 a1 = a & 0x1f;
		const ir::PTXU32 b1 = b & 0x1f;
		const ir::PTXU32 sum = a1 + b1;
		const bool positionOverflow = instr.shiftMode ==
			ir::PTXInstruction::ShiftMode::Clamp && a >= 32;
		const bool widthOverflow = instr.shiftMode ==
			ir::PTXInstruction::ShiftMode::Clamp && b >= 32;
		const ir::PTXU32 mask0 = positionOverflow ? 0 : (~0u << a1);
		ir::PTXU32 mask1 = 0;
		if (sum < 32 && !positionOverflow && !widthOverflow) {
			mask1 = b1 == 0 ? ~0u : (~0u << sum);
		}

		setRegAsU32(threadID, instr.d.reg, mask0 & ~mask1);
	}
}

void executive::CooperativeThreadArray::eval_Brev(CTAContext& context,
	const ir::PTXInstruction& instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::b32: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB32 a = operandAsB32(threadID, instr.a);
			ir::PTXB32 d = hydrazine::brev(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
		break;
	}
	case ir::PTXOperand::b64: {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB64 d = hydrazine::brev(a);
			setRegAsB64(threadID, instr.d.reg, d);
		}
		break;
	}
	default: {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
	}
}

void executive::CooperativeThreadArray::eval_Bfe(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int tid = 0; tid < threadCount; tid++) {
		if (!context.predicated(tid, instr)) {
			continue;
		}
		bool size32bit = (instr.type == ir::PTXOperand::u32
			|| instr.type == ir::PTXOperand::s32);
		bool isSigned = (instr.type == ir::PTXOperand::s32
			|| instr.type == ir::PTXOperand::s64);
		ir::PTXU64 result = size32bit
			? hydrazine::bfe<ir::PTXU32>(operandAsU32(tid, instr.a),
				operandAsU32(tid, instr.b), operandAsU32(tid, instr.c), isSigned)
			: hydrazine::bfe<ir::PTXU64>(operandAsU64(tid, instr.a),
				operandAsU32(tid, instr.b), operandAsU32(tid, instr.c), isSigned);
		if (size32bit) {
			setRegAsU32(tid, instr.d.reg,
				hydrazine::bit_cast<ir::PTXU32, ir::PTXU64>(result));
		}
		else {
			setRegAsU64(tid, instr.d.reg, result);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Bar(CTAContext& context,
	const ir::PTXInstruction& instr) {
	trace();

	size_t participating = context.active.size();

	if (instr.a.addressMode != ir::PTXOperand::Invalid) {
		report("eval_Bar() - barrier name must be constant");
		throw RuntimeException("barrier name must be a constant in this "
			"version of Ocelot ir::PTX Emulator",
			context.PC, instr);
	}

	if (instr.b.addressMode != ir::PTXOperand::Invalid) {
		if (instr.b.addressMode == ir::PTXOperand::Immediate) {
			participating = (size_t)instr.b.imm_uint;
		}
		else {
			report("eval_Bar() - number of threads participating in barrier "
				"must be constant");
			throw RuntimeException(
				"number of threads participating in barrier must be "
					"constant operand in this version of Ocelot",
				context.PC, instr);
		}
	}

	if (participating != context.active.size()) {
		throw RuntimeException(
			"Ocelot does not yet support partial-CTA barriers",
			context.PC, instr);
	}

	reconvergenceMechanism->eval_Bar(context, instr);
}

void executive::CooperativeThreadArray::eval_Bra(CTAContext &context,
	const ir::PTXInstruction &instr) {
	using namespace boost;

	dynamic_bitset<> branch = context.active;
	dynamic_bitset<> fall_through(branch.size(), 0);

	// determine divergence
	for (int i = 0; i < threadCount; i++) {
		if (context.predicated(i, instr)) {
			// typical branch
			branch[i] = context.active[i];
			fall_through[i] = false;
		}
		else {
			fall_through[i] = context.active[i];
			branch[i] = false;
		}
	}

	if (traceEvents) {
		currentEvent.fallthrough = fall_through;
		currentEvent.taken = branch;
	}

#if REPORT_BRA
	report("  active threads       [" << context.active.count() << "] "
		<< context.active);
	report("  branching threads    [" << branch.count() << "] " << branch);
	report("  fall-through threads [" << fall_through.count() << "] "
		<< fall_through);
	report("  branch target PC " << instr.branchTargetInstruction);
	report("  reconverge PC " << instr.reconvergeInstruction);
#endif

	// dispatch to reconvergence mechanism
	reconvergenceMechanism->eval_Bra(context, instr, branch, fall_through);

	trace();
}

/*!
	Reconverge instruction is inserted into the ir::PTX during analysis and construction of the
	emulated kernel.
*/
void executive::CooperativeThreadArray::eval_Reconverge(
	CTAContext &context, const ir::PTXInstruction &instr) {
	using namespace std;
	trace();

	// dispatch
	reconvergenceMechanism->eval_Reconverge(context, instr);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Brkpt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	context.executionState = CTAContext::Break;
}

void executive::CooperativeThreadArray::copyArgument(const ir::PTXOperand& s,
	CTAContext& context) {
	reportE(REPORT_CALL, " Copying " << ir::PTXOperand::bytes(s.type)
		<< " bytes from previous stack frame at " << s.offset
		<< " to current frame at " << s.offset );
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.active[thread]) continue;
		char* stackPointer = (char*)functionCallStack.stackFramePointer(thread);
		char* previousStackPointer =
			(char*)functionCallStack.previousStackFramePointer(thread);
		std::memcpy(stackPointer + s.offset, previousStackPointer + s.offset,
			ir::PTXOperand::bytes(s.type));
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Call(CTAContext &context,
	const ir::PTXInstruction &instr) {
	typedef std::unordered_map<ir::PTXU64, CTAContext> TargetMap;
	trace();

	// copy any register operands into parameter stack memory
	for (ir::PTXOperand::Array::const_iterator
		argument = instr.b.array.begin();
		argument != instr.b.array.end(); ++argument) {
		if (argument->addressMode != ir::PTXOperand::Register) continue;

		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if(!context.predicated(threadID, instr)) continue;

			ir::PTXU64 data = getRegAsU64(threadID, argument->reg);

			char* base = (char*) functionCallStack.stackFramePointer(threadID);

			std::memcpy(base + argument->offset, &data,
				ir::PTXOperand::bytes(argument->type));
		}
	}

	// Is this a direct or indirect call?
	if (instr.a.addressMode == ir::PTXOperand::Register) {
		// Complex indirect call handling
		reportE(REPORT_CALL, " indirect call [" << context.active << "]" );

		// Create contexts for all unique targets
		TargetMap targets;

		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if(!context.predicated(threadID, instr)) continue;
			ir::PTXU64 targetPC = getRegAsU64(threadID, instr.a.reg);
			reportE(REPORT_CALL, "  Thread " << threadID
				<< " jumped to PC " << targetPC )
			TargetMap::iterator targetContext = targets.find(targetPC);
			if (targetContext == targets.end()) {
				CTAContext newContext(context);
				newContext.active.reset();
				newContext.PC = targetPC;
				targetContext = targets.insert(
					std::make_pair(targetPC, newContext)).first;
			}

			targetContext->second.active[threadID] = true;
		}

		reportE(REPORT_CALL, " Created " << targets.size()
			<< " unique target contexts.");

		// Get a reference to the base stack pointer
		unsigned int stackSize = functionCallStack.stackFrameSize();
		unsigned int stackOffset = functionCallStack.offset();
		unsigned int callPC = context.PC;
		functionCallStack.saveFrame();

		++context.PC;

		reportE(REPORT_CALL, " Caller stack starts at offset " << stackOffset);
		reportE(REPORT_CALL, " Caller size " << stackSize);

		// Push all active contexts
		for (TargetMap::iterator targetContext = targets.begin();
			targetContext != targets.end(); ++targetContext) {

			const EmulatedKernel* targetKernel = kernel->getKernel(
				targetContext->second.PC);
			if( targetKernel == 0 ) {
				std::stringstream stream;

				stream << targetContext->second.PC;

				throw RuntimeException("attempted to call function at PC '"
					+ stream.str() + "' in module '"
					+ ((std::ostringstream&)(std::ostringstream() << kernel->module->id())).str() +
					"', but there is no function at that PC.",
					context.PC, instr);
			}

			reportE(REPORT_CALL,
				"  call was taken, increasing stack size by ("
				<< targetKernel->parameterMemorySize() << " parameter) ("
				<< targetKernel->registerCount() << " registers) ("
				<< targetKernel->localMemorySize() << " local memory) ("
				<< targetKernel->totalSharedMemorySize()
				<< " sharedMemorySize)");

			functionCallStack.pushFrame(targetKernel->parameterMemorySize(),
				targetKernel->registerCount(), targetKernel->localMemorySize(),
				targetKernel->totalSharedMemorySize(), callPC,
				stackOffset, stackSize);
			for (ir::PTXOperand::Array::const_iterator
				argument = instr.b.array.begin();
				argument != instr.b.array.end(); ++argument) {
				for (int threadID = 0; threadID != threadCount; ++threadID) {
					if(!targetContext->second.active[threadID]) continue;
					char* base = (char*)
						functionCallStack.savedStackFramePointer(threadID);
					char* target = (char*)functionCallStack.stackFramePointer(
						threadID);
					std::memcpy(target + argument->offset,
						base + argument->offset,
						ir::PTXOperand::bytes(argument->type));
				}
			}

			reconvergenceMechanism->push(targetContext->second);
		}
	}
	else {

		// was this function external?
		ir::ExternalFunctionSet::ExternalFunction*
			external = kernel->findExternalFunction(instr.a.identifier);
		if(external != 0) {
			// get the prototype
			ir::Module::FunctionPrototypeMap::const_iterator prototype
				= kernel->module->prototypes().find(instr.a.identifier);
			if(prototype == kernel->module->prototypes().end()) {
				throw RuntimeException("no prototype for external function '"
					+ instr.a.identifier + "' in module '"
					+ ((std::ostringstream&)(std::ostringstream() << kernel->module->id())).str()
					+ "'", context.PC, instr);
			}

			for (int threadID = 0; threadID != threadCount; ++threadID) {
				if(!context.predicated(threadID, instr)) continue;
				reportE(REPORT_CALL, " thread " << threadID
					<< " calling external function " << instr.a.identifier);
				external->call(functionCallStack.stackFramePointer(threadID),
					prototype->second);
			}

			// copy register operands back from the stack
			for (ir::PTXOperand::Array::const_iterator
				argument = instr.d.array.begin();
				argument != instr.d.array.end(); ++argument) {
				if (argument->addressMode != ir::PTXOperand::Register) continue;

				for (int threadID = 0; threadID != threadCount; ++threadID) {
					if(!context.predicated(threadID, instr)) continue;

					ir::PTXU64 data = 0;
					char* base =
						(char*) functionCallStack.stackFramePointer(threadID);

					std::memcpy(&data, base + argument->offset,
						ir::PTXOperand::bytes(argument->type));
					setRegAsU64(threadID, argument->reg, data);
				}
			}

			++context.PC;
		}
		else
		{
			// Handle Cuda Nested Parallelism
			if (instr.a.identifier == "cudaLaunchDevice") {
				eval_cudaLaunchDevice(context, instr);
				return;
			}
			if (instr.a.identifier == "cudaDeviceSynchronize") {
				eval_cudaSynchronizeDevice(context, instr);
				return;
			}

			reportE(REPORT_CALL, " direct call to PC "
				<< instr.branchTargetInstruction)

			// Handle lazy function linking
			if (instr.branchTargetInstruction == -1) {
				reportE(REPORT_CALL, " lazy linking against kernel '"
					<< instr.a.identifier << "'");
				kernel->lazyLink(context.PC, instr.a.identifier);
				currentEvent.instruction = &currentInstruction(context);
			}

			const ir::PTXInstruction& jittedInstr = currentInstruction(context);
			assert(jittedInstr.branchTargetInstruction != -1);

			// Simple direct call handling
			if (jittedInstr.uni) {
				reportE(REPORT_CALL, " uniform direct call" );
				int firstActive = context.active.find_first();
				bool taken = context.predicated(firstActive, jittedInstr);

				if (taken) {
					reportE(REPORT_CALL,
						"  call was taken, increasing stack size by ("
						<< jittedInstr.a.stackMemorySize << " stack) ("
						<< jittedInstr.a.registerCount << " registers) ("
						<< jittedInstr.a.localMemorySize << " local memory) ("
						<< jittedInstr.a.sharedMemorySize
						<< " sharedMemorySize)");
					functionCallStack.pushFrame(jittedInstr.a.stackMemorySize,
						jittedInstr.a.registerCount,
						jittedInstr.a.localMemorySize,
						jittedInstr.a.sharedMemorySize, context.PC,
						functionCallStack.offset(),
						functionCallStack.stackFrameSize());
					for (ir::PTXOperand::Array::const_iterator
						argument = jittedInstr.b.array.begin();
						argument != jittedInstr.b.array.end(); ++argument) {
						copyArgument(*argument, context);
					}

					CTAContext targetContext(context);

					targetContext.PC = jittedInstr.branchTargetInstruction;
					++context.PC;

					reconvergenceMechanism->push(targetContext);
				}
			}
			else {
				reportE(REPORT_CALL, " divergent direct call" );

				CTAContext targetContext(context);

				targetContext.active = context.predicateMask(jittedInstr);

				reportE(REPORT_CALL, " taken threads - ["
					<< targetContext.active << "]" );

				if (targetContext.active.any()) {
					reportE(REPORT_CALL,
						"  call was taken, increasing stack size by ("
						<< jittedInstr.a.stackMemorySize << " stack) ("
						<< jittedInstr.a.registerCount << " registers) ("
						<< jittedInstr.a.localMemorySize << " local memory) ("
						<< jittedInstr.a.sharedMemorySize
						<< " sharedMemorySize)");
					functionCallStack.pushFrame(jittedInstr.a.stackMemorySize,
						jittedInstr.a.registerCount,
						jittedInstr.a.localMemorySize,
						jittedInstr.a.sharedMemorySize, context.PC,
						functionCallStack.offset(),
						functionCallStack.stackFrameSize());
					for (ir::PTXOperand::Array::const_iterator
						argument = jittedInstr.b.array.begin();
						argument != jittedInstr.b.array.end(); ++argument) {
						copyArgument(*argument, targetContext);
					}

					targetContext.PC = jittedInstr.branchTargetInstruction;

					++context.PC;
					reconvergenceMechanism->push(targetContext);
				}
				else {
					++context.PC;
				}
			}
		}
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Clz(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = hydrazine::countLeadingZeros(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB32 d = hydrazine::countLeadingZeros(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_CNot(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB16 d, a = operandAsB16(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a = operandAsB64(threadID, instr.a);
			d = (a == 0 ? 1 : 0);
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Fns(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;

		const ir::PTXU32 mask = operandAsB32(threadID, instr.a);
		const ir::PTXU32 base = operandAsB32(threadID, instr.b);
		const ir::PTXS32 offset = operandAsS32(threadID, instr.c);
		ir::PTXU32 result = 0xffffffffu;
		if (base > 31) {
			report("warning: fns base outside 0..31 is undefined by PTX; "
				"Ocelot returns 0xffffffff");
		}
		else if (offset == 0) {
			if ((mask >> base) & 1u) result = base;
		}
		else {
			int pos = static_cast<int>(base);
			ir::PTXS64 count = std::abs(static_cast<ir::PTXS64>(offset)) - 1;
			const int inc = offset > 0 ? 1 : -1;
			while (pos >= 0 && pos < 32) {
				if ((mask >> pos) & 1u) {
					if (count == 0) {
						result = pos;
						break;
					}
					--count;
				}
				pos += inc;
			}
		}
		setRegAsB32(threadID, instr.d.reg, result);
	}
}

void executive::CooperativeThreadArray::eval_CopySign(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;

			ir::PTXF32 a = operandAsF32(tid, instr.a);
			ir::PTXF32 b = operandAsF32(tid, instr.b);
			ir::PTXU32 d = (hydrazine::bit_cast<ir::PTXU32>(b) & 0x7fffffff) |
				(hydrazine::bit_cast<ir::PTXU32>(a) & 0x80000000);

			setRegAsF32(tid, instr.d.reg,
				hydrazine::bit_cast<ir::PTXF32>(d));
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;

			ir::PTXF64 a = operandAsF64(tid, instr.a);
			ir::PTXF64 b = operandAsF64(tid, instr.b);
			ir::PTXU64 d = (hydrazine::bit_cast<ir::PTXU64>(b)
				& 0x7fffffffffffffffULL) |
				(hydrazine::bit_cast<ir::PTXU64>(a)
				& 0x8000000000000000ULL);

			setRegAsF64(tid, instr.d.reg,
				hydrazine::bit_cast<ir::PTXF64>(d));
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cos(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d,
				a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, (ir::PTXF32)cos(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = cos(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

template< typename Destination, typename Source >
static Destination cvtInteger(Source value, int modifier) {
	if( !(modifier & ir::PTXInstruction::sat) ) {
		return static_cast<Destination>(value);
	}

	if( std::numeric_limits<Source>::is_signed ) {
		const ir::PTXS64 signedValue = static_cast<ir::PTXS64>(value);
		if( signedValue < 0 ) {
			if( !std::numeric_limits<Destination>::is_signed ) return 0;

			const ir::PTXS64 minimum = static_cast<ir::PTXS64>(
				(std::numeric_limits<Destination>::min)());
			return signedValue < minimum
				? (std::numeric_limits<Destination>::min)()
				: static_cast<Destination>(value);
		}
	}

	const ir::PTXU64 unsignedValue = static_cast<ir::PTXU64>(value);
	const ir::PTXU64 maximum = static_cast<ir::PTXU64>(
		(std::numeric_limits<Destination>::max)());
	return unsignedValue > maximum
		? (std::numeric_limits<Destination>::max)()
		: static_cast<Destination>(value);
}

template< typename Float >
static Float cvtSaturate(Float value, int modifier) {
	if( !(modifier & ir::PTXInstruction::sat) ) return value;
	if( hydrazine::isnan(value) || value <= 0 ) return 0;
	return value >= 1 ? 1 : value;
}

template< typename Int >
static ir::PTXF32 toF32(Int value, int modifier) {
	int mode = hydrazine::fegetround();
	if (modifier & ir::PTXInstruction::rn) {
		hydrazine::fesetround(FE_TONEAREST);
	} else if (modifier & ir::PTXInstruction::rz) {
		hydrazine::fesetround(FE_TOWARDZERO);
	} else if (modifier & ir::PTXInstruction::rm) {
		hydrazine::fesetround(FE_DOWNWARD);
	} else if (modifier & ir::PTXInstruction::rp) {
		hydrazine::fesetround(FE_UPWARD);
	}
	ir::PTXF32 d = cvtSaturate(static_cast<ir::PTXF32>(value), modifier);
	hydrazine::fesetround(mode);
	return d;
}

template< typename Int >
static ir::PTXF64 toF64(Int value, int modifier) {
	int mode = hydrazine::fegetround();
	if (modifier & ir::PTXInstruction::rn) {
		hydrazine::fesetround(FE_TONEAREST);
	} else if (modifier & ir::PTXInstruction::rz) {
		hydrazine::fesetround(FE_TOWARDZERO);
	} else if (modifier & ir::PTXInstruction::rm) {
		hydrazine::fesetround(FE_DOWNWARD);
	} else if (modifier & ir::PTXInstruction::rp) {
		hydrazine::fesetround(FE_UPWARD);
	}
	ir::PTXF64 d = cvtSaturate(static_cast<ir::PTXF64>(value), modifier);
	hydrazine::fesetround(mode);
	return d;
}

template< typename Source >
static ir::PTXU16 toF16(Source value, int modifier) {
	if( (modifier & ir::PTXInstruction::relu) && hydrazine::isnan(value) ) {
		return 0x7fff;
	}
	if( (modifier & ir::PTXInstruction::relu) && value < 0 ) value = 0;
	int mode = hydrazine::fegetround();
	if (modifier & ir::PTXInstruction::rn) {
		hydrazine::fesetround(FE_TONEAREST);
	} else if (modifier & ir::PTXInstruction::rz) {
		hydrazine::fesetround(FE_TOWARDZERO);
	} else if (modifier & ir::PTXInstruction::rm) {
		hydrazine::fesetround(FE_DOWNWARD);
	} else if (modifier & ir::PTXInstruction::rp) {
		hydrazine::fesetround(FE_UPWARD);
	}
	ir::PTXF16 half = value;
	if( modifier & ir::PTXInstruction::sat ) {
		half = cvtSaturate(static_cast<ir::PTXF32>(half), modifier);
	}
	hydrazine::fesetround(mode);
	ir::PTXU16 bits;
	std::memcpy(&bits, &half, sizeof(bits));
	return bits;
}

static ir::PTXU16 f32ToBF16(ir::PTXF32 value, int modifier) {
	if( (modifier & ir::PTXInstruction::relu) && value < 0 ) value = 0;
	ir::PTXU32 bits = hydrazine::bit_cast<ir::PTXU32>(value);
	if ((bits & 0x7fffffffU) > 0x7f800000U) {
		// NVIDIA canonical NaN
		return 0x7fffU;
	}
	ir::PTXU16 upper = static_cast<ir::PTXU16>(bits >> 16);
	if( modifier & ir::PTXInstruction::rz ) return upper;
	ir::PTXU16 lower = static_cast<ir::PTXU16>(bits & 0xffffU);
	if (lower > 0x8000U) {
		return upper + 1U;
	} else if (lower < 0x8000U) {
		return upper;
	}
	return upper + (upper & 1U);
}

static ir::PTXU16 f32ToBF16Rn(ir::PTXF32 value) {
	return f32ToBF16(value, ir::PTXInstruction::rn);
}

template< typename Float >
static Float roundToInt(Float a, int modifier) {
	Float fd = 0;
	if (modifier & ir::PTXInstruction::rni) {
		const int previous = hydrazine::fegetround();
		hydrazine::fesetround(FE_TONEAREST);
		fd = hydrazine::nearbyintf(a);
		hydrazine::fesetround(previous);
	} else if (modifier & ir::PTXInstruction::rzi) {
		fd = hydrazine::trunc(a);
	} else if (modifier & ir::PTXInstruction::rmi) {
		fd = floor(a);
	} else if (modifier & ir::PTXInstruction::rpi) {
		fd = ceil(a);
	}
	else {
		fd = a;
	}
	return fd;
}

template< typename Float >
static Float roundToInt(Float a, int modifier, executive::CTAContext &,
	const ir::PTXInstruction &) {
	return roundToInt(a, modifier);
}

template< typename Destination, typename Float >
static Destination cvtFloatToInteger(Float value, int modifier) {
	if (value != value) return 0;
	const Float rounded = roundToInt(value, modifier);
	const int bits = sizeof(Destination) * CHAR_BIT;
	const int magnitudeBits = std::numeric_limits<Destination>::is_signed
		? bits - 1 : bits;
	const Float upper = std::ldexp(Float(1), magnitudeBits);
	if (rounded >= upper) return (std::numeric_limits<Destination>::max)();
	if (std::numeric_limits<Destination>::is_signed && rounded
		< -upper) return (std::numeric_limits<Destination>::min)();
	if (!std::numeric_limits<Destination>::is_signed && rounded < 0) return 0;
	return static_cast<Destination>(rounded);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cvt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	auto setCvtB16 = [this](int threadID, ir::PTXOperand::RegisterType reg,
		ir::PTXU16 value) {
		setRegAsU64(threadID, reg, value);
	};
	auto setCvtF32 = [this](int threadID, ir::PTXOperand::RegisterType reg,
		ir::PTXF32 value) {
		setRegAsU64(threadID, reg,
			hydrazine::bit_cast<ir::PTXU32>(value));
	};
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;

		ir::PTXOperand::DataType sourceType = instr.a.type;

		if (instr.a.relaxedType != ir::PTXOperand::TypeSpecifier_invalid) {
			sourceType = instr.a.relaxedType;
		}
		if( instr.type == ir::PTXOperand::f16x2 ) {
			const ir::PTXU32 d =
				static_cast<ir::PTXU32>(toF16(
					operandAsF32(threadID, instr.a), instr.modifier)) << 16
				| toF16(operandAsF32(threadID, instr.b), instr.modifier);
			setRegAsU64(threadID, instr.d.reg, d);
			continue;
		}
		if( instr.type == ir::PTXOperand::bf16x2 ) {
			const ir::PTXU32 d =
				static_cast<ir::PTXU32>(f32ToBF16(
					operandAsF32(threadID, instr.a), instr.modifier)) << 16
				| f32ToBF16(operandAsF32(threadID, instr.b), instr.modifier);
			setRegAsU64(threadID, instr.d.reg, d);
			continue;
		}
		if( instr.type == ir::PTXOperand::tf32 ) {
			setRegAsU64(threadID, instr.d.reg,
				tf32FromF32Rna(operandAsF32(threadID, instr.a)));
			continue;
		}

		switch (sourceType) {
			case ir::PTXOperand::b8: // fall through
			case ir::PTXOperand::u8:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsB8(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::u64: // fall through
						{
							setRegAsU64(threadID, instr.d.reg,
								operandAsB8(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsU8(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsB8(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsB8(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s8:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsS8(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::s8: // fall through
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64:
						{
							ir::PTXS8 a = operandAsS8(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::b8:
						{
							setRegAsU64(threadID, instr.d.reg,
								cvtInteger<ir::PTXU8>(
									operandAsS8(threadID, instr.a), instr.modifier));
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU16>(
								operandAsS8(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU32>(
								operandAsS8(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU64>(
								operandAsS8(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsS8(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsS8(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b16: // fall through
			case ir::PTXOperand::u16:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsB16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsU16(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg,
								operandAsB16(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsU16(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS16 d = cvtInteger<ir::PTXS16>(
								operandAsU16(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsB16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsB16(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s16:
			{
				// s16 to one of the following
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsS16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsS16(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16: // fall through
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64:
						{
							ir::PTXS16 a = operandAsS16(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::u8: // fall through
					case ir::PTXOperand::b8:
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsS16(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU16>(
								operandAsS16(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU32>(
								operandAsS16(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU64>(
								operandAsS16(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsS16(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsS16(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b32: // fall through
			case ir::PTXOperand::u32:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsU32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsU32(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16:
						{
							ir::PTXU16 d = cvtInteger<ir::PTXU16>(
								operandAsU32(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32: // fall through
					case ir::PTXOperand::s64: // fall through
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg,
								operandAsU32(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsU32(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS16 d = cvtInteger<ir::PTXS16>(
								operandAsU32(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXS32 d = cvtInteger<ir::PTXS32>(
								operandAsU32(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsU32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsU32(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s32:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsS32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsS32(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16:
						{
							ir::PTXU16 d = cvtInteger<ir::PTXU16>(
								operandAsS32(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU32>(
								operandAsS32(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						setRegAsU64(threadID, instr.d.reg,
							cvtInteger<ir::PTXU64>(
								operandAsS32(threadID, instr.a), instr.modifier));
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsS32(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS16 d = cvtInteger<ir::PTXS16>(
								operandAsS32(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32: // fall through
					case ir::PTXOperand::s64:
						{
							ir::PTXS32 a = operandAsS32(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsS32(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsS32(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::s64:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsS64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8: // fall through
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::u16: // fall through
					case ir::PTXOperand::b16: // fall through
						{
							ir::PTXU16 d = cvtInteger<ir::PTXU16>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							ir::PTXU32 d = cvtInteger<ir::PTXU32>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							if(instr.modifier & ir::PTXInstruction::sat) {
								a = max(a, 0);
							}
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS16 d = cvtInteger<ir::PTXS16>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXS32 d = cvtInteger<ir::PTXS32>(
								operandAsS64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64:
						{
							ir::PTXS64 a = operandAsS64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg, a);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsS64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsS64(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::b64:
			case ir::PTXOperand::u64:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsU64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXU8 d = cvtInteger<ir::PTXU8>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						{
							ir::PTXU16 d = cvtInteger<ir::PTXU16>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							ir::PTXU32 d = cvtInteger<ir::PTXU32>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsU64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg,
								operandAsU64(threadID, instr.a));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXS8 d = cvtInteger<ir::PTXS8>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXS16 d = cvtInteger<ir::PTXS16>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXS32 d = cvtInteger<ir::PTXS32>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::s64:
						{
							ir::PTXS64 d = cvtInteger<ir::PTXS64>(
								operandAsU64(threadID, instr.a), instr.modifier);
							setRegAsS64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								toF32(operandAsU64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::f64:
						{
							setRegAsF64(threadID, instr.d.reg,
								toF64(operandAsU64(threadID, instr.a),
								instr.modifier));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::f16: // fall through
			case ir::PTXOperand::f32:
			{
				ir::PTXF32 a = sourceType == ir::PTXOperand::f16
					? f16ToF32(operandAsU16(threadID, instr.a))
					: ftz(instr.modifier, operandAsF32(threadID, instr.a));
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							a = roundToInt(a, instr.modifier);
							setCvtB16(threadID, instr.d.reg,
								toF16(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU8>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						{
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU16>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU32>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU64>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s8:
						{
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS8>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s16:
						{
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS16>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s32:
						{
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS32>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s64:
						{
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS64>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::f32:
						{
							a = roundToInt(a, instr.modifier, context,
								instr);

							setCvtF32(threadID, instr.d.reg,
								sat(instr.modifier, a));
						}
						break;
					case ir::PTXOperand::f64:
						{
							ir::PTXF64 d = roundToInt(a, instr.modifier);
							d = toF64(d, instr.modifier);
							setRegAsF64(threadID, instr.d.reg, d);
						}
						break;
					case ir::PTXOperand::bf16:
						{
							ir::PTXU16 d = f32ToBF16(a, instr.modifier);
							setCvtB16(threadID, instr.d.reg, d);
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::bf16:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(bf16ToF32(operandAsU16(threadID,
								instr.a)), instr.modifier));
						}
						break;
					case ir::PTXOperand::f32:
						{
							setCvtF32(threadID, instr.d.reg,
								bf16ToF32(operandAsU16(threadID, instr.a)));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			case ir::PTXOperand::f64:
			{
				switch (instr.type) {
					case ir::PTXOperand::f16:
						{
							setCvtB16(threadID, instr.d.reg,
								toF16(operandAsF64(threadID, instr.a),
								instr.modifier));
						}
						break;
					case ir::PTXOperand::pred: // fall through
					case ir::PTXOperand::b8: // fall through
					case ir::PTXOperand::u8:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU8>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b16: // fall through
					case ir::PTXOperand::u16:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU16>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b32: // fall through
					case ir::PTXOperand::u32:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU32>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::b64: // fall through
					case ir::PTXOperand::u64:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsU64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXU64>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s8:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS8>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s16:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS16>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s32:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS32>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::s64:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							setRegAsS64(threadID, instr.d.reg,
								cvtFloatToInteger<ir::PTXS64>(a, instr.modifier));
						}
						break;
					case ir::PTXOperand::f32:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							a = ftz(instr.modifier, toF32(a, instr.modifier));
							if(instr.modifier & ir::PTXInstruction::sat) {
								if (a != a) a = 0.0;
								a = min(1.0, a);
								a = max(a, 0.0);
							}
							setCvtF32(threadID, instr.d.reg,
								sat(instr.modifier, a));
						}
						break;
					case ir::PTXOperand::f64:
						{
							ir::PTXF64 a = operandAsF64(threadID, instr.a);
							a = roundToInt(a, instr.modifier);
							setRegAsF64(threadID, instr.d.reg,
								sat(instr.modifier, a));
						}
						break;
					default:
						throw RuntimeException("conversion not implemented",
							context.PC, instr);
						break;
				}
			}
			break;
			default:
				throw RuntimeException("conversion not implemented",
					context.PC, instr);
				break;
		}

	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Cvta(CTAContext &context,
	const ir::PTXInstruction &instr) {

	trace();

	if (!instr.toAddrSpace) {
		// convert to generic address space
		// source operand has explicitly address space

		switch (instr.type) {

		case ir::PTXOperand::u32:
		{
			ir::PTXU32 addrSpaceBase = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
				}
					break;
				case ir::PTXInstruction::Global: // DO NOTHING
				case ir::PTXInstruction::Local:  // DO NOTHING
					break;
				case ir::PTXInstruction::Param:
					addrSpaceBase = (ir::PTXU32)(kernel->scheduler
						? kernel->scheduler->argumentMemory()
						: (ir::PTXU64)kernel->ArgumentMemory);
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase,
						functionCallStack.sharedMemoryPointer());
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}

			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}

				ir::PTXU32 srcAddr = operandAsU32(tid, instr.a) + addrSpaceBase;

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU32 localMemPtr;
					if (instr.a.addressMode != ir::PTXOperand::Address
						|| !instr.a.isGlobalLocal) {
						hydrazine::bit_cast(localMemPtr,
							functionCallStack.localMemoryPointer(tid));
					}
					else {
						hydrazine::bit_cast(localMemPtr,
							functionCallStack.globalLocalMemoryPointer(tid));
					}

					srcAddr += localMemPtr;
				}

				setRegAsU32(tid, instr.d.reg, srcAddr);
			}
		}
			break;

		case ir::PTXOperand::u64:
		{
			ir::PTXU64 addrSpaceBase = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
				}
					break;
				case ir::PTXInstruction::Global: // DO NOTHING
				case ir::PTXInstruction::Local:  // DO NOTHING
					break;
				case ir::PTXInstruction::Param:
					addrSpaceBase = kernel->scheduler
						? kernel->scheduler->argumentMemory()
						: (ir::PTXU64)kernel->ArgumentMemory;
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase,
						functionCallStack.sharedMemoryPointer());
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}

			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}

				ir::PTXU64 srcAddr = operandAsU64(tid, instr.a) + addrSpaceBase;

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU64 localMemPtr;
					if (instr.a.addressMode != ir::PTXOperand::Address
						|| !instr.a.isGlobalLocal) {
						hydrazine::bit_cast(localMemPtr,
							functionCallStack.localMemoryPointer(tid));
					}
					else {
						hydrazine::bit_cast(localMemPtr,
							functionCallStack.globalLocalMemoryPointer(tid));
					}

					srcAddr += localMemPtr;
				}

				setRegAsU64(tid, instr.d.reg, srcAddr);
			}
		}
			break;

		default:
			throw RuntimeException("cvta instruction not valid for "
				"specified data size", context.PC, instr);
			break;
		};
	}
	else {
		// convert FROM generic address
		// dest operand has explicitly address space
		switch (instr.type) {
		case ir::PTXOperand::u32:
		{
			ir::PTXU32 addrSpaceBase = 0;
			ir::PTXU32 addrSpaceSize = 0;

			switch (instr.addressSpace) {
				case ir::PTXInstruction::Global: // DO NOTHING
				{
					addrSpaceSize = (ir::PTXU32)0xffffffff;
				}
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase,
						functionCallStack.sharedMemoryPointer());
					addrSpaceSize = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.sharedMemorySize());
				}
					break;
				case ir::PTXInstruction::Local: // DO NOTHING
				{

				}
					break;
				case ir::PTXInstruction::Param:
					addrSpaceBase = (ir::PTXU32)(kernel->scheduler
						? kernel->scheduler->argumentMemory()
						: (ir::PTXU64)kernel->ArgumentMemory);
					addrSpaceSize = kernel->argumentMemorySize();
					break;
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
					addrSpaceSize = kernel->constMemorySize();
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}


			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 srcAddr = operandAsU32(tid, instr.a);

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU32 localSize = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.localMemorySize());
					ir::PTXU32 localBase = hydrazine::bit_cast<ir::PTXU32>(
						functionCallStack.localMemoryPointer(tid));

					if (srcAddr >= localBase ||
						srcAddr < localSize + localBase) {
						addrSpaceSize = localSize;
						addrSpaceBase = localBase;
					}
					else {
						addrSpaceSize = hydrazine::bit_cast<ir::PTXU32>(
							functionCallStack.globalLocalMemorySize());
						addrSpaceBase = hydrazine::bit_cast<ir::PTXU32>(
							functionCallStack.globalLocalMemoryPointer(tid));
					}
				}

				if (srcAddr >= addrSpaceSize + addrSpaceBase
					|| srcAddr < addrSpaceBase) {
					throw RuntimeException("cvta instruction - source "
						"address is not part of addressed region",
						context.PC, instr);
				}

				srcAddr -= addrSpaceBase;

				setRegAsU32(tid, instr.d.reg, srcAddr);
			}
		}
			break;

		case ir::PTXOperand::u64:
		{
			ir::PTXU64 addrSpaceBase = 0;
			ir::PTXU64 addrSpaceSize = 0;
			switch (instr.addressSpace) {
				case ir::PTXInstruction::Global: // DO NOTHING
				{
					addrSpaceSize = (ir::PTXU64)0xffffffffffffffffULL;
				}
					break;
				case ir::PTXInstruction::Shared:
				{
					hydrazine::bit_cast(addrSpaceBase,
						functionCallStack.sharedMemoryPointer());
					addrSpaceSize = hydrazine::bit_cast<ir::PTXU64, size_t >(
						functionCallStack.sharedMemorySize());
				}
					break;
				case ir::PTXInstruction::Local:
				{

				}
					break;
				case ir::PTXInstruction::Param:
					addrSpaceBase = kernel->scheduler
						? kernel->scheduler->argumentMemory()
						: (ir::PTXU64)kernel->ArgumentMemory;
					addrSpaceSize = kernel->argumentMemorySize();
					break;
				case ir::PTXInstruction::Const:
				{
					hydrazine::bit_cast(addrSpaceBase, kernel->ConstMemory);
					addrSpaceSize = kernel->constMemorySize();
				}
					break;
				default:
					throw RuntimeException(
						"Cvta instruction cannot convert from specified "
							"address space to generic addresses",
						context.PC, instr);
					break;
			}

			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 srcAddr = operandAsU64(tid, instr.a);

				if (instr.addressSpace == ir::PTXInstruction::Local) {
					ir::PTXU64 localSize = hydrazine::bit_cast<ir::PTXU64>(
						functionCallStack.localMemorySize());
					ir::PTXU64 localBase = hydrazine::bit_cast<ir::PTXU64>(
						functionCallStack.localMemoryPointer(tid));

					if (srcAddr >= localBase ||
						srcAddr < localSize + localBase) {
						addrSpaceSize = localSize;
						addrSpaceBase = localBase;
					}
					else {
						addrSpaceSize = hydrazine::bit_cast<ir::PTXU64>(
							functionCallStack.globalLocalMemorySize());
						addrSpaceBase = hydrazine::bit_cast<ir::PTXU64>(
							functionCallStack.globalLocalMemoryPointer(tid));
					}
				}

				if (srcAddr >= addrSpaceSize + addrSpaceBase
					|| srcAddr < addrSpaceBase) {
					throw RuntimeException("cvta instruction - source "
						"address is not part of addressed region",
						context.PC, instr);
				}

				srcAddr -= addrSpaceBase;

				setRegAsU64(tid, instr.d.reg, srcAddr);
			}
		}
			break;

		default:
			throw RuntimeException("cvta instruction not valid for "
				"specified data size", context.PC, instr);
			break;
		};
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Div(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			if(ir::PTXInstruction::approx & instr.modifier) {
				if(std::fabs(b) > std::ldexp(1.0f, 126) && !hydrazine::isinf(b)) {
					if(hydrazine::isinf(a)) {
						d = std::numeric_limits<ir::PTXF32>::quiet_NaN();
					} else {
						const bool negative = std::signbit(a) != std::signbit(b);
						d = hydrazine::copysign(0.0f, negative ? -1.0f : 1.0f);
					}
				} else {
					d = roundedMul(a, roundedDiv(1.0f, b,
						ir::PTXInstruction::rn), ir::PTXInstruction::rn);
				}
			} else {
				d = roundedDiv(a, b, instr.modifier);
			}
			d = ftz(instr.modifier, d);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);
			d = roundedDiv(a, b, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zeroa at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			if(b == 0) {
				throw RuntimeException("Divide by zero at: "
					+ kernel->location(context.PC), context.PC, instr);
			}
			d = a / b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Dp(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	const bool fourWay = instr.opcode == ir::PTXInstruction::Dp4a;
	const unsigned int lanes = fourWay ? 4 : 2;
	const unsigned int aWidth = fourWay ? 8 : 16;
	const unsigned int bStart = !fourWay
		&& instr.modifier == ir::PTXInstruction::hi ? 2 : 0;
	auto extract = [](ir::PTXU32 value, unsigned int bit,
		unsigned int width, ir::PTXOperand::DataType type) {
		const ir::PTXU32 field = (value >> bit) & ((1u << width) - 1);
		if( type == ir::PTXOperand::s32 && (field & (1u << (width - 1))) ) {
			return static_cast<ir::PTXS32>(field) - (1 << width);
		}
		return static_cast<ir::PTXS32>(field);
	};
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;
		const ir::PTXU32 a = operandAsU32(threadID, instr.a);
		const ir::PTXU32 b = operandAsU32(threadID, instr.b);
		ir::PTXU32 d = operandAsU32(threadID, instr.c);
		for (unsigned int i = 0; i < lanes; ++i) {
			const ir::PTXS32 va = extract(a, i * aWidth, aWidth, instr.type);
			const ir::PTXS32 vb = extract(b, (bStart + i) * 8, 8, instr.bType);
			d += static_cast<ir::PTXU32>(va * vb);
		}
		setRegAsU32(threadID, instr.d.reg, d);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Ex2(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d,
				a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, hydrazine::exp2f(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(instr.modifier,
				operandAsU16(threadID, instr.a)));
			setRegAsB16(threadID, instr.d.reg,
				toF16(hydrazine::exp2f(a), instr.modifier));
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Exit(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	reconvergenceMechanism->eval_Exit(context, instr);
}


/*!

*/
void executive::CooperativeThreadArray::eval_Fma(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16x2) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			ir::PTXU32 a = operandAsU32(tid, instr.a), b = operandAsU32(tid, instr.b);
			ir::PTXU32 c = operandAsU32(tid, instr.c);
			ir::PTXU16 al = static_cast<ir::PTXU16>(a), ah = a >> 16;
			ir::PTXU16 bl = static_cast<ir::PTXU16>(b), bh = b >> 16;
			ir::PTXU16 cl = static_cast<ir::PTXU16>(c), ch = c >> 16;
			auto fma = [&](ir::PTXU16 x, ir::PTXU16 y, ir::PTXU16 z) {
				return ftzF16(instr.modifier, toF16(roundedFma<double>(
					f16ToF32(ftzF16(instr.modifier, x)),
					f16ToF32(ftzF16(instr.modifier, y)),
					f16ToF32(ftzF16(instr.modifier, z)), instr.modifier), instr.modifier));
			};
			ir::PTXU16 dl = fma(al, bl, cl), dh = fma(ah, bh, ch);
			setRegAsU32(tid, instr.d.reg, static_cast<ir::PTXU32>(dl) |
				(static_cast<ir::PTXU32>(dh) << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			ir::PTXF32 d = 0,
				a = ftz(instr.modifier, operandAsF32(tid, instr.a)),
				b = ftz(instr.modifier, operandAsF32(tid, instr.b)),
				c = ftz(instr.modifier, operandAsF32(tid, instr.c));

			d = ftz(instr.modifier, sat(instr.modifier,
				roundedFma(a, b, c, instr.modifier)));

			setRegAsF32(tid, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(tid, instr.a),
				b = operandAsF64(tid, instr.b), c = operandAsF64(tid, instr.c);
			d = roundedFma(a, b, c, instr.modifier);
			setRegAsF64(tid, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(instr.modifier,
				operandAsU16(tid, instr.a)));
			ir::PTXF32 b = f16ToF32(ftzF16(instr.modifier,
				operandAsU16(tid, instr.b)));
			ir::PTXF32 c = f16ToF32(ftzF16(instr.modifier,
				operandAsU16(tid, instr.c)));
			setRegAsB16(tid, instr.d.reg,
				ftzF16(instr.modifier, toF16(roundedFma<double>(
					a, b, c, instr.modifier), instr.modifier)));
		}
	}
	else if (instr.type == ir::PTXOperand::bf16) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;

			ir::PTXF32 a = bf16ToF32(operandAsU16(tid, instr.a));
			ir::PTXF32 b = bf16ToF32(operandAsU16(tid, instr.b));
			ir::PTXF32 c = bf16ToF32(operandAsU16(tid, instr.c));
			ir::PTXU16 d = f32ToBF16(fmaF32OneRound(a, b, c), instr.modifier);
			setRegAsB16(tid, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::bf16x2) {
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) continue;
			ir::PTXU32 a = operandAsU32(tid, instr.a), b = operandAsU32(tid, instr.b),
				c = operandAsU32(tid, instr.c);
			ir::PTXU16 al = static_cast<ir::PTXU16>(a), ah = a >> 16,
				bl = static_cast<ir::PTXU16>(b), bh = b >> 16;
			ir::PTXU16 cl = static_cast<ir::PTXU16>(c), ch = c >> 16;
			auto fma = [&](ir::PTXU16 x, ir::PTXU16 y, ir::PTXU16 z) {
				return f32ToBF16(fmaF32OneRound(bf16ToF32(x), bf16ToF32(y),
					bf16ToF32(z)), instr.modifier);
			};
			ir::PTXU16 dl = fma(al, bl, cl), dh = fma(ah, bh, ch);
			setRegAsU32(tid, instr.d.reg, static_cast<ir::PTXU32>(dl) |
				(static_cast<ir::PTXU32>(dh) << 16));
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Mma(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	const std::string error = instr.valid();
	if (!error.empty()) {
		throw RuntimeException(error, context.PC, instr);
	}

	const int completeThreads = (threadCount / 32) * 32;
	for (int threadID = completeThreads; threadID < threadCount; ++threadID) {
		if (context.predicated(threadID, instr)) {
			throw RuntimeException("mma requires all warp lanes to participate",
				context.PC, instr);
		}
	}

	const ir::PTXOperand::DataType inputType = instr.a.type;
	if (inputType == ir::PTXOperand::b1) {
		const int rows = instr.mmaShape == ir::PTXInstruction::MmaM8N8K128 ? 8 : 16;
		const int kWords = instr.mmaShape == ir::PTXInstruction::MmaM16N8K256 ? 8 : 4;
		for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
			int participants = 0;
			for (int lane = 0; lane < 32; ++lane) participants += context.predicated(warpStart + lane, instr);
			if (!participants) continue;
			if (participants != 32) {
				throw RuntimeException("mma requires all warp lanes to participate", context.PC, instr);
			}
			// Each word holds 32 consecutive K bits; A register pairs select rows and K halves.
			ir::PTXU32 A[16][8], B[8][8], D[16][8];
			for (int lane = 0; lane < 32; ++lane) {
				const int threadID = warpStart + lane;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (unsigned i = 0; i < instr.a.array.size(); ++i) {
					const int row = groupID + (i & 1) * 8, word = threadInGroup + (i / 2) * 4;
					A[row][word] = getRegAsB32(threadID, instr.a.array[i].reg);
				}
				for (unsigned i = 0; i < instr.b.array.size(); ++i) {
					const int word = threadInGroup + i * 4;
					B[word][groupID] = getRegAsB32(threadID, instr.b.array[i].reg);
				}
				for (unsigned i = 0; i < instr.c.array.size(); ++i) {
					const int row = groupID + (i / 2) * 8, col = threadInGroup * 2 + (i & 1);
					D[row][col] = getRegAsB32(threadID, instr.c.array[i].reg);
				}
			}
			for (int row = 0; row < rows; ++row) {
				for (int col = 0; col < 8; ++col) {
					for (int word = 0; word < kWords; ++word) {
						const ir::PTXU32 bits = instr.booleanOperator == ir::PTXInstruction::BoolXor
							? A[row][word] ^ B[word][col] : A[row][word] & B[word][col];
						D[row][col] += hydrazine::popc(bits);
					}
				}
			}
			for (int lane = 0; lane < 32; ++lane) {
				for (unsigned i = 0; i < instr.d.array.size(); ++i) {
					const int row = (lane >> 2) + (i / 2) * 8, col = (lane & 3) * 2 + (i & 1);
					setRegAsB32(warpStart + lane, instr.d.array[i].reg, D[row][col]);
				}
			}
		}
		return;
	}
	const bool subByteInput = inputType == ir::PTXOperand::s4 ||
		inputType == ir::PTXOperand::u4;
	const bool intInput = subByteInput || inputType == ir::PTXOperand::s8 ||
		inputType == ir::PTXOperand::u8;

	if (instr.mmaShape == ir::PTXInstruction::MmaM8N8K4 &&
		inputType == ir::PTXOperand::f64) {
		for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
			int participants = 0;
			for (int lane = 0; lane < 32; ++lane) {
				if (context.predicated(warpStart + lane, instr)) ++participants;
			}
			if (participants == 0) continue;
			if (participants != 32) {
				throw RuntimeException("mma requires all warp lanes to participate",
					context.PC, instr);
			}

			ir::PTXF64 A[8][4], B[4][8], D[8][8];
			for (int lane = 0; lane < 32; ++lane) {
				const int threadID = warpStart + lane;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				A[groupID][threadInGroup] = getRegAsF64(threadID, instr.a.array[0].reg);
				B[threadInGroup][groupID] = getRegAsF64(threadID, instr.b.array[0].reg);
				for (int i = 0; i < 2; ++i) {
					const int col = 2 * threadInGroup + i;
					D[groupID][col] = getRegAsF64(threadID, instr.c.array[i].reg);
				}
			}

			for (int row = 0; row < 8; ++row) {
				for (int col = 0; col < 8; ++col) {
					for (int k = 0; k < 4; ++k) {
						D[row][col] = roundedFma(A[row][k], B[k][col],
							D[row][col], instr.modifier);
					}
				}
			}

			for (int lane = 0; lane < 32; ++lane) {
				const int threadID = warpStart + lane;
				const int groupID = lane >> 2;
				const int threadInGroup = lane & 3;
				for (int i = 0; i < 2; ++i) {
					const int col = 2 * threadInGroup + i;
					setRegAsF64(threadID, instr.d.array[i].reg, D[groupID][col]);
				}
			}
		}
		return;
	}
	if (inputType == ir::PTXOperand::f64) {
		throw RuntimeException("unsupported f64 mma shape", context.PC, instr);
	}
	if (intInput) {
		// Packed 8-bit and 4-bit integer inputs accumulate into s32.
		auto signExtend = [](ir::PTXU32 reg, unsigned int elementIndex,
			ir::PTXOperand::DataType type) -> ir::PTXS32 {
			const unsigned bits = (type == ir::PTXOperand::s4 ||
				type == ir::PTXOperand::u4) ? 4 : 8;
			const ir::PTXU32 field = (reg >> (elementIndex * bits)) & ((1u << bits) - 1);
			if ((type == ir::PTXOperand::s4 || type == ir::PTXOperand::s8) &&
				(field & (1u << (bits - 1)))) {
				return static_cast<ir::PTXS32>(field) - (1 << bits);
			}
			return static_cast<ir::PTXS32>(field);
		};

		auto satfiniteClamp = [&instr](int64_t acc) -> ir::PTXS32 {
			if (instr.modifier & ir::PTXInstruction::satfinite) {
				const int64_t upper = (std::numeric_limits<ir::PTXS32>::max)();
				const int64_t lower = (std::numeric_limits<ir::PTXS32>::min)();
				if (acc > upper) acc = upper;
				else if (acc < lower) acc = lower;
			}
			return static_cast<ir::PTXS32>(acc);
		};

		for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
			int participants = 0;
			for (int lane = 0; lane < 32; ++lane) {
				if (context.predicated(warpStart + lane, instr)) {
					++participants;
				}
			}
			if (participants == 0) continue;
			if (participants != 32) {
				throw RuntimeException("mma requires all warp lanes to participate",
					context.PC, instr);
			}

			if (instr.mmaShape == ir::PTXInstruction::MmaM8N8K16 ||
				instr.mmaShape == ir::PTXInstruction::MmaM8N8K32) {
				const int elementsPerRegister = subByteInput ? 8 : 4;
				const int kCount = subByteInput ? 32 : 16;
				ir::PTXS32 A[8][32] = {};
				ir::PTXS32 B[32][8] = {};
				ir::PTXS32 C[8][8] = {};

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;

					ir::PTXU32 aReg = operandAsU32(threadID, instr.a.array[0]);
					for (int i = 0; i < elementsPerRegister; ++i) {
						A[groupID][threadInGroup * elementsPerRegister + i] =
							signExtend(aReg, i, instr.a.type);
					}

					ir::PTXU32 bReg = operandAsU32(threadID, instr.b.array[0]);
					for (int i = 0; i < elementsPerRegister; ++i) {
						B[threadInGroup * elementsPerRegister + i][groupID] =
							signExtend(bReg, i, instr.b.type);
					}

					for (int i = 0; i < 2; ++i) {
						C[groupID][threadInGroup * 2 + i] =
							operandAsS32(threadID, instr.c.array[i]);
					}
				}

				ir::PTXS32 D[8][8];
				for (int row = 0; row < 8; ++row) {
					for (int col = 0; col < 8; ++col) {
						int64_t acc = C[row][col];
						for (int k = 0; k < kCount; ++k) {
							acc += static_cast<int64_t>(A[row][k]) *
								static_cast<int64_t>(B[k][col]);
						}
						D[row][col] = satfiniteClamp(acc);
					}
				}

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;
					for (int i = 0; i < 2; ++i) {
						setRegAsS32(threadID, instr.d.array[i].reg,
							D[groupID][threadInGroup * 2 + i]);
					}
				}
			}
			else if ((instr.mmaShape == ir::PTXInstruction::MmaM16N8K32 && !subByteInput) ||
				instr.mmaShape == ir::PTXInstruction::MmaM16N8K64 ||
				(instr.mmaSparse && instr.mmaShape == ir::PTXInstruction::MmaM16N8K128)) {
				const int elementsPerRegister = subByteInput ? 8 : 4;
				const int kCount = instr.mmaSparse ? (instr.mmaShape == ir::PTXInstruction::MmaM16N8K128 ? 128 :
					instr.mmaShape == ir::PTXInstruction::MmaM16N8K64 ? 64 : 32) : elementsPerRegister * 8;
				ir::PTXS32 A[16][128] = {};
				ir::PTXS32 B[128][8] = {};
				ir::PTXS32 C[16][8] = {};

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;

					for (int i = 0; i < instr.a.array.size() * elementsPerRegister; ++i) {
						int row = ((i / elementsPerRegister) & 1) ? groupID + 8 : groupID;
						int col = threadInGroup * elementsPerRegister + i % elementsPerRegister +
							(i >= 2 * elementsPerRegister ? kCount / 2 : 0);
						if (instr.mmaSparse) {
							const int rowHalf = (i / elementsPerRegister) & 1, kHalf = i / (2 * elementsPerRegister);
							const int metadataLane = kCount == (subByteInput ? 128 : 64) ?
								rowHalf + 2 * kHalf : 2 * instr.mmaSelector.imm_uint + rowHalf;
							const ir::PTXU32 metadata = operandAsB32(warpStart + groupID * 4 + metadataLane, instr.q);
							// A 4-bit input index selects two consecutive values.
							const int valuesPerIndex = subByteInput ? 2 : 1;
							const int bit = threadInGroup * 8 + ((i % elementsPerRegister) / valuesPerIndex) * 2;
							const int chunk = (i % elementsPerRegister) / (2 * valuesPerIndex);
							col = threadInGroup * elementsPerRegister * 2 + chunk * 4 * valuesPerIndex +
								kHalf * (kCount / 2) + ((metadata >> bit) & 3) * valuesPerIndex + i % valuesPerIndex;
						}
						ir::PTXU32 reg = operandAsU32(threadID, instr.a.array[i / elementsPerRegister]);
						A[row][col] = signExtend(reg, i % elementsPerRegister, instr.a.type);
					}

					for (int i = 0; i < instr.b.array.size() * elementsPerRegister; ++i) {
						int row = threadInGroup * elementsPerRegister + i % elementsPerRegister +
							(i / elementsPerRegister) * (instr.mmaSparse ? elementsPerRegister * 4 : kCount / 2);
						int col = groupID;
						ir::PTXU32 reg = operandAsU32(threadID, instr.b.array[i / elementsPerRegister]);
						B[row][col] = signExtend(reg, i % elementsPerRegister, instr.b.type);
					}

					for (int i = 0; i < 4; ++i) {
						int row = groupID + (i >= 2 ? 8 : 0);
						int col = threadInGroup * 2 + (i & 1);
						C[row][col] = operandAsS32(threadID, instr.c.array[i]);
					}
				}

				ir::PTXS32 D[16][8];
				for (int row = 0; row < 16; ++row) {
					for (int col = 0; col < 8; ++col) {
						int64_t acc = C[row][col];
						for (int k = 0; k < kCount; ++k) {
							acc += static_cast<int64_t>(A[row][k]) *
								static_cast<int64_t>(B[k][col]);
						}
						D[row][col] = satfiniteClamp(acc);
					}
				}

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;
					for (int i = 0; i < 4; ++i) {
						int row = groupID + (i >= 2 ? 8 : 0);
						int col = threadInGroup * 2 + (i & 1);
						setRegAsS32(threadID, instr.d.array[i].reg, D[row][col]);
					}
				}
			}
			else if (instr.mmaShape == ir::PTXInstruction::MmaM16N8K16 ||
				(instr.mmaShape == ir::PTXInstruction::MmaM16N8K32 && subByteInput)) {
				const int elementsPerRegister = subByteInput ? 8 : 4;
				const int kCount = elementsPerRegister * 4;
				ir::PTXS32 A[16][32] = {};
				ir::PTXS32 B[32][8] = {};
				ir::PTXS32 C[16][8] = {};

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;

					for (int i = 0; i < 2 * elementsPerRegister; ++i) {
						int row = (i < elementsPerRegister) ? groupID : groupID + 8;
						int col = threadInGroup * elementsPerRegister + i % elementsPerRegister;
						ir::PTXU32 reg = operandAsU32(threadID, instr.a.array[i / elementsPerRegister]);
						A[row][col] = signExtend(reg, i % elementsPerRegister, instr.a.type);
					}

					for (int i = 0; i < elementsPerRegister; ++i) {
						int row = threadInGroup * elementsPerRegister + i;
						int col = groupID;
						ir::PTXU32 reg = operandAsU32(threadID, instr.b.array[0]);
						B[row][col] = signExtend(reg, i, instr.b.type);
					}

					for (int i = 0; i < 4; ++i) {
						int row = groupID + (i >= 2 ? 8 : 0);
						int col = threadInGroup * 2 + (i & 1);
						C[row][col] = operandAsS32(threadID, instr.c.array[i]);
					}
				}

				ir::PTXS32 D[16][8];
				for (int row = 0; row < 16; ++row) {
					for (int col = 0; col < 8; ++col) {
						int64_t acc = C[row][col];
						for (int k = 0; k < kCount; ++k) {
							acc += static_cast<int64_t>(A[row][k]) *
								static_cast<int64_t>(B[k][col]);
						}
						D[row][col] = satfiniteClamp(acc);
					}
				}

				for (int lane = 0; lane < 32; ++lane) {
					int threadID = warpStart + lane;
					int groupID = lane >> 2;
					int threadInGroup = lane & 3;
					for (int i = 0; i < 4; ++i) {
						int row = groupID + (i >= 2 ? 8 : 0);
						int col = threadInGroup * 2 + (i & 1);
						setRegAsS32(threadID, instr.d.array[i].reg, D[row][col]);
					}
				}
			}
			else {
				throw RuntimeException("unsupported integer mma shape",
					context.PC, instr);
			}
		}
		return;
	}

	if (instr.mmaShape == ir::PTXInstruction::MmaM8N8K4 &&
		inputType == ir::PTXOperand::f16) {
		for (int warp = 0; warp < completeThreads; warp += 32) {
			int participants = 0;
			for (int lane = 0; lane < 32; ++lane) participants += context.predicated(warp + lane, instr);
			if (!participants) continue;
			if (participants != 32) throw RuntimeException("mma requires all warp lanes to participate", context.PC, instr);
			// Four independent products: lanes 4*g..4*g+3 and 16+4*g..19+4*g.
			ir::PTXF32 A[4][8][4], B[4][4][8], D[4][8][8];
			for (int lane = 0; lane < 32; ++lane) {
				const int group = (lane >> 2) & 3, row = (lane & 3) + (lane >= 16 ? 4 : 0);
				for (int i = 0; i < 4; ++i) {
					const int aRow = instr.mmaAColumnMajor ? i + (lane >= 16 ? 4 : 0) : row;
					const int aCol = instr.mmaAColumnMajor ? lane & 3 : i;
					const int bRow = instr.mmaBColumnMajor ? i : lane & 3;
					const int bCol = instr.mmaBColumnMajor ? row : i + (lane >= 16 ? 4 : 0);
					A[group][aRow][aCol] = mmaHalf(*this, warp + lane, instr.a.array[i / 2], i & 1, inputType);
					B[group][bRow][bCol] = mmaHalf(*this, warp + lane, instr.b.array[i / 2], i & 1, inputType);
				}
				for (int i = 0; i < 8; ++i) {
					if (instr.c.type == ir::PTXOperand::f32) {
						const int cRow = (lane & 1) + (i & 2) + (lane >= 16 ? 4 : 0);
						const int cCol = (i & 4) + (lane & 2) + (i & 1);
						D[group][cRow][cCol] = operandAsF32(warp + lane, instr.c.array[i]);
					} else D[group][row][i] = mmaHalf(*this, warp + lane, instr.c.array[i / 2], i & 1, inputType);
				}
			}
			for (int group = 0; group < 4; ++group)
				for (int row = 0; row < 8; ++row)
					for (int col = 0; col < 8; ++col)
						for (int k = 0; k < 4; ++k) {
							D[group][row][col] = std::fma(A[group][row][k], B[group][k][col], D[group][row][col]);
							if (instr.type == ir::PTXOperand::f16) D[group][row][col] = f16ToF32(toF16(D[group][row][col], 0));
						}
			for (int lane = 0; lane < 32; ++lane) {
				const int group = (lane >> 2) & 3, row = (lane & 3) + (lane >= 16 ? 4 : 0);
				if (instr.type == ir::PTXOperand::f32) {
					for (int i = 0; i < 8; ++i) {
						const int dRow = (lane & 1) + (i & 2) + (lane >= 16 ? 4 : 0);
						const int dCol = (i & 4) + (lane & 2) + (i & 1);
						setRegAsF32(warp + lane, instr.d.array[i].reg, D[group][dRow][dCol]);
					}
				} else for (int i = 0; i < 4; ++i) setRegAsB32(warp + lane, instr.d.array[i].reg,
					toF16(D[group][row][2 * i], 0) | (static_cast<ir::PTXU32>(toF16(D[group][row][2 * i + 1], 0)) << 16));
			}
		}
		return;
	}

	if (instr.mmaShape == ir::PTXInstruction::MmaM8N8K4) {
		throw RuntimeException("unsupported m8n8k4 input type", context.PC, instr);
	}

	const bool halfAccumulator = instr.type == ir::PTXOperand::f16;
	const bool tf32Input = inputType == ir::PTXOperand::tf32;
	const bool sparseTf32K16 = instr.mmaSparse && tf32Input && instr.mmaShape == ir::PTXInstruction::MmaM16N8K16;
	const bool m16n8k8 = instr.mmaShape == ir::PTXInstruction::MmaM16N8K8;
	const bool m16n8k4 = instr.mmaShape == ir::PTXInstruction::MmaM16N8K4;
	const bool sparseK32 = instr.mmaSparse && instr.mmaShape == ir::PTXInstruction::MmaM16N8K32;
	for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
		int participants = 0;
		for (int lane = 0; lane < 32; ++lane) {
			if (context.predicated(warpStart + lane, instr)) {
				++participants;
			}
		}
		if (participants == 0) continue;
		if (participants != 32) {
			throw RuntimeException("mma requires all warp lanes to participate",
				context.PC, instr);
		}

		ir::PTXF32 A[16][32] = {};
		ir::PTXF32 B[32][8] = {};
		ir::PTXF32 C[16][8] = {};

		for (int lane = 0; lane < 32; ++lane) {
			int threadID = warpStart + lane;
			int groupID = lane >> 2;
			int threadInGroup = lane & 3;

			if (tf32Input) {
				for (int i = 0; i < (m16n8k4 || (instr.mmaSparse && !sparseTf32K16) ? 2 : 4); ++i) {
					int row = (i & 1) ? groupID + 8 : groupID;
					int col = threadInGroup + (i >= 2 ? 4 : 0);
					if (instr.mmaSparse) {
						const int metadataLane = sparseTf32K16 ? 2 * instr.mmaSelector.imm_uint + i / 2 : instr.mmaSelector.imm_uint;
						const int metadataThread = warpStart + groupID * 4 + metadataLane;
						const ir::PTXU32 metadata = operandAsB32(metadataThread, instr.q);
						const int bit = (i & 1) * 16 + threadInGroup * 4;
						// Legal indices 0x4 and 0xe select the first or second TF32 element.
						col = threadInGroup * 2 + (i / 2) * 8 + ((metadata >> (bit + 3)) & 1);
					}
					A[row][col] = tf32FromF32(operandAsF32(threadID,
						instr.a.array[i]));
				}

				for (int i = 0; i < (sparseTf32K16 ? 4 : m16n8k4 ? 1 : 2); ++i) {
					int row = threadInGroup + i * 4;
					int col = groupID;
					B[row][col] = tf32FromF32(operandAsF32(threadID,
						instr.b.array[i]));
				}
			}
			else if (m16n8k8) {
				for (int i = 0; i < 4; ++i) {
					int row = i < 2 ? groupID : groupID + 8;
					int col = threadInGroup * 2 + (i & 1);
					A[row][col] = mmaHalf(*this, threadID,
						instr.a.array[i / 2], i & 1, inputType);
				}

				for (int i = 0; i < 2; ++i) {
					int row = threadInGroup * 2 + (i & 1);
					B[row][groupID] = mmaHalf(*this, threadID,
						instr.b.array[0], i, inputType);
				}
			}
			else if (instr.mmaShape == ir::PTXInstruction::MmaM16N8K16 || sparseK32) {
				if (instr.mmaSparse) {
					for (int i = 0; i < (sparseK32 ? 8 : 4); ++i) {
						const int row = groupID + ((i / 2) & 1) * 8;
						// The selected pair splits K into two 16-column halves.
						const int metadataLane = sparseK32 ? 2 * instr.mmaSelector.imm_uint + i / 4 : instr.mmaSelector.imm_uint;
						const int metadataThread = warpStart + groupID * 4 + metadataLane;
						const ir::PTXU32 metadata = operandAsB32(metadataThread, instr.q);
						const int bit = ((i / 2) & 1) * 16 + threadInGroup * 4 + (i & 1) * 2;
						const int col = threadInGroup * 4 + (i / 4) * 16 + ((metadata >> bit) & 3);
						A[row][col] = mmaHalf(*this, threadID, instr.a.array[i / 2], i & 1, inputType);
					}
				}
				else for (int i = 0; i < 8; ++i) {
					int row = (i < 2 || (i >= 4 && i < 6)) ? groupID : groupID + 8;
					int col = threadInGroup * 2 + (i & 1) + (i >= 4 ? 8 : 0);
					A[row][col] = mmaHalf(*this, threadID,
						instr.a.array[i / 2], i & 1, inputType);
				}

				for (int i = 0; i < (sparseK32 ? 8 : 4); ++i) {
					int row = threadInGroup * 2 + (i & 1) + (i / 2) * 8;
					int col = groupID;
					B[row][col] = mmaHalf(*this, threadID,
						instr.b.array[i / 2], i & 1, inputType);
				}
			}

			for (int i = 0; i < 4; ++i) {
				int row = groupID + (i >= 2 ? 8 : 0);
				int col = threadInGroup * 2 + (i & 1);
				C[row][col] = halfAccumulator
					? mmaHalf(*this, threadID, instr.c.array[i / 2], i & 1,
						ir::PTXOperand::f16)
					: operandAsF32(threadID, instr.c.array[i]);
			}
		}

		ir::PTXF32 D[16][8];
		for (int row = 0; row < 16; ++row) {
			for (int col = 0; col < 8; ++col) {
				D[row][col] = C[row][col];
				for (int k = 0; k < (sparseK32 ? 32 : m16n8k4 ? 4 : m16n8k8 ? 8 : 16); ++k) {
					D[row][col] = std::fma(A[row][k], B[k][col], D[row][col]);
					if (halfAccumulator) {
						D[row][col] = f16ToF32(toF16(D[row][col], instr.modifier));
					}
				}
			}
		}

		for (int lane = 0; lane < 32; ++lane) {
			int threadID = warpStart + lane;
			int groupID = lane >> 2;
			int threadInGroup = lane & 3;
			if (halfAccumulator) {
				ir::PTXU32 packed[2] = {0, 0};
				for (int i = 0; i < 4; ++i) {
					int row = groupID + (i >= 2 ? 8 : 0);
					int col = threadInGroup * 2 + (i & 1);
					packed[i / 2] |= static_cast<ir::PTXU32>(
						toF16(D[row][col], instr.modifier)) << (16 * (i & 1));
				}
				setRegAsB32(threadID, instr.d.array[0].reg, packed[0]);
				setRegAsB32(threadID, instr.d.array[1].reg, packed[1]);
			}
			else for (int i = 0; i < 4; ++i) {
				int row = groupID + (i >= 2 ? 8 : 0);
				int col = threadInGroup * 2 + (i & 1);
				setRegAsF32(threadID, instr.d.array[i].reg,
					D[row][col]);
			}
		}
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Isspacep(CTAContext &context,
	const ir::PTXInstruction &instr) {

	trace();
	switch (instr.addressSpace) {
	case ir::PTXInstruction::Const:
	{
		if (instr.a.type == ir::PTXOperand::u32) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) continue;
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 base;
				hydrazine::bit_cast(base, kernel->ConstMemory);
				const ir::PTXU32 size = kernel->constMemorySize();
				setRegAsPredicate(tid, instr.d.reg,
					addressInRange(ptr, base, size));
			}
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) continue;
				ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				ir::PTXU64 base;
				hydrazine::bit_cast(base, kernel->ConstMemory);
				const ir::PTXU64 size = kernel->constMemorySize();
				setRegAsPredicate(tid, instr.d.reg,
					addressInRange(ptr, base, size));
			}
		}
	}
		break;
	case ir::PTXInstruction::Param:
	{
		if (instr.a.type == ir::PTXOperand::u32) {
			const ir::PTXU32 base = static_cast<ir::PTXU32>(kernel->scheduler
				? kernel->scheduler->argumentMemory()
				: reinterpret_cast<ir::PTXU64>(kernel->ArgumentMemory));
			const ir::PTXU32 size = static_cast<ir::PTXU32>(kernel->scheduler
				? kernel->scheduler->argumentMemorySize()
				: kernel->argumentMemorySize());
			for (int tid = 0; tid < threadCount; ++tid) {
				if (!context.predicated(tid, instr)) continue;
				const ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				setRegAsPredicate(tid, instr.d.reg,
					addressInRange(ptr, base, size));
			}
		}
		else {
			const ir::PTXU64 base = kernel->scheduler
				? kernel->scheduler->argumentMemory()
				: reinterpret_cast<ir::PTXU64>(kernel->ArgumentMemory);
			const ir::PTXU64 size = kernel->scheduler
				? kernel->scheduler->argumentMemorySize()
				: kernel->argumentMemorySize();
			for (int tid = 0; tid < threadCount; ++tid) {
				if (!context.predicated(tid, instr)) continue;
				const ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				setRegAsPredicate(tid, instr.d.reg,
					addressInRange(ptr, base, size));
			}
		}
	}
		break;
	case ir::PTXInstruction::Local:
	{
		if (instr.a.type == ir::PTXOperand::u32) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				const ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				setRegAsPredicate(tid, instr.d.reg,
					functionCallStack.isLocalMemoryAddress(ptr, tid, 32));
			}
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				const ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				setRegAsPredicate(tid, instr.d.reg,
					functionCallStack.isLocalMemoryAddress(ptr, tid, 64));
			}
		}
	}
		break;
	case ir::PTXInstruction::Shared:
	{
		if (instr.a.type == ir::PTXOperand::u32) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 sharedMemPtr;
				ir::PTXU32 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if (addressInRange(ptr, sharedMemPtr, sharedMemSize)) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			}
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				ir::PTXU64 sharedMemPtr;
				ir::PTXU64 sharedMemSize = functionCallStack.sharedMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				if (addressInRange(ptr, sharedMemPtr, sharedMemSize)) {
					setRegAsPredicate(tid, instr.d.reg, 1);
				}
				else {
					setRegAsPredicate(tid, instr.d.reg, 0);
				}
			}
		}
	}
		break;
	case ir::PTXInstruction::Global:
	{
		if (instr.a.type == ir::PTXOperand::u32) {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU32 ptr = operandAsU32(tid, instr.a);
				ir::PTXU32 sharedMemPtr;
				ir::PTXU32 sharedMemSize = functionCallStack.sharedMemorySize();
				ir::PTXU32 constMemPtr;
				const ir::PTXU32 constMemSize = kernel->constMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				hydrazine::bit_cast(constMemPtr, kernel->ConstMemory);
				const bool inShared = addressInRange(ptr, sharedMemPtr, sharedMemSize);
				const bool inLocal =
					functionCallStack.isLocalMemoryAddress(ptr, tid, 32);
				const bool inConst = addressInRange(ptr, constMemPtr, constMemSize);
				setRegAsPredicate(tid, instr.d.reg, !(inShared || inLocal || inConst));
			}
		}
		else {
			for (int tid = 0; tid < threadCount; tid++) {
				if (!context.predicated(tid, instr)) {
					continue;
				}
				ir::PTXU64 ptr = operandAsU64(tid, instr.a);
				ir::PTXU64 sharedMemPtr;
				ir::PTXU64 sharedMemSize = functionCallStack.sharedMemorySize();
				ir::PTXU64 constMemPtr;
				const ir::PTXU64 constMemSize = kernel->constMemorySize();
				hydrazine::bit_cast(sharedMemPtr,
					functionCallStack.sharedMemoryPointer());
				hydrazine::bit_cast(constMemPtr, kernel->ConstMemory);
				const bool inShared = addressInRange(ptr, sharedMemPtr, sharedMemSize);
				const bool inLocal =
					functionCallStack.isLocalMemoryAddress(ptr, tid, 64);
				const bool inConst = addressInRange(ptr, constMemPtr, constMemSize);
				setRegAsPredicate(tid, instr.d.reg, !(inShared || inLocal || inConst));
			}
		}
	}
		break;
	default:
		throw RuntimeException("isspacep - invalid address space for test",
			context.PC, instr);
		break;
	}

}

/*!

*/
void executive::CooperativeThreadArray::normalLoad(int threadID,
	const ir::PTXInstruction &instr, const char* source) {
	switch (instr.type) {
		case ir::PTXOperand::s8:
			{
				ir::PTXS8 word = *((ir::PTXS8*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::u8:
			{
				ir::PTXU8 word = *((ir::PTXU8*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::s16:		// fall through
			{
				ir::PTXS16 word = *((ir::PTXS16*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::u16:
			{
				ir::PTXU16 word = *((ir::PTXU16*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = *((ir::PTXF32*)source);
#if REPORT_LD

				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsF32(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::s32:
			{
				ir::PTXS32 word = *((ir::PTXS32*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsS64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::u32:
			{
				ir::PTXU32 word = *((ir::PTXU32*)source);
#if REPORT_LD
				report( "    Loaded " << word << " from "
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				ir::PTXU64 word = *((ir::PTXU64*)source);
#if REPORT_LD
				report( "    Loaded " << (void*)word << " from "
					<< (int*)source );
#endif
				setRegAsU64(threadID, instr.d.reg, word);
			}
			break;
		default:
			break;
	}
}

void executive::CooperativeThreadArray::vectorLoad(int threadID,
	const ir::PTXInstruction &instr, const char* source, unsigned int elementSize) {

	for (ir::PTXOperand::Array::const_iterator i = instr.d.array.begin();
		i != instr.d.array.end(); ++i, source += elementSize) {

		switch (instr.type) {
			case ir::PTXOperand::s8:
				{
					ir::PTXS8 word = *((ir::PTXS8*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b8:		// fall through
			case ir::PTXOperand::u8:
				{
					ir::PTXU8 word = *((ir::PTXU8*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::s16:
				{
					ir::PTXS16 word = *((ir::PTXS16*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b16:		// fall through
			case ir::PTXOperand::u16:
				{
					ir::PTXU16 word = *((ir::PTXU16*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::f32:
			{

				ir::PTXF32 word = *((ir::PTXF32*)((unsigned long long)source));

#if REPORT_LD
					report( "    Loaded " << word << " from "
						<< (int*)source << " - register " << instr.d.reg
						<< "; alternative " << i->reg);
#endif

				setRegAsF32(threadID, i->reg, word);
			}
			break;
			case ir::PTXOperand::s32:
				{
					ir::PTXS32 word = *((ir::PTXS32*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsS64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::b32:		// fall through
			case ir::PTXOperand::u32:
				{
					ir::PTXU32 word = *((ir::PTXU32*)source);

#if REPORT_LD
						report( "    Loaded " << word << " from "
							<< (int*)source );
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			case ir::PTXOperand::f64:		// fall through
			case ir::PTXOperand::b64:		// fall through
			case ir::PTXOperand::s64:		// fall through
			case ir::PTXOperand::u64:
				{
					ir::PTXU64 word = *((ir::PTXU64*)source);

#if REPORT_LD
						report( "    Loaded " << (void*)word << " from "
							<< (int*)source );
#endif

					setRegAsU64(threadID, i->reg, word);
				}
				break;
			default:
				break;
		}
	}
}


/*!

*/
void executive::CooperativeThreadArray::eval_Ld(CTAContext &context,
	const ir::PTXInstruction &instr) {

	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Register:
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.a.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand",
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				source += getRegAsU64(threadID, instr.a.reg);
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand",
					context.PC, instr);
		}

		source += instr.a.offset;

		switch (instr.addressSpace) {
			case ir::PTXInstruction::Param:
				{
					if (instr.a.isArgument) {
						source += kernel->scheduler->argumentMemory();
					}
					else {
						source += (ir::PTXU64)
							functionCallStack.stackFramePointer(threadID);
					}
				}
				break;
			case ir::PTXInstruction::Const:
				{
					source += (ir::PTXU64) kernel->ConstMemory;
				}
				break;
			case ir::PTXInstruction::Generic:
			case ir::PTXInstruction::Global:
				{
				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			case ir::PTXInstruction::Local:
				{
					if (instr.a.addressMode == ir::PTXOperand::Address &&
						instr.a.isGlobalLocal) {
						source += (ir::PTXU64)
							functionCallStack.globalLocalMemoryPointer(
							threadID);
					}
					else {
						source += (ir::PTXU64)
							functionCallStack.localMemoryPointer(threadID);
					}
				}
				break;
			default:
				throw RuntimeException("unsupported address space",
					context.PC, instr);
		}

		if(instr.d.vec == ir::PTXOperand::v1) {
			normalLoad(threadID, instr, source);
		}
		else {
			vectorLoad(threadID, instr, source, elementSize);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Ldmatrix(CTAContext& context,
	const ir::PTXInstruction& instr) {
	trace();
	const std::string error = instr.valid();
	if (!error.empty()) throw RuntimeException(error, context.PC, instr);
	const auto* shared = static_cast<const unsigned char*>(functionCallStack.sharedMemoryPointer());
	const ir::PTXU64 size = functionCallStack.sharedMemorySize();
	const int completeThreads = (threadCount / 32) * 32;
	for (int threadID = completeThreads; threadID < threadCount; ++threadID) {
		if (context.predicated(threadID, instr)) {
			throw RuntimeException("ldmatrix requires all warp lanes to participate",
				context.PC, instr);
		}
	}
	for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
		int participants = 0;
		for (int lane = 0; lane < 32; ++lane) participants += context.predicated(warpStart + lane, instr);
		if (!participants) continue;
		if (participants != 32)
			throw RuntimeException("ldmatrix requires all warp lanes to participate", context.PC, instr);
		ir::PTXU16 matrices[4][8][8];
		// Build the matrix
		for (int matrix = 0; matrix < int(instr.vec); ++matrix) {
			for (int row = 0; row < 8; ++row) {
				const int supplier = warpStart + matrix * 8 + row;
				const bool address32 = instr.a.bytes() == 4;
				ir::PTXU64 offset = address32 ? operandAsU32(supplier, instr.a) : operandAsU64(supplier, instr.a);
				if (instr.addressSpace == ir::PTXInstruction::Generic) {
					const auto base = (ir::PTXU64)shared;
					offset = address32 ? ir::PTXU32(offset - ir::PTXU32(base)) : offset - base;
				}
				if ((offset & 15) || offset > size || size - offset < 16)
					throw RuntimeException("ldmatrix row must be aligned and inside shared memory", context.PC, instr);
				std::memcpy(matrices[matrix][row], shared + offset, 16);
			}
		}
		for (int lane = 0; lane < 32; ++lane) {
			for (int matrix = 0; matrix < int(instr.vec); ++matrix) {
				ir::PTXU32 packed = 0;
				for (int half = 0; half < 2; ++half) {
					int row = lane / 4, col = lane % 4 * 2 + half;
					if (instr.matrixTranspose) std::swap(row, col);
					packed |= ir::PTXU32(matrices[matrix][row][col]) << (16 * half);
				}
				setRegAsB32(warpStart + lane, instr.d.array[matrix].reg, packed);
			}
		}
	}
}

void executive::CooperativeThreadArray::eval_Ldu(CTAContext &context,
	const ir::PTXInstruction &instr) {

	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.a.addressMode) {
				case ir::PTXOperand::Register:
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.a.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.a.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand",
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	if (instr.addressSpace != ir::PTXInstruction::Global &&
		instr.addressSpace != ir::PTXInstruction::Generic) {
		throw RuntimeException("ldu - only generic and .global address "
			"space supported", context.PC, instr);
		return;
	}

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		const char *source = 0;

		switch (instr.a.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				source += getRegAsU64(threadID, instr.a.reg);
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.a.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand",
					context.PC, instr);
		}

		source += instr.a.offset;

		// these should just work if we don't add any offsets to them since
		// it's a generic pointer

		if(instr.d.vec == ir::PTXOperand::v1) {
			normalLoad(threadID, instr, source);
		}
		else {
			vectorLoad(threadID, instr, source, elementSize);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Lg2(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, hydrazine::log2f(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Lop3(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;

		const ir::PTXU32 a = operandAsU32(threadID, instr.a);
		const ir::PTXU32 b = operandAsU32(threadID, instr.b);
		const ir::PTXU32 c = operandAsU32(threadID, instr.c);
		const ir::PTXU32 immLut = operandAsU32(threadID, instr.immLut);
		ir::PTXU32 d = 0;
		for (unsigned int bit = 0; bit < 32; ++bit) {
			const unsigned int aBit = (a >> bit) & 1u;
			const unsigned int bBit = (b >> bit) & 1u;
			const unsigned int cBit = (c >> bit) & 1u;
			const unsigned int index = (aBit << 2) | (bBit << 1) | cBit;
			const unsigned int resultBit = (immLut >> index) & 1u;

			d |= resultBit << bit;
		}
		if( instr.d.addressMode != ir::PTXOperand::BitBucket ) {
			setRegAsU32(threadID, instr.d.reg, d);
		}
		if( instr.pq.addressMode != ir::PTXOperand::Invalid ) {
			const bool q = operandAsPredicate(threadID, instr.q);
			const bool p = instr.booleanOperator == ir::PTXInstruction::BoolAnd
				? d != 0 && q : d != 0 || q;
			setRegAsPredicate(threadID, instr.pq.reg, p);
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Mad24(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("instruction not implemented", context.PC, instr);
}

/*!
	Mad:

		mad[.hi,.lo,.wide][.sat].itype d, a, b, c;
		mad[.rnd][.sat].ftype d, a, b, c;

		t = a * b;
		n = bitwidth of type;
		d = t + c;            // for floating-point and .wide
		d = t<2n-1..n> + c;   // for .hi variant
		d = t<n-1..0> + c;    // for .lo variant

*/
void executive::CooperativeThreadArray::eval_Mad(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	ir::PTXOperand::DataType type = instr.type;

	switch (type) {
	case ir::PTXOperand::u16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 a = operandAsU16(threadID, instr.a);
			ir::PTXU16 b = operandAsU16(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU16 c = operandAsU16(threadID, instr.c);
				ir::PTXU16 t = (ir::PTXU16)(((ir::PTXU32)a * (ir::PTXU32)b) >> 16);
				ir::PTXU16 d = t + c;
				setRegAsU16(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = (ir::PTXU32)a * (ir::PTXU32)b;
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU16 c = operandAsU16(threadID, instr.c);
				ir::PTXU16 t = a * b;
				ir::PTXU16 d = t + c;
				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = (ir::PTXU32)(((ir::PTXU64)a * (ir::PTXU64)b) >> 32);
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU64 c = operandAsU64(threadID, instr.c);
				ir::PTXU64 t = (ir::PTXU64)a * (ir::PTXU64)b;
				ir::PTXU64 d = t + c;
				setRegAsU64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU32 c = operandAsU32(threadID, instr.c);
				ir::PTXU32 t = a * b;
				ir::PTXU32 d = t + c;
				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::u64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d = 0,
				a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b),
				c = operandAsU64(threadID, instr.c);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXU64 hi = 0;
				ir::PTXU64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi + c;
			}
			else {
				ir::PTXU64 t = a * b;
				d = t + c;
			}
			setRegAsU64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 a = operandAsS16(threadID, instr.a);
			ir::PTXS16 b = operandAsS16(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS16 c = operandAsS16(threadID, instr.c);
				ir::PTXS16 t = (ir::PTXS16)(((ir::PTXS32)a * (ir::PTXS32)b) >> 16);
				ir::PTXS16 d = t + c;
				setRegAsS16(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = (ir::PTXS32)a * (ir::PTXS32)b;
				ir::PTXS32 d = t + c;
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS16 c = operandAsS16(threadID, instr.c);
				ir::PTXS16 t = a * b;
				ir::PTXS16 d = t + c;
				setRegAsS16(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 a = operandAsS32(threadID, instr.a);
			ir::PTXS32 b = operandAsS32(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = (ir::PTXS32)(((ir::PTXS64)a * (ir::PTXS64)b) >> 32);
				ir::PTXS32 d = 0;
				if (instr.modifier & ir::PTXInstruction::sat) {
					ir::PTXS64 td = (ir::PTXS64)t + (ir::PTXS64)c;
					td = max(td, (ir::PTXS64)INT_MIN);
					td = min(td, (ir::PTXS64)INT_MAX);
					d = td;
				}
				else {
					d = t + c;
				}
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS64 c = operandAsS64(threadID, instr.c);
				ir::PTXS64 t = (ir::PTXS64)a * (ir::PTXS64)b;
				ir::PTXS64 d = t + c;
				setRegAsS64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS32 c = operandAsS32(threadID, instr.c);
				ir::PTXS32 t = a * b;
				ir::PTXS32 d = t + c;
				setRegAsS32(threadID, instr.d.reg, d);
			}
		}
	} break;
	case ir::PTXOperand::s64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d = 0,
				a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b),
				c = operandAsS64(threadID, instr.c);

			if (instr.modifier & ir::PTXInstruction::hi) {
				ir::PTXS64 hi = 0;
				ir::PTXS64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi + c;
			}
			else {
				ir::PTXS64 t = a * b;
				d = t + c;
			}
			setRegAsS64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::f32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 d = 0,
				a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b)),
				c = ftz(instr.modifier, operandAsF32(threadID, instr.c));

			d = ftz(instr.modifier,
				sat(instr.modifier, roundedFma(a, b, c, instr.modifier)));

			setRegAsF32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::f64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF64 d = 0,
				a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b),
				c = operandAsF64(threadID, instr.c);

			d = roundedFma(a, b, c, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Max(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 a = operandAsF32(threadID, instr.a),
				b = operandAsF32(threadID, instr.b);
			setRegAsF32(threadID, instr.d.reg,
				minMaxF32(instr.modifier, a, b, true));
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = b;
			}
			else if(hydrazine::isnan(b))
			{
				d = a;
			}
			else
			{
				d = (a > b ? a : b);
			}

			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f16 || instr.type == ir::PTXOperand::bf16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d = minMaxHalf(instr.type, instr.modifier,
				operandAsU16(threadID, instr.a), operandAsU16(threadID, instr.b), true);

			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f16x2 || instr.type == ir::PTXOperand::bf16x2) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			const ir::PTXU32 a = operandAsU32(threadID, instr.a);
			const ir::PTXU32 b = operandAsU32(threadID, instr.b);
			const ir::PTXU16 aLow = static_cast<ir::PTXU16>(a);
			const ir::PTXU16 aHigh = static_cast<ir::PTXU16>(a >> 16);
			const ir::PTXU16 bLow = static_cast<ir::PTXU16>(b);
			const ir::PTXU16 bHigh = static_cast<ir::PTXU16>(b >> 16);
			const ir::PTXU32 low = static_cast<ir::PTXU32>(
				minMaxHalf(instr.type, instr.modifier, aLow, bLow, true));
			const ir::PTXU32 high = static_cast<ir::PTXU32>(
				minMaxHalf(instr.type, instr.modifier, aHigh, bHigh, true));
			setRegAsU32(threadID, instr.d.reg, low | (high << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = (a > b ? a : b);
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Min(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16 || instr.type == ir::PTXOperand::bf16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			setRegAsB16(threadID, instr.d.reg, minMaxHalf(instr.type, instr.modifier,
				operandAsU16(threadID, instr.a), operandAsU16(threadID, instr.b), false));
		}
	}
	else if (instr.type == ir::PTXOperand::f16x2 || instr.type == ir::PTXOperand::bf16x2) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			const ir::PTXU32 a = operandAsU32(threadID, instr.a);
			const ir::PTXU32 b = operandAsU32(threadID, instr.b);
			const ir::PTXU16 aLow = static_cast<ir::PTXU16>(a);
			const ir::PTXU16 aHigh = static_cast<ir::PTXU16>(a >> 16);
			const ir::PTXU16 bLow = static_cast<ir::PTXU16>(b);
			const ir::PTXU16 bHigh = static_cast<ir::PTXU16>(b >> 16);
			const ir::PTXU32 low = static_cast<ir::PTXU32>(
				minMaxHalf(instr.type, instr.modifier, aLow, bLow, false));
			const ir::PTXU32 high = static_cast<ir::PTXU32>(
				minMaxHalf(instr.type, instr.modifier, aHigh, bHigh, false));
			setRegAsU32(threadID, instr.d.reg, low | (high << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXF32 a = operandAsF32(threadID, instr.a),
				b = operandAsF32(threadID, instr.b);
			setRegAsF32(threadID, instr.d.reg,
				minMaxF32(instr.modifier, a, b, false));
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);

			if(hydrazine::isnan(a))
			{
				d = b;
			}
			else if(hydrazine::isnan(b))
			{
				d = a;
			}
			else
			{
				d = (a < b ? a : b);
			}

			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = (a < b ? a : b);
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Membar(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	/*! No need to do anything here. */
}

/*!

*/
void executive::CooperativeThreadArray::eval_Fence(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	/*! No need to do anything here -- see eval_Membar; this simulator executes
		every instruction to completion for every thread before the next one
		starts, so there is no reordering for fence to guard against. */
}

////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Activemask(CTAContext& context,
	const ir::PTXInstruction& instr) {
	trace();
	const std::string error = instr.valid();
	if (!error.empty()) throw RuntimeException(error, context.PC, instr);
	for (int warpStart = 0; warpStart < threadCount; warpStart += 32) {
		ir::PTXU32 mask = 0;
		const int lanes = min(32, threadCount - warpStart);
		for (int lane = 0; lane < lanes; ++lane)
			if (context.predicated(warpStart + lane, instr)) mask |= ir::PTXU32(1) << lane;
		for (int lane = 0; lane < lanes; ++lane)
			if (context.predicated(warpStart + lane, instr)) setRegAsB32(warpStart + lane, instr.d.reg, mask);
	}
}

void executive::CooperativeThreadArray::eval_Movmatrix(CTAContext& context,
	const ir::PTXInstruction& instr) {
	trace();
	const std::string error = instr.valid();
	if (!error.empty()) throw RuntimeException(error, context.PC, instr);
	const int completeThreads = (threadCount / 32) * 32;
	for (int threadID = completeThreads; threadID < threadCount; ++threadID) {
		if (context.predicated(threadID, instr)) {
			throw RuntimeException("movmatrix requires all warp lanes to participate",
				context.PC, instr);
		}
	}
	for (int warpStart = 0; warpStart < completeThreads; warpStart += 32) {
		int participants = 0;
		for (int lane = 0; lane < 32; ++lane) participants += context.predicated(warpStart + lane, instr);
		if (!participants) continue;
		if (participants != 32)
			throw RuntimeException("movmatrix requires all warp lanes to participate", context.PC, instr);
		ir::PTXU16 matrix[8][8];
		// Gather all source fragments before writing potentially aliased destinations.
		for (int row = 0; row < 8; ++row) {
			for (int col = 0; col < 8; col += 2) {
				const int lane = row * 4 + col / 2;
				const ir::PTXU32 packed = operandAsB32(warpStart + lane, instr.a);
				matrix[row][col] = ir::PTXU16(packed);
				matrix[row][col + 1] = ir::PTXU16(packed >> 16);
			}
		}
		for (int lane = 0; lane < 32; ++lane) {
			ir::PTXU32 packed = 0;
			for (int half = 0; half < 2; ++half) {
				const int row = lane / 4, col = (lane % 4) * 2 + half;
				packed |= ir::PTXU32(matrix[col][row]) << (16 * half);
			}
			setRegAsB32(warpStart + lane, instr.d.reg, packed);
		}
	}
}

void executive::CooperativeThreadArray::eval_Mov(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	if (instr.a.addressMode == ir::PTXOperand::Register) {
		eval_Mov_reg(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::Special) {
		eval_Mov_sreg(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::Immediate) {
		eval_Mov_imm(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::Address) {
		eval_Mov_addr(context, instr);
	}
	else if (instr.a.addressMode == ir::PTXOperand::FunctionName) {
		eval_Mov_func(context, instr);
	}
	else {
		throw RuntimeException(std::string("unimplemented address mode ")
			+ ir::PTXOperand::toString(instr.a.addressMode), context.PC, instr);
	}
}

template<typename D, typename S>
static D gather(S s0, S s1)
{
	D result = s1;
	result <<= sizeof(S) * 8;
	result |= s0;
	return result;
}

template<typename D, typename S>
static D gather(S s0, S s1, S s2, S s3)
{
	D result = s3;
	result <<= sizeof(S) * 8;
	result |= s2;
	result <<= sizeof(S) * 8;
	result |= s1;
	result <<= sizeof(S) * 8;
	result |= s0;
	return result;
}

template<typename D, unsigned int index, unsigned int bytes, typename S>
static D scatter(S s)
{
	ir::PTXU64 mask = 1;
	mask <<= bytes * 8;
	mask -= 1;
	s >>= index * bytes * 8;
	s &= mask;
	return s;
}

void executive::CooperativeThreadArray::eval_Mov_reg(CTAContext &context,
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;

		switch (instr.type) {
		case ir::PTXOperand::u16:
		case ir::PTXOperand::s16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				setRegAsU16(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				if (instr.a.array.size() == 0) {
					a = operandAsU32(threadID, instr.a);
				}
				else {
					assert(instr.a.array.size() == 2);
					a = gather<ir::PTXB16>(operandAsB8(threadID, instr.a.array[0]),
						operandAsB8(threadID, instr.a.array[1]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsB16(threadID, instr.d.reg, a);
				}
				else {
					assert(instr.d.array.size() == 2);
					setRegAsB8(threadID, instr.d.array[0].reg,
						scatter<ir::PTXB8, 0, 1>(a));
					setRegAsB8(threadID, instr.d.array[1].reg,
						scatter<ir::PTXB8, 1, 1>(a));
				}
			}
			break;
		case ir::PTXOperand::u32:
		case ir::PTXOperand::s32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				setRegAsU32(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				if (instr.a.array.size() == 0) {
					a = operandAsU32(threadID, instr.a);
				}
				else if (instr.a.array.size() == 2) {
					a = gather<ir::PTXB32>(operandAsB16(threadID,
						instr.a.array[0]), operandAsB16(threadID,
						instr.a.array[1]));
				}
				else {
					assert(instr.a.array.size() == 4);
					a = gather<ir::PTXB32>(operandAsB8(threadID, instr.a.array[0]),
						operandAsB8(threadID, instr.a.array[1]),
						operandAsB8(threadID, instr.a.array[2]),
						operandAsB8(threadID, instr.a.array[3]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsU32(threadID, instr.d.reg, a);
				}
				else if (instr.d.array.size() == 2) {
					setRegAsU16(threadID, instr.d.array[0].reg,
						scatter<ir::PTXB16, 0, 2>(a));
					setRegAsU16(threadID, instr.d.array[1].reg,
						scatter<ir::PTXB16, 1, 2>(a));
				}
				else {
					assert(instr.d.array.size() == 4);
					setRegAsB8(threadID, instr.d.array[0].reg,
						scatter<ir::PTXB8, 0, 1>(a));
					setRegAsB8(threadID, instr.d.array[1].reg,
						scatter<ir::PTXB8, 1, 1>(a));
					setRegAsB8(threadID, instr.d.array[2].reg,
						scatter<ir::PTXB8, 2, 1>(a));
					setRegAsB8(threadID, instr.d.array[3].reg,
						scatter<ir::PTXB8, 3, 1>(a));
				}
			}
			break;
		case ir::PTXOperand::u64:
		case ir::PTXOperand::s64:
			{
				ir::PTXU64 a = operandAsU64(threadID, instr.a);
				setRegAsU64(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::b64:
			{
				ir::PTXU64 a;
				if (instr.a.array.size() == 0) {
					a = operandAsU64(threadID, instr.a);
				}
				else if (instr.a.array.size() == 2) {
					a = gather<ir::PTXB64>(operandAsB32(threadID,
						instr.a.array[0]),
						operandAsB32(threadID, instr.a.array[1]));
				}
				else {
					assert(instr.a.array.size() == 4);
					a = gather<ir::PTXB64>(
						operandAsB16(threadID, instr.a.array[0]),
						operandAsB16(threadID, instr.a.array[1]),
						operandAsB16(threadID, instr.a.array[2]),
						operandAsB16(threadID, instr.a.array[3]));
				}
				if (instr.d.array.size() == 0) {
					setRegAsB64(threadID, instr.d.reg, a);
				}
				else if (instr.d.array.size() == 2) {
					setRegAsB32(threadID, instr.d.array[0].reg,
						scatter<ir::PTXB32, 0, 4>(a));
					setRegAsB32(threadID, instr.d.array[1].reg,
						scatter<ir::PTXB32, 1, 4>(a));
				}
				else {
					assert(instr.d.array.size() == 4);
					setRegAsB16(threadID, instr.d.array[0].reg,
						scatter<ir::PTXB16, 0, 2>(a));
					setRegAsB16(threadID, instr.d.array[1].reg,
						scatter<ir::PTXB16, 1, 2>(a));
					setRegAsB16(threadID, instr.d.array[2].reg,
						scatter<ir::PTXB16, 2, 2>(a));
					setRegAsB16(threadID, instr.d.array[3].reg,
						scatter<ir::PTXB16, 3, 2>(a));
				}
			}
			break;
		case ir::PTXOperand::f32:
			{
				setRegAsF32(threadID, instr.d.reg,
					operandAsF32(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::f64:
			{
				setRegAsF64(threadID, instr.d.reg,
					operandAsF64(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::pred:
			{
				setRegAsPredicate(threadID, instr.d.reg,
					operandAsU64(threadID, instr.a));
			}
			break;
		default:
			break;
		}
	}
}

void executive::CooperativeThreadArray::eval_Mov_sreg(CTAContext &context,
	const ir::PTXInstruction &instr) {

	if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d = operandAsU16(threadID, instr.a);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = operandAsU32(threadID, instr.a);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = operandAsS32(threadID, instr.a);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Mov_imm(CTAContext &context,
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;

		switch (instr.d.type) {
		case ir::PTXOperand::u16:
		case ir::PTXOperand::s16:
		case ir::PTXOperand::b16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				setRegAsU16(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::f16:
			{
				ir::PTXU16 a = operandAsU16(threadID, instr.a);
				setRegAsB16(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::u32:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::b32:
			{
				ir::PTXU32 a = operandAsU32(threadID, instr.a);
				setRegAsU32(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::u64:
		case ir::PTXOperand::s64:
		case ir::PTXOperand::b64:
			{
				ir::PTXU64 a = operandAsU64(threadID, instr.a);
				setRegAsU64(threadID, instr.d.reg, a);
			}
			break;
		case ir::PTXOperand::f32:
			{
				setRegAsF32(threadID, instr.d.reg,
					operandAsF32(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::f64:
			{
				setRegAsF64(threadID, instr.d.reg,
					operandAsF64(threadID, instr.a));
			}
			break;
		case ir::PTXOperand::pred:
			{
				setRegAsPredicate(threadID, instr.d.reg,
					operandAsU64(threadID, instr.a));
			}
			break;
		default:
			break;
		}
	}
}

void executive::CooperativeThreadArray::eval_Mov_indirect(CTAContext &context,
	const ir::PTXInstruction &instr) {

}

void executive::CooperativeThreadArray::eval_Mov_addr(CTAContext &context,
	const ir::PTXInstruction &instr) {
	eval_Mov_imm(context, instr);
}

void executive::CooperativeThreadArray::eval_Mov_func(CTAContext &context,
	const ir::PTXInstruction &instr) {
	if (instr.branchTargetInstruction == -1) {
		kernel->lazyLink(context.PC, instr.a.identifier);
	}

	const ir::PTXInstruction& jittedInstr = currentInstruction(context);

	currentEvent.instruction = &jittedInstr;

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, jittedInstr)) continue;
		setRegAsU64(threadID, jittedInstr.d.reg,
			jittedInstr.branchTargetInstruction);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Mul24(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();

	if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 a = operandAsU32(threadID, instr.a), b = operandAsU32(threadID, instr.b);
			ir::PTXU64 dw = ( a & 0x00ffffff ) * ( b & 0x00ffffff );
			ir::PTXU32 d;
			if (instr.modifier & ir::PTXInstruction::hi) {
				d = ((dw >> 12) & 0xffffffff);
			}
			else {
				d = (dw & 0xffffffff);
			}
			setRegAsU32(threadID, instr.d.reg, d);

		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 a = operandAsS32(threadID, instr.a), b = operandAsS32(threadID, instr.b);

			if( a < 0 )
			{
				a |= 0xffffffffff000000ULL;
			}
			else
			{
				a &= 0x00ffffff;
			}
			if( b < 0 )
			{
				b |= 0xffffffffff000000ULL;
			}
			else
			{
				b &= 0x00ffffff;
			}

			ir::PTXS64 dw = a * b;
			ir::PTXS32 d;
			if (instr.modifier & ir::PTXInstruction::hi) {
				d = ((dw >> 12) & 0xffffffff);
			}
			else {
				d = dw;
			}
			setRegAsS32(threadID, instr.d.reg, d);

		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Mul(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16x2) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU16 al = static_cast<ir::PTXU16>(a), ah = a >> 16;
			ir::PTXU16 bl = static_cast<ir::PTXU16>(b), bh = b >> 16;
			ir::PTXU16 dl = ftzF16(effectiveModifier, toF16(
				roundedMul(f16ToF32(ftzF16(effectiveModifier, al)),
					f16ToF32(ftzF16(effectiveModifier, bl)), effectiveModifier),
				effectiveModifier));
			ir::PTXU16 dh = ftzF16(effectiveModifier, toF16(
				roundedMul(f16ToF32(ftzF16(effectiveModifier, ah)),
					f16ToF32(ftzF16(effectiveModifier, bh)), effectiveModifier),
				effectiveModifier));
			setRegAsU32(threadID, instr.d.reg, static_cast<ir::PTXU32>(dl) |
				(static_cast<ir::PTXU32>(dh) << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.a)));
			ir::PTXF32 b = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.b)));
			ir::PTXU16 d = toF16(roundedMul(a, b, effectiveModifier), effectiveModifier);
			setRegAsB16(threadID, instr.d.reg, ftzF16(effectiveModifier, d));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier,
				roundedMul(a, b, instr.modifier)));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a), b = operandAsF64(threadID, instr.b);
			d = roundedMul(a, b, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 a = operandAsU16(threadID, instr.a), b = operandAsU16(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU32 d = a * b;
				setRegAsU32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU32 dw = a * b;
				ir::PTXU16 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 16) & 0x0ffff);
				}
				else {
					d = (dw & 0x0ffff);
				}
				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 a = operandAsU32(threadID, instr.a), b = operandAsU32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXU64 d = a * b;
				setRegAsU64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXU64 dw = a * b;
				ir::PTXU32 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 32) & 0x0ffffffff);
				}
				else {
					d = (dw & 0x0ffffffff);
				}
				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 a = operandAsS16(threadID, instr.a), b = operandAsS16(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS32 d = a * b;
				setRegAsS32(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS32 dw = a * b;
				ir::PTXS16 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 16) & 0x0ffff);
				}
				else {
					d = (dw & 0x0ffff);
				}
				setRegAsS16(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 a = operandAsS32(threadID, instr.a), b = operandAsS32(threadID, instr.b);
			if (instr.modifier & ir::PTXInstruction::wide) {
				ir::PTXS64 d = a * b;
				setRegAsS64(threadID, instr.d.reg, d);
			}
			else {
				ir::PTXS64 dw = a * b;
				ir::PTXS32 d;
				if (instr.modifier & ir::PTXInstruction::hi) {
					d = ((dw >> 32) & 0x0ffffffff);
				}
				else {
					d = dw;
				}
				setRegAsS32(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		if (instr.modifier & ir::PTXInstruction::lo) {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
					b = operandAsS64(threadID, instr.b);
				d = a * b;
				setRegAsS64(threadID, instr.d.reg, d);
			}
		}
		else {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
					b = operandAsS64(threadID, instr.b);
				ir::PTXS64 hi = 0;
				ir::PTXS64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi;
				setRegAsS64(threadID, instr.d.reg, d);
			}
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		if (instr.modifier & ir::PTXInstruction::lo) {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				d = a * b;
				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		else {
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXU64 hi = 0;
				ir::PTXU64 lo = 0;
				hydrazine::multiplyHiLo( hi, lo, a, b );
				d = hi;
				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Neg(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16x2 || instr.type == ir::PTXOperand::bf16x2) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			a = static_cast<ir::PTXU32>(ftzF16(instr.modifier,
				static_cast<ir::PTXU16>(a))) |
				(static_cast<ir::PTXU32>(ftzF16(instr.modifier, a >> 16)) << 16);
			setRegAsU32(threadID, instr.d.reg, a ^ 0x80008000u);
		}
	}
	else if (instr.type == ir::PTXOperand::bf16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = ftz(instr.modifier,
				bf16ToF32(operandAsU16(threadID, instr.a)));
			ir::PTXU16 d = f32ToBF16Rn(-a);
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(instr.modifier,
				operandAsU16(threadID, instr.a)));
			ir::PTXU16 d = toF16(-a, instr.modifier);
			setRegAsB16(threadID, instr.d.reg, ftzF16(instr.modifier, d));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = operandAsF32(threadID, instr.a);
			d = ftz(instr.modifier, -a);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = -a;
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a);
			d = a == (std::numeric_limits<ir::PTXS16>::min)() ? a : -a;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a);
			d = a == (std::numeric_limits<ir::PTXS32>::min)() ? a : -a;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a);
			d = a == (std::numeric_limits<ir::PTXS64>::min)() ? a : -a;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Not(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			bool d, a = operandAsPredicate(threadID, instr.a);
			d = !a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB16 d, a = operandAsB16(threadID, instr.a);
			d = ~a;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = ~a;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a = operandAsB64(threadID, instr.a);
			d = ~a;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Or(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			bool d, a = operandAsPredicate(threadID, instr.a),
				b = operandAsPredicate(threadID, instr.b);
			d = b || a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB16 d, a = operandAsB16(threadID, instr.a),
				b = operandAsB16(threadID, instr.b);
			d = a | b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a),
				b = operandAsB32(threadID, instr.b);
			d = a | b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a = operandAsB64(threadID, instr.a),
				b = operandAsB64(threadID, instr.b);
			d = a | b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Pmevent(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	/*! No need to do anything here. */
}

/*!

*/
void executive::CooperativeThreadArray::eval_Popc(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a);
			d = hydrazine::popc(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXB64 a = operandAsB64(threadID, instr.a);
			ir::PTXB32 d = hydrazine::popc(a);
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

// Emulator-owned encoding of the opaque 64-bit mbarrier object and arrival token.
static constexpr ir::PTXU32 mbarrierCountMask = 0xfffff, mbarrierPhaseMask = 0x7fffff;
static constexpr unsigned mbarrierExpectedShift = 20, mbarrierPhaseShift = 40, mbarrierInitializedShift = 63;

void executive::CooperativeThreadArray::eval_MbarrierPendingCount(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int thread = 0; thread < threadCount; ++thread)
		if (context.predicated(thread, instr))
			setRegAsU32(thread, instr.d.reg, operandAsU64(thread, instr.a) & mbarrierCountMask);
}

void executive::CooperativeThreadArray::eval_Mbarrier(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	auto* shared = (unsigned char*)functionCallStack.sharedMemoryPointer();
	const ir::PTXU64 sharedSize = functionCallStack.sharedMemorySize();
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		const bool address32 = instr.a.bytes() == 4;
		ir::PTXU64 offset;
		if (address32) offset = operandAsU32(thread, instr.a);
		else offset = operandAsU64(thread, instr.a);
		if (instr.addressSpace == ir::PTXInstruction::Generic && instr.a.addressMode != ir::PTXOperand::Address) {
			const ir::PTXU64 sharedAddress = (ir::PTXU64)shared;
			offset -= sharedAddress;
			// A 32-bit generic pointer uses the low 32 bits of the host address.
			if (address32) offset = ir::PTXU32(offset);
		}
		if (offset % 8 || offset > sharedSize || sharedSize - offset < 8) {
			report("warning: mbarrier address is misaligned or outside shared memory; undefined behavior");
			continue;
		}
		ir::PTXU64 state = 0;
		switch (instr.opcode) {
		case ir::PTXInstruction::MbarrierInit: {
			const ir::PTXU32 count = operandAsU32(thread, instr.b);
			if (count == 0 || count > mbarrierCountMask) { report("warning: mbarrier.init count is outside 1..1048575; undefined behavior"); continue; }
			// Opaque encoding: pending[19:0], expected[39:20], phase[62:40], initialized[63].
			state = (ir::PTXU64(1) << mbarrierInitializedShift) | count | (ir::PTXU64(count) << mbarrierExpectedShift);
			break;
		}
		case ir::PTXInstruction::MbarrierInval:
			break; // Store zero to invalidate the object.
		case ir::PTXInstruction::MbarrierTestWait:
		case ir::PTXInstruction::MbarrierTestWaitParity: {
			std::memcpy(&state, shared + offset, sizeof(state));
			const bool parity = instr.opcode == ir::PTXInstruction::MbarrierTestWaitParity;
			const ir::PTXU64 phase = (state >> mbarrierPhaseShift) & mbarrierPhaseMask;
			const ir::PTXU64 requested = parity ? operandAsU32(thread,instr.b) : (operandAsU64(thread,instr.b) >> mbarrierPhaseShift) & mbarrierPhaseMask;
			if (!(state >> mbarrierInitializedShift) || (parity ? requested > 1 : ((phase - requested) & mbarrierPhaseMask) > 1)) {
				report("warning: mbarrier.test_wait has an invalid phase, parity or uninitialized object; undefined behavior");
				continue;
			}
			bool complete;
			if (parity) {
				const ir::PTXU32 currentParity = phase & 1;
				complete = currentParity != requested;
			} else {
				complete = phase != requested;
			}
			setRegAsPredicate(thread, instr.d.reg, complete);
			continue; // test_wait never changes the barrier or blocks the executing thread.
		}
		case ir::PTXInstruction::CpAsyncMbarrierArrive:
			std::memcpy(&state, shared + offset, sizeof(state));
			// Copies complete immediately: increment plus completion has zero net effect.
			if (!(state >> mbarrierInitializedShift) || (state & mbarrierCountMask) == mbarrierCountMask)
				report("warning: cp.async.mbarrier.arrive on invalid object or overflowing count; undefined behavior");
			continue;
		case ir::PTXInstruction::MbarrierArrive:
		case ir::PTXInstruction::MbarrierArriveDrop:
		case ir::PTXInstruction::CpAsyncMbarrierArriveNoInc: {
			std::memcpy(&state, shared + offset, sizeof(state));
			const ir::PTXU64 token = state;
			ir::PTXU32 pending = state & mbarrierCountMask;
			ir::PTXU32 expected = (state >> mbarrierExpectedShift) & mbarrierCountMask;
			const bool copyArrival = instr.opcode == ir::PTXInstruction::CpAsyncMbarrierArriveNoInc;
			const ir::PTXU32 count = copyArrival ? 1 : operandAsU32(thread, instr.b);
			const bool drop = instr.opcode == ir::PTXInstruction::MbarrierArriveDrop;
			if (!(state >> mbarrierInitializedShift) || count == 0 || count > pending || (drop && count > expected) || (instr.mbarrierNoComplete && count == pending)) {
				report("warning: invalid mbarrier arrival count, phase completion or uninitialized object; undefined behavior");
				continue;
			}
			pending -= count;
			if (drop) expected -= count;
			ir::PTXU64 phase = (state >> mbarrierPhaseShift) & mbarrierPhaseMask;
			if (pending == 0) { phase = (phase + 1) & mbarrierPhaseMask; pending = expected; }
			state = (ir::PTXU64(1) << mbarrierInitializedShift) | (phase << mbarrierPhaseShift) | (ir::PTXU64(expected) << mbarrierExpectedShift) | pending;
			if (!copyArrival && instr.d.addressMode != ir::PTXOperand::BitBucket) setRegAsB64(thread, instr.d.reg, token);
			break;
		}
		default: assertM(false, "Invalid mbarrier opcode");
		}
		std::memcpy(shared + offset, &state, sizeof(state));
	}
}

void executive::CooperativeThreadArray::eval_CpAsync(CTAContext &context,
	const ir::PTXInstruction &instr) {
	const std::string error = instr.valid();
	if (!error.empty()) throw RuntimeException(error, context.PC, instr);
	trace();
	auto* shared = (unsigned char*)functionCallStack.sharedMemoryPointer();
	const ir::PTXU64 sharedSize = functionCallStack.sharedMemorySize();
	const unsigned int size = instr.b.imm_uint;
	// Complete the copy immediately; commit/wait instructions have no pending work.
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		const ir::PTXU64 dst = instr.d.bytes() == 4 ? operandAsU32(thread, instr.d) : operandAsU64(thread, instr.d);
		const ir::PTXU64 src = instr.a.bytes() == 4 ? operandAsU32(thread, instr.a) : operandAsU64(thread, instr.a);
		unsigned int sourceSize = size;
		if (instr.c.type == ir::PTXOperand::pred) {
			const bool ignore = instr.c.condition == ir::PTXOperand::PT ||
				(instr.c.condition != ir::PTXOperand::nPT && operandAsPredicate(thread, instr.c));
			if (ignore) sourceSize = 0;
		} else if (instr.c.addressMode != ir::PTXOperand::Invalid) sourceSize = operandAsU32(thread, instr.c);
		if (sourceSize > size) {
			report("warning: cp.async src-size " << sourceSize << " exceeds cp-size " << size
				<< "; undefined behavior at PC " << context.PC << ", thread " << thread);
			continue; // Leave this destination unchanged instead of making an invalid host copy.
		}
		if (dst % size || (sourceSize && src % size) || dst > sharedSize || sharedSize - dst < size) {
			report("warning: cp.async misaligned address or out-of-bounds shared destination"
				<< "; undefined behavior at PC " << context.PC << ", thread " << thread);
			continue;
		}
		if (sourceSize) std::memcpy(shared + dst, (const void*)src, sourceSize);
		std::memset(shared + dst + sourceSize, 0, size - sourceSize);
	}
}

void executive::CooperativeThreadArray::eval_Prefetch(CTAContext &context, const ir::PTXInstruction &instr) {
	report(instr.toString());

	if (traceEvents) {
		currentEvent.memory_size = 4;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.d.addressMode) {
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.d.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.d.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand",
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();
}

void executive::CooperativeThreadArray::eval_Prefetchu(CTAContext &context, const ir::PTXInstruction &instr) {

	if (traceEvents) {
		currentEvent.memory_size = 4;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			const char *source = 0;

			switch (instr.d.addressMode) {
				case ir::PTXOperand::Indirect:
					source += getRegAsU64(threadID, instr.d.reg);
					break;
				case ir::PTXOperand::Address:
				case ir::PTXOperand::Immediate:
					source += instr.d.imm_uint;
					break;
				default:
					throw RuntimeException(
						"unsupported address mode for source operand",
						context.PC, instr);
			}

			source += instr.a.offset;
			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();
}

/*!

*/
void executive::CooperativeThreadArray::eval_Prmt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::b32) {
		switch( instr.permuteMode ) {
			case ir::PTXInstruction::DefaultPermute: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<hydrazine::DefaultPermute>(
						a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ForwardFourExtract: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<
						hydrazine::ForwardFourExtract>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::BackwardFourExtract: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<
						hydrazine::BackwardFourExtract>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ReplicateEight: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::ReplicateEight>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::EdgeClampLeft: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::EdgeClampLeft>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::EdgeClampRight: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = 0;
					d = hydrazine::permute<hydrazine::EdgeClampRight>(a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
			case ir::PTXInstruction::ReplicateSixteen: {
				for (int threadID = 0; threadID < threadCount; threadID++) {
					if (!context.predicated(threadID, instr)) continue;

					ir::PTXB32 a = operandAsB32(threadID, instr.a);
					ir::PTXB32 b = operandAsB32(threadID, instr.b);
					ir::PTXB32 c = operandAsB32(threadID, instr.c);
					ir::PTXB32 d = hydrazine::permute<hydrazine::ReplicateSixteen>(
						a, b, c);
					setRegAsB32(threadID, instr.d.reg, d);
				}
				break;
			}
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Rcp(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			// Without .ftz, .approx must still support subnormal inputs
			// (e.g. rcp.approx.f32(2^-127) ~= 2^127); only .ftz flushes
			// the input to signed zero, whose reciprocal is signed
			// infinity through the division below.
			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, roundedDiv(1.0f, a,
				instr.modifier & ir::PTXInstruction::approx
					? ir::PTXInstruction::rn : instr.modifier));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			if (instr.modifier & ir::PTXInstruction::approx) {
				const ir::PTXU64 high = 0xffffffff00000000ull;
				if (hydrazine::isnan(a)) {
					d = hydrazine::bit_cast<ir::PTXF64>(0x7fffffff00000000ull);
				} else {
					a = hydrazine::bit_cast<ir::PTXF64>(
						hydrazine::bit_cast<ir::PTXU64>(a) & high);
					if (issubnormal_(a)) a = hydrazine::copysign(0.0, a);
					d = roundedDiv(1.0, a, ir::PTXInstruction::rn);
					if (issubnormal_(d)) d = hydrazine::copysign(0.0, d);
					d = hydrazine::bit_cast<ir::PTXF64>(
						hydrazine::bit_cast<ir::PTXU64>(d) & high);
				}
			} else {
				d = roundedDiv(1.0, a, instr.modifier);
			}
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Red(CTAContext &context, const ir::PTXInstruction &instr) {
	ir::PTXInstruction::AtomicOperation operation;
	switch (instr.reductionOperation) {
		case ir::PTXInstruction::ReductionAnd: operation = ir::PTXInstruction::AtomicAnd; break;
		case ir::PTXInstruction::ReductionXor: operation = ir::PTXInstruction::AtomicXor; break;
		case ir::PTXInstruction::ReductionOr: operation = ir::PTXInstruction::AtomicOr; break;
		case ir::PTXInstruction::ReductionAdd: operation = ir::PTXInstruction::AtomicAdd; break;
		case ir::PTXInstruction::ReductionInc: operation = ir::PTXInstruction::AtomicInc; break;
		case ir::PTXInstruction::ReductionDec: operation = ir::PTXInstruction::AtomicDec; break;
		case ir::PTXInstruction::ReductionMin: operation = ir::PTXInstruction::AtomicMin; break;
		case ir::PTXInstruction::ReductionMax: operation = ir::PTXInstruction::AtomicMax; break;
		default:
			throw RuntimeException("unsupported reduction operation",
				context.PC, instr);
	}
	evalAtomicRMW(context, instr, operation, false);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Rem(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = CTARemainder(a, b);
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Ret(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	if (functionCallStack.isTheCurrentFrameMain()) {
		eval_Exit(context, instr);
		return;
	}

	int returnedPC = functionCallStack.returnPC();

	reportE(REPORT_RET, "Returned from function call at PC "
		<< returnedPC );
	const ir::PTXInstruction& call = kernel->instructions[returnedPC];
	reportE(REPORT_RET, " Previous stack size ("
		<< functionCallStack.callerFrameSize() );
	unsigned int offset = 0;
	for (ir::PTXOperand::Array::const_iterator argument = call.d.array.begin();
		argument != call.d.array.end(); ++argument) {
		for (int threadID = 0; threadID != threadCount; ++threadID) {
			if (!context.predicated(threadID, instr)) continue;

			char* callerPointer =
				(char*)functionCallStack.callerFramePointer(threadID);
			char* pointer =
				(char*)functionCallStack.stackFramePointer(threadID);

			reportE(REPORT_RET, " For thread " << threadID << " copying "
				<< argument->toString() << " ["
				<< ir::PTXOperand::bytes(argument->type)
				<< " bytes] from new frame at "
				<< offset << " to caller frame at "
				<< argument->offset
				<< " - destination: "
				<< (void *)(callerPointer + argument->offset)
				<< " - caller stack size: "
					<< functionCallStack.callerFrameSize());

			std::memcpy(callerPointer + argument->offset, pointer + offset,
				ir::PTXOperand::bytes(argument->type));

		}
		offset += ir::PTXOperand::bytes(argument->type);
	}

	functionCallStack.popFrame();
	reconvergenceMechanism->pop();

	// if we returned all the way back, copy operands into registers
	CTAContext &returnedContext = getActiveContext();

	if (returnedContext.PC == returnedPC + 1) {
		for (ir::PTXOperand::Array::const_iterator
			argument = call.d.array.begin();
			argument != call.d.array.end(); ++argument) {
			if (argument->addressMode != ir::PTXOperand::Register) continue;

			for (int threadID = 0; threadID != threadCount; ++threadID) {
				if (!returnedContext.predicated(threadID, instr)) continue;

				char* frame =
					(char*)functionCallStack.stackFramePointer(threadID);

				ir::PTXU64 data = 0;

				std::memcpy(&data, frame + argument->offset,
					ir::PTXOperand::bytes(argument->type));

				setRegAsU64(threadID, argument->reg, data);
			}
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Rsqrt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, roundedDiv(1.0f,
				roundedSqrt(a, ir::PTXInstruction::rn), ir::PTXInstruction::rn));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			if (instr.modifier & ir::PTXInstruction::ftz) {
				const ir::PTXU64 high = 0xffffffff00000000ull;
				if (hydrazine::isnan(a)) {
					d = hydrazine::bit_cast<ir::PTXF64>(0x7fffffff00000000ull);
				} else {
					a = hydrazine::bit_cast<ir::PTXF64>(
						hydrazine::bit_cast<ir::PTXU64>(a) & high);
					if (issubnormal_(a)) a = hydrazine::copysign(0.0, a);
					d = roundedDiv(1.0,
						roundedSqrt(a, ir::PTXInstruction::rn),
						ir::PTXInstruction::rn);
					if (issubnormal_(d)) d = hydrazine::copysign(0.0, d);
					d = hydrazine::bit_cast<ir::PTXF64>(
						hydrazine::bit_cast<ir::PTXU64>(d) & high);
				}
			} else {
				d = roundedDiv(1.0, roundedSqrt(a,
					ir::PTXInstruction::rn), ir::PTXInstruction::rn);
			}
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	sad.type d, a, b, c;

	d = c + ((a<b) ? b-a : a-b);

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };

*/
void executive::CooperativeThreadArray::eval_Sad(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {
	case ir::PTXOperand::u16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b),
				c = operandAsU16(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU16(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b),
				c = operandAsU32(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::u64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b),
				c = operandAsU64(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsU64(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s16:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b),
				c = operandAsS16(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS16(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b),
				c = operandAsS32(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS32(threadID, instr.d.reg, d);
		}
	} break;
	case ir::PTXOperand::s64:
	{
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b),
				c = operandAsS64(threadID, instr.c);

			d = c + ((a < b) ? b-a : a-b);

			setRegAsS64(threadID, instr.d.reg, d);
		}
	} break;
	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!
	Select between source operands according to the predicate value
selp.type d, a, b, c;

.type = { .b16, .b32, .b64,
          .u16, .u32, .u64,
          .s16, .s32, .s64,
                .f32, .f64 };

*/
void executive::CooperativeThreadArray::eval_SelP(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	switch (instr.type) {
		case ir::PTXOperand::b16:	// fall through
		case ir::PTXOperand::s16:	// fall through
		case ir::PTXOperand::u16:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXU16 a = operandAsU16(threadID, instr.a),
					b = operandAsU16(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);

				ir::PTXU16 d = (c ? a : b);

				setRegAsU16(threadID, instr.d.reg, d);
			}
		}
			break;

		case ir::PTXOperand::s32:	// fall through
		case ir::PTXOperand::b32:	// fall through
		case ir::PTXOperand::u32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXU32 a = operandAsU32(threadID, instr.a),
					b = operandAsU32(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);

				ir::PTXU32 d = (c ? a : b);

				setRegAsU32(threadID, instr.d.reg, d);
			}
		}
		break;

		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXF32 a = operandAsF32(threadID, instr.a),
					b = operandAsF32(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);

				ir::PTXF32 d = (c ? a : b);

				setRegAsF32(threadID, instr.d.reg, d);
			}
		}
		break;

		case ir::PTXOperand::f64:	// fall through
		case ir::PTXOperand::s64:	// fall through
		case ir::PTXOperand::b64:	// fall through
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				bool c = operandAsPredicate(threadID, instr.c);
				ir::PTXU64 d = (c ? a : b);

				reportE( REPORT_SELP, "  thread[" << threadID << "] d = 0x"
					<< std::hex << d << ", " << instr.d.identifier << " = "
					<< getRegAsU64(threadID, instr.d.reg) << std::dec);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	Compare two numeric values with a relational operator
	and [optionally] combine this result
	with a predicate value by applying a boolean operator

setp.CmpOp.type p[|q], a, b;
setp.CmpOp.BoolOp.type p[|q], a, b, [!]c;

.type = { .b16, .b32, .b64,
	.u16, .u32, .u64,
	.s16, .s32, .s64,
	.f32, .f64 };

*/
void executive::CooperativeThreadArray::eval_SetP(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	switch (instr.type) {

		// unsigned int types [extended to 64-bit uint]
		case ir::PTXOperand::pred:
		case ir::PTXOperand::b16:
		case ir::PTXOperand::b32:
		case ir::PTXOperand::b64:
		case ir::PTXOperand::u16:
		case ir::PTXOperand::u32:
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				bool c = true;
				bool t = false;

				ir::PTXU64 a, b;

				switch (instr.type) {
					case ir::PTXOperand::pred:
						a = (ir::PTXU64)operandAsPredicate(threadID, instr.a) & 0x1;
						b = (ir::PTXU64)operandAsPredicate(threadID, instr.b) & 0x1;
						break;
					case ir::PTXOperand::s16:
					case ir::PTXOperand::b16:
					case ir::PTXOperand::u16:
						a = (ir::PTXU64)operandAsU16(threadID, instr.a) & 0x0ffff;
						b = (ir::PTXU64)operandAsU16(threadID, instr.b) & 0x0ffff;
						break;
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::b32:
						a = (ir::PTXU64)operandAsU32(
							threadID, instr.a) & 0x0ffffffff;
						b = (ir::PTXU64)operandAsU32(
							threadID, instr.b) & 0x0ffffffff;
						break;
					default:
						a = operandAsU64(threadID, instr.a);
						b = operandAsU64(threadID, instr.b);
						break;
				}

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;

					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					default:
						throw RuntimeException(
							"invalid comparison operator for unsigned int type",
							context.PC, instr);
				}

				// now apply the bool op
				bool p = false, q = false;

				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t  || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t  && !c) || (!t && c);
						q = (!t && !c) || (t  && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a
					<< ", " << instr.b.identifier << " = " << b
					<< " condition = " << t << ", input = " << c << " "
					<< instr.d.identifier << " = " << p << ", q = " << q );

				if (instr.d.addressMode != ir::PTXOperand::BitBucket)
					setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid
					&& instr.pq.addressMode != ir::PTXOperand::BitBucket) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;

		// signed int types [extended to 64-bit int]
		case ir::PTXOperand::s16:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::s64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				bool c = true;	// read operator somehow
				bool t = false;

				ir::PTXS64 a, b;

				switch (instr.type) {
					case ir::PTXOperand::s16:
						a = (ir::PTXS64)operandAsS16(threadID, instr.a);
						b = (ir::PTXS64)operandAsS16(threadID, instr.b);
						break;
					case ir::PTXOperand::s32:
						a = (ir::PTXS64)operandAsS32(threadID, instr.a);
						b = (ir::PTXS64)operandAsS32(threadID, instr.b);
						break;
					default:
						a = operandAsS64(threadID, instr.a);
						b = operandAsS64(threadID, instr.b);
						break;
				}

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will
				// get this wrong twice or less
				switch (instr.comparisonOperator) {

					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;

					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;

					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;

					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;

					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}

				// now apply the bool op
				bool p = false, q = false;
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t && !c) || (!t && c);
						q = (!t && !c) || (t && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a
					<< ", " << instr.b.identifier << " = " << b
					<< " condition = " << t << ", input = " << c << " "
					<< instr.d.identifier << " = " << p << ", q = " << q );

				if (instr.d.addressMode != ir::PTXOperand::BitBucket)
					setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid
					&& instr.pq.addressMode != ir::PTXOperand::BitBucket) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;

		// floating-point types [widened to double after type-specific FTZ]
		case ir::PTXOperand::f16:
		case ir::PTXOperand::f32:
		case ir::PTXOperand::f64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXF64 a;
				ir::PTXF64 b;

				switch (instr.type) {
					case ir::PTXOperand::f16:
						a = f16ToF32(ftzF16(instr.modifier, operandAsU16(threadID, instr.a)));
						b = f16ToF32(ftzF16(instr.modifier, operandAsU16(threadID, instr.b)));
						break;
					case ir::PTXOperand::f32:
						a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
						b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
						break;
					default:
						a = operandAsF64(threadID, instr.a);
						b = operandAsF64(threadID, instr.b);
						break;
				}

				bool c = true;	// read operator somehow
				bool t = false;

				const bool hasNaN = hydrazine::isnan(a) || hydrazine::isnan(b);

				const bool unorderedOp =
						instr.comparisonOperator == ir::PTXInstruction::Equ ||
						instr.comparisonOperator == ir::PTXInstruction::Neu ||
						instr.comparisonOperator == ir::PTXInstruction::Ltu ||
						instr.comparisonOperator == ir::PTXInstruction::Leu ||
						instr.comparisonOperator == ir::PTXInstruction::Gtu ||
						instr.comparisonOperator == ir::PTXInstruction::Geu;

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = hasNaN ? unorderedOp :  (a == b);
						break;
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = hasNaN ? unorderedOp :  (a != b);
						break;

					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = hasNaN ? unorderedOp :  (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = hasNaN ? unorderedOp :  (a <= b);
						break;

					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = hasNaN ? unorderedOp :  (a > b);
						break;

					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = hasNaN ? unorderedOp :  (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					case ir::PTXInstruction::Nan:
						t = hydrazine::isnan(a) || hydrazine::isnan(b);
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}

				// now apply the bool op
				bool p = false, q = false;
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						p = (t && c);
						q = (!t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						p = (t || c);
						q = (!t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						p = (t && !c) || (!t && c);
						q = (!t && !c) || (t && c);
						break;
					default:
						p = t;
						q = !t;
						break;
				}

				reportE(REPORT_SETP, "    " << instr.a.identifier << " = " << a
					<< ", " << instr.b.identifier << " = " << b
					<< " condition = " << t << ", input = " << c << " "
					<< instr.d.identifier << " = " << p << ", q = " << q );

				if (instr.d.addressMode != ir::PTXOperand::BitBucket)
					setRegAsPredicate(threadID, instr.d.reg, p);
				if (instr.pq.addressMode != ir::PTXOperand::Invalid
					&& instr.pq.addressMode != ir::PTXOperand::BitBucket) {
					setRegAsPredicate(threadID, instr.pq.reg, q);
				}
			}
		}
		break;

		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	ir::PTX set instruction

	t = (a CmpOp b) ? 1 : 0;
	if (isFloat(dtype))
		d = BoolOp(t, c) ? 1.0f : 0x00000000;
	else
		d = BoolOp(t, c) ? 0xFFFFFFFF : 0x00000000;

*/
void executive::CooperativeThreadArray::eval_Set(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	switch (instr.a.type) {

		// unsigned int types [extended to 64-bit uint]
		case ir::PTXOperand::b16:
		case ir::PTXOperand::b32:
		case ir::PTXOperand::b64:
		case ir::PTXOperand::u16:
		case ir::PTXOperand::u32:
		case ir::PTXOperand::u64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				bool c = true;	// read predicate somehow
				bool t = false;

				ir::PTXU64 a, b;

				switch (instr.a.type) {
					case ir::PTXOperand::u16:
					case ir::PTXOperand::b16:
					case ir::PTXOperand::s16:
						a = (ir::PTXU64)operandAsU16(threadID, instr.a);
						b = (ir::PTXU64)operandAsU16(threadID, instr.b);
						break;
					case ir::PTXOperand::b32:
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						a = (ir::PTXU64)operandAsU32(threadID, instr.a);
						b = (ir::PTXU64)operandAsU32(threadID, instr.b);
						break;
					default:
						a = operandAsU64(threadID, instr.a);
						b = operandAsU64(threadID, instr.b);
						break;
				}

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;

					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;
					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}

				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;

		// signed int types [extended to 64-bit int]
		case ir::PTXOperand::s16:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::s64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				bool c = true;	// read operator somehow
				bool t = false;

				ir::PTXS64 a, b;

				switch (instr.a.type) {
					case ir::PTXOperand::s16:
					case ir::PTXOperand::u16:
					case ir::PTXOperand::b16:
						a = (ir::PTXS64)operandAsS16(threadID, instr.a);
						b = (ir::PTXS64)operandAsS16(threadID, instr.b);
						break;
					case ir::PTXOperand::b32:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::s32:
						a = (ir::PTXS64)operandAsS32(threadID, instr.a);
						b = (ir::PTXS64)operandAsS32(threadID, instr.b);
						break;
					default:
						a = operandAsS64(threadID, instr.a);
						b = operandAsS64(threadID, instr.b);
						break;
				}

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {

					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;

					case ir::PTXInstruction::Ne:
						t = (a != b);
						break;

					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;

					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;

					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					default:
						throw RuntimeException("invalid comparison operator "
						"for unsigned int type", context.PC, instr);
				}

				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;

		// floating-point types, with f16 widened before comparison
		case ir::PTXOperand::f16:
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXF32 a, b;
				if (instr.a.type == ir::PTXOperand::f16) {
					a = f16ToF32(ftzF16(instr.modifier,
						operandAsU16(threadID, instr.a)));
					b = f16ToF32(ftzF16(instr.modifier,
						operandAsU16(threadID, instr.b)));
				} else {
					a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
					b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
				}
				bool c = true;	// read operator somehow
				bool t = false;

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ne:
						t = !hydrazine::isnan(b) && !hydrazine::isnan(a) && (a != b);
						break;

					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;

					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;

					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;

					case ir::PTXInstruction::Nan:
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}

				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Nan:
						// if either is NaN, set t to true
						t = (hydrazine::isnan(a) || hydrazine::isnan(b) || t);
						break;
					default:
						break;
				}

				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

					switch (instr.type) {
						case ir::PTXOperand::s32:
						case ir::PTXOperand::u32:
							setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
							break;
						case ir::PTXOperand::f16:
							setRegAsB16(threadID, instr.d.reg, t ? 0x3c00 : 0x0000);
							break;
						case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;

		// double-precision float
		case ir::PTXOperand::f64:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;

				ir::PTXF64 a = operandAsF64(threadID, instr.a),
					b = operandAsF64(threadID, instr.b);
				bool c = true;
				bool t = false;

				if (instr.c.addressMode == ir::PTXOperand::Register) {
					c = operandAsPredicate(threadID, instr.c);
				}

				// any branch predictor worth its salt will get this wrong twice or less
				switch (instr.comparisonOperator) {

					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Eq:
						t = (a == b);
						break;

					case ir::PTXInstruction::Neu:
						t = (a != b);
						break;
					case ir::PTXInstruction::Ne:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b)
							&& (a != b);
						break;

					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Lo:	// fall through
					case ir::PTXInstruction::Lt:
						t = (a < b);
						break;

					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Ls:	// fall through
					case ir::PTXInstruction::Le:
						t = (a <= b);
						break;

					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Hi:	// fall through
					case ir::PTXInstruction::Gt:
						t = (a > b);
						break;

					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Hs:	// fall through
					case ir::PTXInstruction::Ge:
						t = (a >= b);
						break;

					case ir::PTXInstruction::Num:
						t = !hydrazine::isnan(a) && !hydrazine::isnan(b);
						break;
					case ir::PTXInstruction::Nan:
						t = hydrazine::isnan(a) || hydrazine::isnan(b);
						break;

					default:
						throw RuntimeException("invalid comparison operator "
							"for unsigned int type", context.PC, instr);
				}

				switch (instr.comparisonOperator) {
					case ir::PTXInstruction::Equ:
					case ir::PTXInstruction::Neu:
					case ir::PTXInstruction::Ltu:
					case ir::PTXInstruction::Leu:
					case ir::PTXInstruction::Gtu:
					case ir::PTXInstruction::Geu:
					case ir::PTXInstruction::Nan:
						// if either is NaN, set t to true
						t = (hydrazine::isnan(a) || hydrazine::isnan(b) || t);
						break;
					default:
						break;
				}


				// now apply the bool op
				switch (instr.booleanOperator) {
					case ir::PTXInstruction::BoolAnd:
						t = (t && c);
						break;
					case ir::PTXInstruction::BoolOr:
						t = (t || c);
						break;
					case ir::PTXInstruction::BoolXor:
						t = (t && !c) || (!t && c);
						break;
					default:
						break;
				}

				switch (instr.type) {
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
						setRegAsU32(threadID, instr.d.reg, (t ? 0xFFFFFFFF : 0x00));
						break;
					case ir::PTXOperand::f32:
						setRegAsF32(threadID, instr.d.reg, (t ? 1.0f : 0.0f));
						break;
					default:
						break;
				}
			}
		}
		break;

		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_Shf(CTAContext &context, const ir::PTXInstruction &instr) {
    trace();
    if (instr.type != ir::PTXOperand::b32) {
        throw RuntimeException("unsupported data type", context.PC, instr);
    }

    for (int threadID = 0; threadID < threadCount; threadID++) {
        if (!context.predicated(threadID, instr)) continue;

        ir::PTXB32 d, a, b;
        unsigned int n;

        if (instr.a.addressMode == ir::PTXOperand::Immediate) {
            a = instr.a.imm_uint;
        } else if (instr.a.addressMode == ir::PTXOperand::Register) {
            a = getRegAsB32(threadID, instr.a.reg);
        } else {
            throw RuntimeException("unsupported address mode for a", context.PC, instr);
        }

        if (instr.b.addressMode == ir::PTXOperand::Immediate) {
            b = instr.b.imm_uint;
        } else if (instr.b.addressMode == ir::PTXOperand::Register) {
            b = getRegAsB32(threadID, instr.b.reg);
        } else {
            throw RuntimeException("unsupported address mode for b", context.PC, instr);
        }

        if (instr.c.addressMode == ir::PTXOperand::Immediate) {
            n = instr.c.imm_uint;
        } else if (instr.c.addressMode == ir::PTXOperand::Register) {
            n = getRegAsU32(threadID, instr.c.reg);
        } else {
            throw RuntimeException("unsupported address mode for c", context.PC, instr);
        }

        if (instr.shiftMode == ir::PTXInstruction::ShiftMode::Wrap) {
            n &= 31;  // Wrap to lower 5 bits
        } else if (instr.shiftMode == ir::PTXInstruction::ShiftMode::Clamp) {
            n = min(n, 32U);  // Clamp to 32
        } else {
            throw RuntimeException("unsupported shift mode", context.PC, instr);
        }

        const ir::PTXB64 pair = (static_cast<ir::PTXB64>(b) << 32) | a;
        if (instr.shiftDirection == ir::PTXInstruction::ShiftLeft) {
            d = (pair << n) >> 32;
        } else if (instr.shiftDirection == ir::PTXInstruction::ShiftRight) {
            d = pair >> n;
        } else {
            throw RuntimeException("unsupported shift direction", context.PC, instr);
        }

        setRegAsB32(threadID, instr.d.reg, d);
    }
}

/*!

*/
void executive::CooperativeThreadArray::eval_Shl(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	unsigned int b = 0;
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB16 d, a;

			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 16 ? 0 : a << b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 32 ? 0 : a << b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 64 ? 0 : a << b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Shr(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	unsigned int b = 0;
	if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB16 d, a;

			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 16 ? 0 : a >> b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 32 ? 0 : a >> b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsB64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 64 ? 0 : a >> b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 16 ? (a < 0 ? -1 : 0) : a >> b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 32 ? (a < 0 ? -1 : 0) : a >> b;
			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_int;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsS64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 64 ? (a < 0 ? -1 : 0) : a >> b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU16(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 16 ? 0 : a >> b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU32(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 32 ? 0 : a >> b;
			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a;
			if (instr.a.addressMode == ir::PTXOperand::Immediate) {
				a = instr.a.imm_uint;
			}
			else if (instr.a.addressMode == ir::PTXOperand::Register) {
				a = getRegAsU64(threadID, instr.a.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			if (instr.b.addressMode == ir::PTXOperand::Immediate) {
				b = instr.b.imm_int;
			}
			else if (instr.b.addressMode == ir::PTXOperand::Register) {
				b = getRegAsU32(threadID, instr.b.reg);
			}
			else {
				throw RuntimeException("unsupported data type",
					context.PC, instr);
			}
			d = b >= 64 ? 0 : a >> b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Sin(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d,
				a = ftz(instr.modifier, operandAsF32(threadID, instr.a));
			d = ftz(instr.modifier, (ir::PTXF32)sin(a));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = sin(a);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!
	slct.dtype.ctype d, a, b, c;

	.dtype = { .b16, .b32, .b64,
		         .u16, .u32, .u64,
		         .s16, .s32, .s64,
		               .f32, .f64 };

	.ctype = { .s32, .f32 };

	d = (c >= 0) ? a : b;
*/
void executive::CooperativeThreadArray::eval_SlCt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	assert(instr.opcode == ir::PTXInstruction::SlCt);

	switch (instr.c.type) {
		case ir::PTXOperand::u32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXS32 c = operandAsU32(threadID, instr.c);

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		case ir::PTXOperand::s32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXS32 c = operandAsS32(threadID, instr.c);

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		case ir::PTXOperand::f32:
		{
			for (int threadID = 0; threadID < threadCount; threadID++) {
				if (!context.predicated(threadID, instr)) continue;
				ir::PTXU64 a = operandAsU64(threadID, instr.a),
					b = operandAsU64(threadID, instr.b);
				ir::PTXF32 c = ftz(instr.modifier, operandAsF32(threadID, instr.c));

				ir::PTXU64 d = ((c >= 0) ? a : b);

				setRegAsU64(threadID, instr.d.reg, d);
			}
		}
		break;
		default:
			throw RuntimeException("source data type not permitted",
				context.PC, instr);
			break;
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Sqrt(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	assert(instr.opcode == ir::PTXInstruction::Sqrt);

	if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a));

			d = roundedSqrt(a, instr.modifier & ir::PTXInstruction::approx
				? ir::PTXInstruction::rn : instr.modifier);

			setRegAsF32(threadID, instr.d.reg, ftz(instr.modifier, d));
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a);
			d = roundedSqrt(a, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

////////////////////////////////////////////////////////////////////////////////

void executive::CooperativeThreadArray::normalStore(int threadID,
	const ir::PTXInstruction &instr, char* source) {
	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				ir::PTXU8 word = operandAsU8(threadID, instr.a);
#if REPORT_ST
				report( "    Storing " << word << " to "
					<< (int*)source );
#endif
				*((ir::PTXU8*)source) = word;
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				ir::PTXU16 word = operandAsU16(threadID, instr.a);
#if REPORT_ST
				report( "    Storing " << word << " to "
					<< (int*)source );
#endif
				*((ir::PTXU16*)source) = word;
			}
			break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = operandAsF32(threadID, instr.a);
#if REPORT_ST
				report( "    Storing " << word << " to "
					<< (int*)source );
#endif
				*((ir::PTXF32*)source) = word;
			}
			break;
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				ir::PTXU32 word = operandAsU32(threadID, instr.a);
#if REPORT_ST
				report( "    Storing " << word << " to "
					<< (int*)source );
#endif
				*((ir::PTXU32*)source) = word;
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				ir::PTXU64 word = operandAsU64(threadID, instr.a);
#if REPORT_ST
				report( "    Storing " << (void*)word << " to "
					<< (int*)source );
#endif
				*((ir::PTXU64*)source) = word;
			}
			break;
		default:
			break;
	}

}

void executive::CooperativeThreadArray::vectorStore(int threadID,
	const ir::PTXInstruction &instr, char* source, unsigned int elementSize) {
	for (ir::PTXOperand::Array::const_iterator i = instr.a.array.begin();
		i != instr.a.array.end(); ++i, source += elementSize) {
		switch (instr.type) {
			case ir::PTXOperand::b8:		// fall through
			case ir::PTXOperand::s8:		// fall through
			case ir::PTXOperand::u8:
				{
					ir::PTXU8 word = operandAsU8(threadID, *i);
#if REPORT_ST
					report( "    Storing " << word << " to "
						<< (int*)source );
#endif
					*((ir::PTXU8*)source) = word;
				}
				break;
			case ir::PTXOperand::b16:		// fall through
			case ir::PTXOperand::s16:		// fall through
			case ir::PTXOperand::u16:
				{
					ir::PTXU16 word = operandAsU16(threadID, *i);
#if REPORT_ST
					report( "    Storing " << word << " to "
						<< (int*)source );
#endif
					*((ir::PTXU16*)source) = word;
				}
				break;
		case ir::PTXOperand::f32:
			{
				ir::PTXF32 word = operandAsF32(threadID, *i);
#if REPORT_ST
				report( "    Storing " << word << " to "
					<< (int*)source << " - from register " << i->reg );
#endif
				*((ir::PTXF32*)source) = word;
			}
			break;
			case ir::PTXOperand::b32:		// fall through
			case ir::PTXOperand::s32:		// fall through
			case ir::PTXOperand::u32:
				{
					ir::PTXU32 word = operandAsU32(threadID, *i);
#if REPORT_ST
					report( "    Storing " << word << " to "
						<< (int*)source );
#endif
					*((ir::PTXU32*)source) = word;
				}
				break;
			case ir::PTXOperand::f64:		// fall through
			case ir::PTXOperand::b64:		// fall through
			case ir::PTXOperand::s64:		// fall through
			case ir::PTXOperand::u64:
				{
					ir::PTXU64 word = operandAsU64(threadID, *i);
#if REPORT_ST
					report( "    Storing " << (void*)word << " to "
						<< (int*)source );
#endif
					*((ir::PTXU64*)source) = word;
				}
				break;
			default:
				break;
		}
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_St(CTAContext &context,
	const ir::PTXInstruction &instr) {
	size_t elementSize = 0;

	switch (instr.type) {
		case ir::PTXOperand::b8:		// fall through
		case ir::PTXOperand::s8:		// fall through
		case ir::PTXOperand::u8:
			{
				elementSize = sizeof(ir::PTXU8);
			}
			break;
		case ir::PTXOperand::b16:		// fall through
		case ir::PTXOperand::s16:		// fall through
		case ir::PTXOperand::u16:
			{
				elementSize = sizeof(ir::PTXU16);
			}
			break;
		case ir::PTXOperand::f32:		// fall through
		case ir::PTXOperand::b32:		// fall through
		case ir::PTXOperand::s32:		// fall through
		case ir::PTXOperand::u32:
			{
				elementSize = sizeof(ir::PTXU32);
			}
			break;
		case ir::PTXOperand::f64:		// fall through
		case ir::PTXOperand::b64:		// fall through
		case ir::PTXOperand::s64:		// fall through
		case ir::PTXOperand::u64:
			{
				elementSize = sizeof(ir::PTXU64);
			}
			break;
		default:
			throw RuntimeException("unsupported data type", context.PC, instr);
	}

	if (traceEvents) {
		currentEvent.memory_size = elementSize;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) {
				continue;
			}

			char *source = 0;

			switch (instr.d.addressMode) {
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
				switch (instr.d.type) {
					case ir::PTXOperand::b8:		// fall through
					case ir::PTXOperand::s8:		// fall through
					case ir::PTXOperand::u8:
						{
							source += getRegAsU8(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::b16:		// fall through
					case ir::PTXOperand::s16:		// fall through
					case ir::PTXOperand::u16:
						{
							source += getRegAsU16(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::f32:		// fall through
					case ir::PTXOperand::b32:		// fall through
					case ir::PTXOperand::s32:		// fall through
					case ir::PTXOperand::u32:
						{
							source += getRegAsU32(threadID, instr.d.reg );
						}
						break;
					case ir::PTXOperand::f64:		// fall through
					case ir::PTXOperand::b64:		// fall through
					case ir::PTXOperand::s64:		// fall through
					case ir::PTXOperand::u64:
						{
							source += getRegAsU64(threadID, instr.d.reg );
						}
						break;
					default:
						break;
				}
				break;
			case ir::PTXOperand::Address:
			case ir::PTXOperand::Immediate:
				source += instr.d.imm_uint;
				break;
			default:
				throw RuntimeException(
					"unsupported address mode for source operand",
						context.PC, instr);
			}

			source += instr.d.offset;

			currentEvent.memory_addresses.push_back((ir::PTXU64)source);
		}
	}

	trace();

	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) {
			continue;
		}

		char *source = 0;

		switch (instr.d.addressMode) {
		case ir::PTXOperand::Register:
		case ir::PTXOperand::Indirect:
			switch (instr.d.type) {
				case ir::PTXOperand::b8:		// fall through
				case ir::PTXOperand::s8:		// fall through
				case ir::PTXOperand::u8:
					{
						source += getRegAsU8(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::b16:		// fall through
				case ir::PTXOperand::s16:		// fall through
				case ir::PTXOperand::u16:
					{
						source += getRegAsU16(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::f32:		// fall through
				case ir::PTXOperand::b32:		// fall through
				case ir::PTXOperand::s32:		// fall through
				case ir::PTXOperand::u32:
					{
						source += getRegAsU32(threadID, instr.d.reg );
					}
					break;
				case ir::PTXOperand::f64:		// fall through
				case ir::PTXOperand::b64:		// fall through
				case ir::PTXOperand::s64:		// fall through
				case ir::PTXOperand::u64:
					{
						source += getRegAsU64(threadID, instr.d.reg );
					}
					break;
				default:
					break;
			}
			break;
		case ir::PTXOperand::Address:
		case ir::PTXOperand::Immediate:
			source += instr.d.imm_uint;
			break;
		default:
			throw RuntimeException(
				"unsupported address mode for source operand",
					context.PC, instr);
		}

		source += instr.d.offset;

		switch (instr.addressSpace) {
			case ir::PTXInstruction::Param:
				{
					if (instr.d.isArgument) {
						source += kernel->scheduler->argumentMemory();
					}
					else {
						source += (ir::PTXU64)
							functionCallStack.stackFramePointer(threadID);
					}
				}
				break;
				case ir::PTXInstruction::Generic: /* fall through */
				case ir::PTXInstruction::Global:
				{
				}
				break;
			case ir::PTXInstruction::Shared:
				{
					source += (ir::PTXU64) functionCallStack.sharedMemoryPointer();
				}
				break;
			case ir::PTXInstruction::Local:
				{
					if (instr.d.addressMode == ir::PTXOperand::Address &&
						instr.d.isGlobalLocal) {
						source += (ir::PTXU64)
							functionCallStack.globalLocalMemoryPointer(
							threadID);
					}
					else {
						source += (ir::PTXU64)
							functionCallStack.localMemoryPointer(threadID);
					}
				}
				break;
			default:
				throw RuntimeException("unsupported address space",
					context.PC, instr);
		}

		if (instr.a.vec == ir::PTXOperand::v1) {
			normalStore(threadID, instr, source);
		}
		else {
			vectorStore(threadID, instr, source, elementSize);
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

/*!

*/
void executive::CooperativeThreadArray::eval_Sub(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::f16x2) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 a = operandAsU32(threadID, instr.a);
			ir::PTXU32 b = operandAsU32(threadID, instr.b);
			ir::PTXU16 al = static_cast<ir::PTXU16>(a), ah = a >> 16;
			ir::PTXU16 bl = static_cast<ir::PTXU16>(b), bh = b >> 16;
			ir::PTXU16 dl = ftzF16(effectiveModifier, toF16(
				roundedSub(f16ToF32(ftzF16(effectiveModifier, al)),
					f16ToF32(ftzF16(effectiveModifier, bl)), effectiveModifier),
				effectiveModifier));
			ir::PTXU16 dh = ftzF16(effectiveModifier, toF16(
				roundedSub(f16ToF32(ftzF16(effectiveModifier, ah)),
					f16ToF32(ftzF16(effectiveModifier, bh)), effectiveModifier),
					effectiveModifier));
			setRegAsU32(threadID, instr.d.reg, static_cast<ir::PTXU32>(dl) |
				(static_cast<ir::PTXU32>(dh) << 16));
		}
	}
	else if (instr.type == ir::PTXOperand::f16) {
		const int effectiveModifier = instr.modifier | ir::PTXInstruction::rn;
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 a = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.a)));
			ir::PTXF32 b = f16ToF32(ftzF16(effectiveModifier,
				operandAsU16(threadID, instr.b)));
			ir::PTXU16 d = toF16(roundedSub(a, b, effectiveModifier), effectiveModifier);
			setRegAsB16(threadID, instr.d.reg, ftzF16(effectiveModifier, d));
		}
	}
	else if (instr.type == ir::PTXOperand::f32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF32 d, a = ftz(instr.modifier, operandAsF32(threadID, instr.a)),
				b = ftz(instr.modifier, operandAsF32(threadID, instr.b));
			d = ftz(instr.modifier, sat(instr.modifier,
				roundedSub(a, b, instr.modifier)));
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::f64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXF64 d, a = operandAsF64(threadID, instr.a),
				b = operandAsF64(threadID, instr.b);
			d = roundedSub(a, b, instr.modifier);
			setRegAsF64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS16 d, a = operandAsS16(threadID, instr.a),
				b = operandAsS16(threadID, instr.b);
			d = a - b;
			setRegAsS16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS32 d, a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);

			if (instr.modifier & ir::PTXInstruction::sat) {
				ir::PTXS64 la = a;
				ir::PTXS64 lb = b;

				ir::PTXS64 ld = la - lb;

				ld = max((ir::PTXS64)INT_MIN, ld);
				ld = min((ir::PTXS64)INT_MAX, ld);

				d = ld;
			}
			else if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXS32 carry = 0;

				hydrazine::add(d, carry, a, -b, carry);
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a - b;
			}

			setRegAsS32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::s64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXS64 d, a = operandAsS64(threadID, instr.a),
				b = operandAsS64(threadID, instr.b);
			d = a - b;
			setRegAsS64(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU16 d, a = operandAsU16(threadID, instr.a),
				b = operandAsU16(threadID, instr.b);
			d = a - b;
			setRegAsU16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU32 d, a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);

			if (instr.carry & ir::PTXInstruction::CC) {
				ir::PTXU32 carry = 0;

				hydrazine::add(d, carry, a, -b, carry);
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
			else {
				d = a - b;
			}

			setRegAsU32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::u64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXU64 d, a = operandAsU64(threadID, instr.a),
				b = operandAsU64(threadID, instr.b);
			d = a - b;
			setRegAsU64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_SubC(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	switch (instr.type) {

	case ir::PTXOperand::u32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXU32 d = 0,
				a = operandAsU32(threadID, instr.a),
				b = operandAsU32(threadID, instr.b);

			ir::PTXU32 carry = getRegAsU32(threadID, instr.c.reg) - 1;

			hydrazine::add(d, carry, a, -b, carry);

			setRegAsU32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsU32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	case ir::PTXOperand::s32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			ir::PTXS32 d = 0,
				a = operandAsS32(threadID, instr.a),
				b = operandAsS32(threadID, instr.b);

			ir::PTXS32 carry = getRegAsS32(threadID, instr.c.reg) - 1;

			hydrazine::add(d, carry, a, -b, carry);

			setRegAsS32(threadID, instr.d.reg, d);

			if (instr.carry & ir::PTXInstruction::CC) {
				setRegAsS32(threadID, instr.pq.reg, carry);
			}
		}
	} break;

	default:
		throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!
	\brief load from surface memory
*/
void executive::CooperativeThreadArray::eval_Suld(CTAContext &context,
	const ir::PTXInstruction &instr) {
	switch (instr.geometry) {
	case ir::PTXInstruction::_1d:
	{
	}
		break;
	case ir::PTXInstruction::_2d:
	{

	}
		break;
	case ir::PTXInstruction::_3d:
	{

	}
		break;
	default:
		throw RuntimeException("suld - invalid geometry", context.PC, instr);
	}
	throw RuntimeException("suld - not implemented", context.PC, instr);
}

void executive::CooperativeThreadArray::eval_Sured(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("sured - not implemented", context.PC, instr);
}

/*!
	\brief store to surface memory
*/
void executive::CooperativeThreadArray::eval_Sust(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	throw RuntimeException("sust - not implemented", context.PC, instr);
}

/*!

*/
void executive::CooperativeThreadArray::eval_Tanh(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	auto tanhF16 = [](ir::PTXU16 a) {
		return toF16(std::tanh(f16ToF32(a)), ir::PTXInstruction::rn);
	};
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;
		if (instr.type == ir::PTXOperand::f16) {
			setRegAsB16(threadID, instr.d.reg,
				tanhF16(operandAsU16(threadID, instr.a)));
		}
		else if (instr.type == ir::PTXOperand::f16x2) {
			const ir::PTXU32 a = operandAsU32(threadID, instr.a);
			setRegAsU32(threadID, instr.d.reg, tanhF16(a)
				| (static_cast<ir::PTXU32>(tanhF16(a >> 16)) << 16));
		}
		else {
			const ir::PTXF32 a = operandAsF32(threadID, instr.a);
			const ir::PTXF32 d = std::fpclassify(a) == FP_SUBNORMAL
				? a : std::tanh(a);
			setRegAsF32(threadID, instr.d.reg, d);
		}
	}
}

void executive::CooperativeThreadArray::eval_TestP(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	const ir::PTXInstruction::FloatingPointMode mode = instr.floatingPointMode;

	switch(instr.type) {
	case ir::PTXOperand::f32:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			bool d = false;
			ir::PTXF32 a = operandAsF32(threadID, instr.a);

			switch(mode)
			{
			case ir::PTXInstruction::Finite:
			{
				d = !hydrazine::isinf(a) && !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Infinite:
			{
				d = hydrazine::isinf(a);
			}
			break;
			case ir::PTXInstruction::Number:
			{
				d = !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::NotANumber:
			{
				d = hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Normal:
			{
				d = hydrazine::isnormal(a);
			}
			break;
			case ir::PTXInstruction::SubNormal:
			{
				d = issubnormal_(a);
			}
			break;
			default: assertM(false, "Invalid floating point mode.");
			}

			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	break;
	case ir::PTXOperand::f64:
	{
		for (int threadID = 0; threadID  < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;
			bool d = false;
			ir::PTXF64 a = operandAsF64(threadID, instr.a);

			switch(mode)
			{
			case ir::PTXInstruction::Finite:
			{
				d = !hydrazine::isinf(a) && !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Infinite:
			{
				d = hydrazine::isinf(a);
			}
			break;
			case ir::PTXInstruction::Number:
			{
				d = !hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::NotANumber:
			{
				d = hydrazine::isnan(a);
			}
			break;
			case ir::PTXInstruction::Normal:
			{
				d = hydrazine::isnormal(a);
			}
			break;
			case ir::PTXInstruction::SubNormal:
			{
				d = issubnormal_(a);
			}
			break;
			default: assertM(false, "Invalid floating point mode.");
			}

			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	break;
	default: throw RuntimeException("invalid datatype", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Tex(CTAContext &context,
	const ir::PTXInstruction &instr) {
	const ir::Texture& texture = *context.kernel->textures[instr.a.reg];
	if (traceEvents) {
		currentEvent.memory_size = 4;
	}
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		switch (instr.geometry) {
			case ir::PTXInstruction::_1d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXS32 c = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(!instr.c.array.empty());
							assert(instr.d.array.size()==4);
							ir::PTXF32 c = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c);
							if (traceEvents) {
								tex::addresses(texture, c,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type",
							context.PC, instr);
				}
				break;

			case ir::PTXInstruction::_2d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(texture, c0, c1);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(texture, c0, c1);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()>=2);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(texture, c0, c1);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(texture, c0, c1);
							if (traceEvents) {
								tex::addresses(texture, c0, c1,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type",
							context.PC, instr);
				}
				break;

			case ir::PTXInstruction::_3d:
				switch (instr.d.type) {
					case ir::PTXOperand::u32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID,
								instr.c.array[2].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID,
								instr.c.array[2].reg);
							ir::PTXU32 d0 = tex::sample<0,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d1 = tex::sample<1,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d2 = tex::sample<2,ir::PTXU32>(
								texture, c0, c1, c2);
							ir::PTXU32 d3 = tex::sample<3,ir::PTXU32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsU32(threadID, instr.d.array[0].reg, d0);
							setRegAsU32(threadID, instr.d.array[1].reg, d1);
							setRegAsU32(threadID, instr.d.array[2].reg, d2);
							setRegAsU32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::s32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID,
								instr.c.array[2].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID,
								instr.c.array[2].reg);
							ir::PTXS32 d0 = tex::sample<0,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d1 = tex::sample<1,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d2 = tex::sample<2,ir::PTXS32>(
								texture, c0, c1, c2);
							ir::PTXS32 d3 = tex::sample<3,ir::PTXS32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsS32(threadID, instr.d.array[0].reg, d0);
							setRegAsS32(threadID, instr.d.array[1].reg, d1);
							setRegAsS32(threadID, instr.d.array[2].reg, d2);
							setRegAsS32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					case ir::PTXOperand::f32:
						if (instr.c.type == ir::PTXOperand::s32
							|| instr.c.type == ir::PTXOperand::u32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXS32 c0 = getRegAsS32(threadID,
								instr.c.array[0].reg);
							ir::PTXS32 c1 = getRegAsS32(threadID,
								instr.c.array[1].reg);
							ir::PTXS32 c2 = getRegAsS32(threadID,
								instr.c.array[2].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else if (instr.c.type == ir::PTXOperand::f32) {
							assert(instr.c.array.size()==4);
							assert(instr.d.array.size()==4);
							ir::PTXF32 c0 = getRegAsF32(threadID,
								instr.c.array[0].reg);
							ir::PTXF32 c1 = getRegAsF32(threadID,
								instr.c.array[1].reg);
							ir::PTXF32 c2 = getRegAsF32(threadID,
								instr.c.array[2].reg);
							ir::PTXF32 d0 = tex::sample<0,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d1 = tex::sample<1,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d2 = tex::sample<2,ir::PTXF32>(
								texture, c0, c1, c2);
							ir::PTXF32 d3 = tex::sample<3,ir::PTXF32>(
								texture, c0, c1, c2);
							if (traceEvents) {
								tex::addresses(texture, c0, c1, c2,
									currentEvent.memory_addresses);
							}
							setRegAsF32(threadID, instr.d.array[0].reg, d0);
							setRegAsF32(threadID, instr.d.array[1].reg, d1);
							setRegAsF32(threadID, instr.d.array[2].reg, d2);
							setRegAsF32(threadID, instr.d.array[3].reg, d3);
						}
						else {
							throw RuntimeException(
								"invalid texture source type",
								context.PC, instr);
						}
						break;

					default:
						throw RuntimeException(
							"invalid texture destination type",
							context.PC, instr);
				}
				break;
			case ir::PTXInstruction::_cube:
				{
					if (texture.size.z != 6) {
						throw RuntimeException(
							"Invalid texture dimensions. Must have depth of 6.", context.PC, instr);
					}
					if (instr.type == ir::PTXOperand::f32) {
						switch (instr.d.type) {
							case ir::PTXOperand::f32:
							{
								assert(instr.c.array.size()==4);
								assert(instr.d.array.size()==4);
								ir::PTXF32 c0 = getRegAsF32(threadID, instr.c.array[0].reg);
								ir::PTXF32 c1 = getRegAsF32(threadID, instr.c.array[1].reg);
								ir::PTXF32 c2 = getRegAsF32(threadID, instr.c.array[2].reg);

								ir::PTXF32 d0 = tex::sampleCube<0,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d1 = tex::sampleCube<1,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d2 = tex::sampleCube<2,ir::PTXF32>(texture, c0, c1, c2);
								ir::PTXF32 d3 = tex::sampleCube<3,ir::PTXF32>(texture, c0, c1, c2);
								if (traceEvents) {
									tex::addresses(texture, c0, c1, c2,	currentEvent.memory_addresses);
								}
								setRegAsF32(threadID, instr.d.array[0].reg, d0);
								setRegAsF32(threadID, instr.d.array[1].reg, d1);
								setRegAsF32(threadID, instr.d.array[2].reg, d2);
								setRegAsF32(threadID, instr.d.array[3].reg, d3);
							}
							break;
						case ir::PTXOperand::u32:	// fall through for now
						case ir::PTXOperand::s32:	// fall through for now
						default:
							throw RuntimeException(
								"invalid texture destination type",
								context.PC, instr);
						}
					}
					else {
						throw RuntimeException(
							"invalid texture source type",
							context.PC, instr);
					}
				}
				break;
			default:
				throw RuntimeException("invalid texture geometry",
					context.PC, instr);
				break;
		}
	}
	trace();
}

/*!

*/
void executive::CooperativeThreadArray::eval_Trap(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	context.executionState = CTAContext::Trap;
}

void executive::CooperativeThreadArray::eval_Suq(CTAContext &context, const ir::PTXInstruction &instr) {
	// this instruction is identical to txq except for surfaces which we don't distinguish from textures
	eval_Txq(context, instr);
}

void executive::CooperativeThreadArray::eval_Szext(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	for (int threadID = 0; threadID < threadCount; ++threadID) {
		if (!context.predicated(threadID, instr)) continue;

		const ir::PTXU32 a = operandAsU32(threadID, instr.a);
		const ir::PTXU32 b = operandAsU32(threadID, instr.b);
		const ir::PTXU32 b1 = b & 0x1f;
		const bool tooLarge = b >= 32
			&& instr.shiftMode == ir::PTXInstruction::ShiftMode::Clamp;
		const ir::PTXU32 mask = tooLarge ? 0 : (~0u << b1);
		const ir::PTXU32 signPos = (b1 - 1) & 0x1f;
		const bool signBit = b1 != 0 && !tooLarge
			&& instr.type == ir::PTXOperand::s32
			&& ((a >> signPos) & 1u);
		const ir::PTXU32 d = (a & ~mask) | (signBit ? mask : 0);

		setRegAsU32(threadID, instr.d.reg, d);
	}
}

void executive::CooperativeThreadArray::eval_Txq(CTAContext &context, const ir::PTXInstruction &instr) {
	trace();
	const ir::Texture& texture = *context.kernel->textures[instr.a.reg];
	switch (instr.surfaceQuery) {
	case ir::PTXInstruction::Width:	// fall through
	case ir::PTXInstruction::Height:	// fall through
	case ir::PTXInstruction::Depth:
	{
		int ir::Dim3::* dimensions[] = { &ir::Dim3::x, &ir::Dim3::y, &ir::Dim3::z };
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.size.*dimensions[instr.surfaceQuery - ir::PTXInstruction::Width]);
		}
	}
		break;
	case ir::PTXInstruction::ChannelDataType:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.type);
		}
	}
		break;
	case ir::PTXInstruction::ChannelOrder:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
		}
		throw RuntimeException("txq.channel_order - unimplemented query mode", context.PC, instr);
	}
		break;
	case ir::PTXInstruction::NormalizedCoordinates:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.normalize);
		}
	}
		break;
	case ir::PTXInstruction::SamplerFilterMode:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.interpolation);
		}
	}
		break;
	case ir::PTXInstruction::SamplerAddrMode0:	// fall through
	case ir::PTXInstruction::SamplerAddrMode1:	// fall through
	case ir::PTXInstruction::SamplerAddrMode2:
	{
		for (int tid = 0; tid < threadCount; tid++) {
			if (!context.predicated(tid, instr)) {
				continue;
			}
			setRegAsU32(tid, instr.d.reg, texture.addressMode[instr.surfaceQuery - ir::PTXInstruction::SamplerAddrMode0]);
		}
	}
		break;
	default:
		throw RuntimeException("txq - unexpected surface or texture query mode", context.PC, instr);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_WarpSync(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();

	const std::string error = instr.valid();
	if (!error.empty()) throw RuntimeException(error, context.PC, instr);
	reconvergenceMechanism->eval_WarpSync(context, instr);
}

void executive::CooperativeThreadArray::eval_Vote(CTAContext &context,
	const ir::PTXInstruction &instr) {
	std::vector<ir::PTXU32> results(threadCount);
	for (int warpStart = 0; warpStart < threadCount; warpStart += 32) {
		const int lanes = min(32, threadCount - warpStart);
		ir::PTXU32 masks[32] = {}, ballot = 0, participants = 0;
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			const ir::PTXU32 bit = ir::PTXU32(1) << lane;
			participants |= bit;
			masks[lane] = operandAsU32(warpStart + lane, instr.b);
			const bool value = instr.a.condition == ir::PTXOperand::PT ||
				(instr.a.condition != ir::PTXOperand::nPT && operandAsPredicate(warpStart + lane, instr.a));
			if (value) ballot |= bit;
		}
		// All predicates and member masks are gathered before aliased destinations are written.
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			const ir::PTXU32 members = masks[lane] & participants;
			const ir::PTXU32 selected = ballot & members;
			results[warpStart + lane] = instr.vote == ir::PTXInstruction::Ballot ? selected :
				instr.vote == ir::PTXInstruction::All ? selected == members :
				instr.vote == ir::PTXInstruction::Any ? selected != 0 : 
				selected == 0 || selected == members; // uni
		}
	}
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		if (instr.vote == ir::PTXInstruction::Ballot) setRegAsB32(thread, instr.d.reg, results[thread]);
		else setRegAsPredicate(thread, instr.d.reg, results[thread]);
	}
}

void executive::CooperativeThreadArray::eval_Shfl(CTAContext &context,
	const ir::PTXInstruction &instr) {
	std::vector<ir::PTXU32> results(threadCount);
	std::vector<bool> predicates(threadCount);
	for (int warpStart = 0; warpStart < threadCount; warpStart += 32) {
		const int lanes = min(32, threadCount - warpStart);
		ir::PTXU32 values[32] = {};
		for (int lane = 0; lane < lanes; ++lane)
			if (context.predicated(warpStart + lane, instr)) values[lane] = operandAsB32(warpStart + lane, instr.a);
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			const int offset = operandAsU32(warpStart + lane, instr.b) & 31;
			const ir::PTXU32 control = operandAsU32(warpStart + lane, instr.c);
			const int segmask = (control >> 8) & 31;
			const int limit = (lane & segmask) | ((control & 31) & ~segmask);
			int from = lane;
			bool inRange = false;
			switch (instr.shuffleMode) {
				case ir::PTXInstruction::Up: from = lane - offset; inRange = from >= limit; break;
				case ir::PTXInstruction::Down: from = lane + offset; inRange = from <= limit; break;
				case ir::PTXInstruction::Bfly: from = lane ^ offset; inRange = from <= limit; break;
				case ir::PTXInstruction::Idx: from = (lane & segmask) | (offset & ~segmask); inRange = from <= limit; break;
				default: throw RuntimeException("invalid shuffle mode", context.PC, instr);
			}
			predicates[warpStart + lane] = inRange;
			results[warpStart + lane] = values[inRange ? from : lane];
		}
	}
	// All data and controls are read before writing aliased data or predicate destinations.
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		if (instr.d.addressMode != ir::PTXOperand::BitBucket) setRegAsB32(thread, instr.d.reg, results[thread]);
		if (instr.pq.addressMode == ir::PTXOperand::Register) setRegAsPredicate(thread, instr.pq.reg, predicates[thread]);
	}
}

void executive::CooperativeThreadArray::eval_Match(CTAContext &context,
	const ir::PTXInstruction &instr) {
	std::vector<ir::PTXU32> results(threadCount);
	std::vector<bool> unanimous(threadCount);
	for (int warpStart = 0; warpStart < threadCount; warpStart += 32) {
		const int lanes = min(32, threadCount - warpStart);
		ir::PTXU64 values[32] = {};
		ir::PTXU32 masks[32] = {}, participants = 0;
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			participants |= ir::PTXU32(1) << lane;
			masks[lane] = operandAsU32(warpStart + lane, instr.b);
			values[lane] = instr.type == ir::PTXOperand::b64 ?
				operandAsB64(warpStart + lane, instr.a) : operandAsB32(warpStart + lane, instr.a);
		}
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			const ir::PTXU32 members = masks[lane] & participants;
			ir::PTXU32 matching = 0;
			for (int member = 0; member < lanes; ++member) {
				const ir::PTXU32 bit = ir::PTXU32(1) << member;
				if ((members & bit) && values[member] == values[lane])
					matching |= bit;
			}
			const bool allMatch = matching == members;
			unanimous[warpStart + lane] = allMatch;
			results[warpStart + lane] = instr.matchAll ? (allMatch ? members : 0) : matching;
		}
	}
	// Gather every source, mask and result before writing aliased destinations.
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		if (instr.d.addressMode != ir::PTXOperand::BitBucket) setRegAsB32(thread, instr.d.reg, results[thread]);
		if (instr.pq.addressMode == ir::PTXOperand::Register) setRegAsPredicate(thread, instr.pq.reg, unanimous[thread]);
	}
}

void executive::CooperativeThreadArray::eval_Redux(CTAContext &context,
	const ir::PTXInstruction &instr) {
	std::vector<ir::PTXU32> results(threadCount);
	for (int warpStart = 0; warpStart < threadCount; warpStart += 32) {
		const int lanes = min(32, threadCount - warpStart);
		ir::PTXU32 masks[32] = {}, values[32] = {}, participants = 0;
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			participants |= ir::PTXU32(1) << lane;
			masks[lane] = operandAsU32(warpStart + lane, instr.b);
			values[lane] = operandAsB32(warpStart + lane, instr.a);
		}
		for (int lane = 0; lane < lanes; ++lane) {
			if (!context.predicated(warpStart + lane, instr)) continue;
			const ir::PTXU32 members = masks[lane] & participants;
			ir::PTXU32 result = 0;
			const ir::PTXU32 sign = instr.type == ir::PTXOperand::s32 ? 0x80000000u : 0;
			bool first = true;
			for (int member = 0; member < lanes; ++member) {
				if (!(members & (ir::PTXU32(1) << member))) continue;
				const ir::PTXU32 value = values[member];
				if (first) { result = value; first = false; continue; }
				switch (instr.reductionOperation) {
					case ir::PTXInstruction::ReductionAdd: result += value; break;
					case ir::PTXInstruction::ReductionMin: if ((value ^ sign) < (result ^ sign)) result = value; break;
					case ir::PTXInstruction::ReductionMax: if ((value ^ sign) > (result ^ sign)) result = value; break;
					case ir::PTXInstruction::ReductionAnd: result &= value; break;
					case ir::PTXInstruction::ReductionOr: result |= value; break;
					case ir::PTXInstruction::ReductionXor: result ^= value; break;
					default: throw RuntimeException("invalid redux operation", context.PC, instr);
				}
			}
			results[warpStart + lane] = result;
		}
	}
	// Gather all sources and masks before writing potentially aliased destinations.
	for (int thread = 0; thread < threadCount; ++thread) {
		if (!context.predicated(thread, instr)) continue;
		setRegAsB32(thread, instr.d.reg, results[thread]);
	}
}

/*!

*/
void executive::CooperativeThreadArray::eval_Xor(CTAContext &context,
	const ir::PTXInstruction &instr) {
	trace();
	if (instr.type == ir::PTXOperand::pred) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			bool d, a = operandAsPredicate(threadID, instr.a),
				b = operandAsPredicate(threadID, instr.b);
			d = b ^ a;
			setRegAsPredicate(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b16) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB16 d, a = operandAsB16(threadID, instr.a),
				b = operandAsB16(threadID, instr.b);
			d = a ^ b;
			setRegAsB16(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b32) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB32 d, a = operandAsB32(threadID, instr.a),
				b = operandAsB32(threadID, instr.b);
			d = a ^ b;
			setRegAsB32(threadID, instr.d.reg, d);
		}
	}
	else if (instr.type == ir::PTXOperand::b64) {
		for (int threadID = 0; threadID < threadCount; threadID++) {
			if (!context.predicated(threadID, instr)) continue;

			ir::PTXB64 d, a = operandAsB64(threadID, instr.a),
				b = operandAsB64(threadID, instr.b);
			d = a ^ b;
			setRegAsB64(threadID, instr.d.reg, d);
		}
	}
	else {
		throw RuntimeException("unsupported data type", context.PC, instr);
	}
}

void executive::CooperativeThreadArray::eval_cudaLaunchDevice(CTAContext &context,
	const ir::PTXInstruction &instr) {
	for (int threadID = 0; threadID < threadCount; threadID++) {
		if (!context.predicated(threadID, instr)) continue;
		// parameter 0 - the PC of the kernel being launched
		// parameter 1 - parameter buffer address
		// parameter 2 - kernel dim
		// parameter 3 - cta dim
		// parameter 4 - shared memory size
		// parameter 5 - stream

		ir::PTXU64 functionPointer =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 0);
		ir::PTXU64 parameterBuffer  =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 1);
		ir::Dim3   gridDim          =
			getFunctionParameter<ir::Dim3>(threadID, instr, 2);
		ir::Dim3   ctaDim           =
			getFunctionParameter<ir::Dim3>(threadID, instr, 3);
		ir::PTXU32 sharedMemory     =
			getFunctionParameter<ir::PTXU32>(threadID, instr, 4);
		ir::PTXU64 stream           =
			getFunctionParameter<ir::PTXU64>(threadID, instr, 5);

		kernel->scheduler->launch(functionPointer, parameterBuffer, gridDim,
			ctaDim, sharedMemory, stream);

		// return parameter 0 - the status of the call
		setFunctionParameter(threadID, instr, 0, (ir::PTXU32)0);
	}

	context.PC++;
}

void executive::CooperativeThreadArray::eval_cudaSynchronizeDevice(
	CTAContext &context,
	const ir::PTXInstruction &instr) {
	context.executionState = CTAContext::Barrier;
}
