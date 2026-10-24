/*!
	\file PTXInstruction.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief base class for all instructions
*/

#include <ocelot/ir/PTXInstruction.h>
#include <hydrazine/debug.h>
#include <sstream>

std::string ir::PTXInstruction::toString( Level l ) {
	switch( l ) {
		case CtaLevel:    return "cta"; break;
		case GlobalLevel: return "gl";  break;
		case SystemLevel: return "sys";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( Semantics s ) {
	switch( s ) {
		case Sc:      return "sc";      break;
		case AcqRel:  return "acq_rel"; break;
		case Acquire: return "acquire"; break;
		case Release: return "release"; break;
		case Relaxed: return "relaxed"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString(CacheLevel cache) {
	switch( cache ) {
		case L1: return "L1";
		case L2: return "L2";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toStringLoad(CacheOperation operation) {
	switch( operation ) {
		case Ca: return "ca";
		case Cg: return "cg";
		case Cs: return "cs";
		case Cv: return "cv";
		case Nc: return "nc";
		case Lu: return "lu";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toStringStore(CacheOperation operation) {
	switch( operation ) {
		case Cg: return "cg";
		case Cs: return "cs";
		case Wb: return "wb";
		case Wt: return "wt";
		default: break;
	}
	return "";
}


std::string ir::PTXInstruction::toString( PermuteMode m ) {
	switch( m ) {
		case ForwardFourExtract:  return "f4e"; break;
		case BackwardFourExtract: return "b4e"; break;
		case ReplicateEight:      return "rc8"; break;
		case EdgeClampLeft:       return "ecl"; break;
		case EdgeClampRight:      return "ecr"; break;
		case ReplicateSixteen:    return "rc16"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( FloatingPointMode m ) {
	switch( m ) {
		case Finite:     return "finite"; break;
		case Infinite:   return "infinite"; break;
		case Number:     return "number"; break;
		case NotANumber: return "notanumber"; break;
		case Normal:     return "normal"; break;
		case SubNormal:  return "subnormal"; break;
		default: break;
	}
	return "";
}
std::string ir::PTXInstruction::toString( Vec v ) {
	switch( v ) {
		case PTXOperand::v1: return "v1"; break;
		case PTXOperand::v2: return "v2"; break;
		case PTXOperand::v4: return "v4"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( AddressSpace space ) {
	switch( space ) {
		case Const:    return "const";   break;
		case Global:   return "global";  break;
		case Local:    return "local";   break;
		case Param:    return "param";   break;
		case Shared:   return "shared";  break;
		case Texture:  return "tex";     break;
		case Generic:  return "generic"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( AtomicOperation operation ) {
	switch( operation ) {
		case AtomicAnd:  return "and";  break;
		case AtomicOr:   return "or";   break;
		case AtomicXor:  return "xor";  break;
		case AtomicCas:  return "cas";  break;
		case AtomicExch: return "exch"; break;
		case AtomicAdd:  return "add";  break;
		case AtomicInc:  return "inc";  break;
		case AtomicDec:  return "dec";  break;
		case AtomicMin:  return "min";  break;
		case AtomicMax:  return "max";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ReductionOperation operation ) {
	switch( operation ) {
		case ReductionAnd: return "and"; break;
		case ReductionOr:  return "or";  break;
		case ReductionXor: return "xor"; break;
		case ReductionAdd: return "add"; break;
		case ReductionInc: return "inc"; break;
		case ReductionDec: return "dec"; break;
		case ReductionMin: return "min"; break;
		case ReductionMax: return "max"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( SurfaceQuery query ) {
	switch (query) {
		case Width:                  return "width";
		case Height:                 return "height";
		case Depth:                  return "depth";
		case ChannelDataType:        return "channel_data_type";
		case ChannelOrder:           return "channel_order";
		case NormalizedCoordinates:  return "normalized_coords";
		case SamplerFilterMode:      return "filter_mode";
		case SamplerAddrMode0:       return "addr_mode_0";
		case SamplerAddrMode1:       return "addr_mode_1";
		case SamplerAddrMode2:       return "addr_mode_2";
		default:                     break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( FormatMode mode ) {
	switch (mode) {
		case Unformatted:        return ".b";
		case Formatted:          return ".p";
		case FormatMode_Invalid: break;
		default:                 break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ClampOperation clamp ) {
	switch (clamp) {
		case ClampOperation::TrapOOB:                return ".trap";
		case ClampOperation::Clamp:                  return ".clamp";
		case ClampOperation::Zero:                   return ".zero";
		case ClampOperation::Mirror:                 return ".mirror";
		case ClampOperation::ClampOperation_Invalid: break;
		default:                     break;
	}
	return "";
}

std::string ir::PTXInstruction::roundingMode( Modifier modifier ) {
	switch( modifier ) {
		case rn: return "rn"; break;
		case rz: return "rz"; break;
		case rm: return "rm"; break;
		case rp: return "rp"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::modifierString( unsigned int modifier, 
	CarryFlag carry ) {
	std::string result;
	if( modifier & approx ) {
		result += "approx.";
	}
	else if( modifier & wide ) {
		result += "wide.";
	}
	else if( modifier & hi ) {
		result += "hi.";
	}
	else if( modifier & lo ) {
		result += "lo.";
	}
	else if( modifier & rn ) {
		result += "rn.";
	}
	else if( modifier & rna ) {
		result += "rna.";
	}
	else if( modifier & rz ) {
		result += "rz.";
	}
	else if( modifier & rm ) {
		result += "rm.";
	}
	else if( modifier & rp ) {
		result += "rp.";
	}

	if( modifier & ftz ) {
		result += "ftz.";
	}
	if( modifier & nan ) {
		result += "NaN.";
	}
	if( modifier & xorsign ) {
		result += "xorsign.";
	}
	if( modifier & abs ) {
		result += "abs.";
	}
	if( modifier & sat ) {
		result += "sat.";
	}
	if( modifier & relu ) {
		result += "relu.";
	}
	if( carry == CC ) {
		result += "cc.";
	}
	return result;
}

std::string ir::PTXInstruction::toString( Modifier modifier ) {
	switch( modifier ) {
		case hi:     return "hi";     break;
		case lo:     return "lo";     break;
		case wide:   return "wide";   break;
		case sat:    return "sat";    break;
		case rn:     return "rn";     break;
		case rna:    return "rna";    break;
		case rz:     return "rz";     break;
		case rm:     return "rm";     break;
		case rp:     return "rp";     break;
		case approx: return "approx"; break;
		case ftz:    return "ftz";    break;
		case nan:    return "NaN";    break;
		case xorsign:return "xorsign";break;
		case abs:    return "abs";    break;
		case relu:   return "relu";   break;
		default: break;
	}
	return "";	
}


std::string ir::PTXInstruction::toString( BarrierOperation operation) {
	switch (operation) {
		case BarSync:      return "sync";
		case BarArrive:    return "arrive";
		case BarReduction: return "red";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( CmpOp operation ) {
	switch( operation ) {
		case Eq: return "eq";   break;
		case Ne: return "ne";   break;
		case Lt: return "lt";   break;
		case Le: return "le";   break;
		case Gt: return "gt";   break;
		case Ge: return "ge";   break;
		case Lo: return "lo";   break;
		case Ls: return "ls";   break;
		case Hi: return "hi";   break;
		case Hs: return "hs";   break;
		case Equ: return "equ"; break;
		case Neu: return "neu"; break;
		case Ltu: return "ltu"; break;
		case Leu: return "leu"; break;
		case Gtu: return "gtu"; break;
		case Geu: return "geu"; break;
		case Num: return "num"; break;
		case Nan: return "nan"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( BoolOp operation ) {
	switch( operation ) {
		case BoolAnd: return "and"; break;
		case BoolOr:  return "or";  break;
		case BoolXor: return "xor"; break;	
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( Geometry geometry ) {
	switch( geometry ) {
		case _1d: return "1d"; break;
		case _2d: return "2d"; break;
		case _3d: return "3d"; break;
		case _a1d: return "a1d"; break;
		case _a2d: return "a2d"; break;
		case _cube: return "cube"; break;
		case _acube: return "acube"; break;		
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( VoteMode mode ) {
	switch( mode ) {
		case All:    return "all";    break;
		case Any:    return "any";    break;
		case Uni:    return "uni";    break;
		case Ballot: return "ballot"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ShuffleMode mode ) {
	switch( mode ) {
		case Up:   return "up";   break;
		case Down: return "down"; break;
		case Bfly: return "bfly"; break;
		case Idx:  return "idx";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ShiftMode mode ) {
	switch( mode ) {
		case ShiftMode::Clamp:	return "clamp"; break;
		case ShiftMode::Wrap:	return "wrap";	break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ShiftDirection direction ) {
	switch( direction ) {
		case ShiftDirection::ShiftLeft:  return "l"; break;
		case ShiftDirection::ShiftRight: return "r"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ColorComponent color ) {
	switch( color ) {
		case red:   return "r"; break;
		case green: return "g"; break;
		case blue:  return "b"; break;
		case alpha: return "a"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( Opcode opcode ) {
	switch( opcode ) {
		case Abs:        return "abs";        break;
		case Add:        return "add";        break;
		case AddC:       return "addc";       break;
		case And:        return "and";        break;
		case Atom:       return "atom";       break;
		case Bar:        return "bar";        break;
		case Bfe:        return "bfe";        break;
		case Bfi:        return "bfi";        break;
		case Bfind:      return "bfind";      break;
		case Bmsk:       return "bmsk";       break;
		case Bra:        return "bra";        break;
		case Brev:       return "brev";       break;
		case Brkpt:      return "brkpt";      break;
		case Call:       return "call";       break;
		case Clz:        return "clz";        break;
		case CNot:       return "cnot";       break;
		case CopySign:   return "copysign";   break;
		case Cos:        return "cos";        break;
		case Cvt:        return "cvt";        break;
		case Cvta:       return "cvta";       break;
		case Div:        return "div";        break;
		case Dp2a:       return "dp2a";       break;
		case Dp4a:       return "dp4a";       break;
		case Ex2:        return "ex2";        break;
		case Exit:       return "exit";       break;
		case Fma:        return "fma";        break;
		case Fns:        return "fns";        break;
		case Isspacep:   return "isspacep";   break;
		case Ld:         return "ld";         break;
		case Ldu:        return "ldu";        break;
		case Ldmatrix:   return "ldmatrix";   break;
		case Lg2:        return "lg2";        break;
		case Lop3:       return "lop3";       break;
		case Mad24:      return "mad24";      break;
		case Mad:        return "mad";        break;
		case Mma:        return "mma";        break;
		case MadC:       return "madc";        break;
		case Max:        return "max";        break;
		case Membar:     return "membar";     break;
		case Fence:      return "fence";      break;
		case Min:        return "min";        break;
		case Activemask: return "activemask";
		case Movmatrix:  return "movmatrix";
		case Mov:        return "mov";        break;
		case Mul24:      return "mul24";      break;
		case Mul:        return "mul";        break;
		case Neg:        return "neg";        break;
		case Not:        return "not";        break;
		case Or:         return "or";         break;
		case Pmevent:    return "pmevent";    break;
		case Popc:       return "popc";       break;
		case Prefetch:   return "prefetch";   break;
		case Prefetchu:  return "prefetchu";  break;
		case Prmt:       return "prmt";       break;
		case Rcp:        return "rcp";        break;
		case Match:      return "match";
		case Redux:      return "redux";
		case Red:        return "red";        break;
		case Rem:        return "rem";        break;
		case Ret:        return "ret";        break;
		case Rsqrt:      return "rsqrt";      break;
		case Sad:        return "sad";        break;
		case SelP:       return "selp";       break;
		case Set:        return "set";        break;
		case SetP:       return "setp";       break;
		case Shf:        return "shf";        break;
		case Shfl:       return "shfl";       break;
		case Shl:        return "shl";        break;
		case Shr:        return "shr";        break;
		case Sin:        return "sin";        break;
		case SlCt:       return "slct";       break;
		case Sqrt:       return "sqrt";       break;
		case St:         return "st";         break;
		case Sub:        return "sub";        break;
		case SubC:       return "subc";       break;
		case Suld:       return "suld";       break;
		case Sured:      return "sured";      break;
		case Sust:       return "sust";       break;
		case Suq:        return "suq";        break;
		case Szext:      return "szext";      break;
		case Tanh:       return "tanh";       break;
		case TestP:      return "testp";      break;
		case Tex:        return "tex";        break;
		case Tld4:       return "tld4";       break;
		case Txq:        return "txq";        break;
		case Trap:       return "trap";       break;
		case Vabsdiff:   return "vabsdiff";   break;
		case Vadd:       return "vadd";       break;
		case Vmad:       return "vmad";       break;
		case Vmax:       return "vmax";       break;
		case Vmin:       return "vmin";       break;
		case Vset:       return "vset";       break;
		case Vshl:       return "vshl";       break;
		case Vshr:       return "vshr";       break;
		case Vsub:       return "vsub";       break;
		case Vote:       return "vote";       break;
		case Xor:        return "xor";        break;
		case Reconverge: return "reconverge"; break;
		case Phi:        return "phi";        break;
		case Nop:        return "nop";        break;
		case Invalid_Opcode: break;
	}
	return "INVALID";
}

bool ir::PTXInstruction::isPt( const PTXOperand& op )
{
	return op.toString() == "%pt";
}

ir::PTXInstruction::PTXInstruction( Opcode op, const PTXOperand& _d, 
	const PTXOperand& _a, const PTXOperand& _b, const PTXOperand& _c ) 
	: opcode(op), d(_d), a(_a), b(_b), c(_c) {
	ISA = Instruction::PTX;
	type = PTXOperand::s32;
	bType = PTXOperand::TypeSpecifier_invalid;
	modifier = 0;
	reconvergeInstruction = 0;
	branchTargetInstruction = 0;
	vec = PTXOperand::v1;
	mmaShape = MmaShape_Invalid;
	mmaAColumnMajor = false;
	mmaBColumnMajor = true;
	mmaSparse = false;
	matrixTranspose = false;
	voteSync = false;
	matchAll = false;
	pg.condition = PTXOperand::PT;
	pg.type = PTXOperand::pred;
	barrierOperation = BarSync;
	carry = None;
	statementIndex = -1;
	geometry = Geometry_Invalid;
	cc = 0;
	addressSpace = AddressSpace_Invalid;
	tailCall = false;
	mmio = false;
	cacheOperation = Ca;
	booleanOperator = BoolAnd;
	semantics = AcqRel;
	scope = Level_Invalid;
}

ir::PTXInstruction::~PTXInstruction() {

}

bool ir::PTXInstruction::operator==( const PTXInstruction& i ) const {
	return opcode == i.opcode;
}

std::string ir::PTXInstruction::valid() const {
	if (opcode != Mma && (d.vec == PTXOperand::v8 || a.vec == PTXOperand::v8 ||
		b.vec == PTXOperand::v8 || c.vec == PTXOperand::v8)) return "eight-register fragments require mma";
	if( opcode == Min || opcode == Max ) {
		const unsigned int fp32Modifiers = nan | xorsign | abs;
		if( (modifier & fp32Modifiers) && type != PTXOperand::f32
			&& type != PTXOperand::f16 && type != PTXOperand::f16x2
			&& type != PTXOperand::bf16 && type != PTXOperand::bf16x2 ) {
			return "NaN and xorsign.abs modifiers require a floating type";
		}
		if ((modifier & ftz) && (type == PTXOperand::bf16 || type == PTXOperand::bf16x2))
			return "ftz is invalid for bf16 min/max";
		if( (modifier & (xorsign | abs)) != 0
			&& (modifier & (xorsign | abs)) != (xorsign | abs) ) {
			return "xorsign and abs modifiers must be specified together";
		}
	}
	switch (opcode) {
		case Abs: {
			if ( !( type == PTXOperand::s16 || type == PTXOperand::s32 || 
				type == PTXOperand::s64 || type == PTXOperand::f32 || 
				type == PTXOperand::f64 || type == PTXOperand::f16 ||
				type == PTXOperand::bf16 || type == PTXOperand::f16x2 ||
				type == PTXOperand::bf16x2 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );	
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}		
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Add: {
			if ( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if ((type == PTXOperand::f16 || type == PTXOperand::f16x2)
				&& (modifier & ~(rn | ftz | sat))) {
				return "add.f16 only supports .rn, .ftz, and .sat";
			}
			if( carry == CC ) {
				if( ( modifier & sat ) ) {
					return "saturate not supported with carry out";
				}
				if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
					return "invalid instruction type " 
						+ PTXOperand::toString( type );
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case AddC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case And: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;	
		}
		case Atom: {
			if( ( atomicOperation == AtomicAnd || atomicOperation == AtomicOr
				|| atomicOperation == AtomicXor || atomicOperation == AtomicCas
				|| atomicOperation == AtomicExch )
				&& type != PTXOperand::b32 && type != PTXOperand::b64 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for atomic "
					+ toString( atomicOperation );
			}
			if( ( atomicOperation == AtomicInc || atomicOperation == AtomicDec )
				&& type != PTXOperand::u32 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for atomic "
					+ toString( atomicOperation );
			}
			if( atomicOperation == AtomicAdd
				&& type != PTXOperand::u32 && type != PTXOperand::s32
				&& type != PTXOperand::u64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for atomic "
					+ toString( atomicOperation );
			}
			if( ( atomicOperation == AtomicMin || atomicOperation == AtomicMax )
				&& type != PTXOperand::u32 && type != PTXOperand::s32
				&& type != PTXOperand::u64 && type != PTXOperand::s64 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for atomic "
					+ toString( atomicOperation );
			}
			if( !( addressSpace == Shared || addressSpace == Global ) ) {
				return "invalid adress space " + toString( addressSpace );
			}
			break;
		}
		case Bar: {
			if( d.addressMode != PTXOperand::Immediate ) {
				return "only support Immediate targets";
			}
			break;
		}
        case Bfe: {
            if( !( type == PTXOperand::u32 
                   || type == PTXOperand::s32 
                   || type == PTXOperand::u64
                   || type == PTXOperand::s64 ) ){
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( type, d.type ) ) {
                return "operand D type " + PTXOperand::toString( d.type )
                    + " cannot be assigned to " + PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( type, a.type )
                && a.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( a.type )
                    + " cannot be assigned to " + PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( PTXOperand::u32, b.type )
                && b.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( b.type )
                    + " cannot be assigned to " + PTXOperand::toString( PTXOperand::u32 );
            }
            if( !PTXOperand::valid( PTXOperand::u32, c.type )
                && c.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( c.type )
                    + " cannot be assigned to " + PTXOperand::toString( PTXOperand::u32 );
            }

            break;
        }
		case Bfi: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, pq.type ) 
				&& pq.addressMode != PTXOperand::Immediate ) {
				return "operand 1 type " + PTXOperand::toString( pq.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand 2 type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, b.type )  ) {
				return "operand 3 type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			if( !PTXOperand::valid( PTXOperand::u32, c.type )  ) {
				return "operand 4 type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Bfind: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::u64 
				|| type == PTXOperand::s32 || type == PTXOperand::s64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Bmsk: {
			if( type != PTXOperand::b32 ) {
				return "bmsk instruction requires .b32 type";
			}
			if( shiftMode != ShiftMode::Clamp && shiftMode != ShiftMode::Wrap ) {
				return "bmsk instruction requires .clamp or .wrap mode";
			}
			if( !PTXOperand::valid( type, d.type )
				|| !PTXOperand::valid( type, a.type )
				|| !PTXOperand::valid( type, b.type ) ) {
				return "bmsk operands must have 32-bit integer types";
			}
			break;
		}
		case Lop3: {
			if( type != PTXOperand::b32 ) {
				return "lop3 instruction requires .b32 type";
			}
			const bool predicateResult = pq.addressMode != PTXOperand::Invalid;
			if( (d.addressMode == PTXOperand::BitBucket && !predicateResult)
				|| (d.addressMode != PTXOperand::BitBucket
					&& !PTXOperand::valid( type, d.type ))
				|| !PTXOperand::valid( type, a.type )
				|| !PTXOperand::valid( type, b.type ) ) {
				return "lop3 data operands must have 32-bit integer types";
			}
			if( !PTXOperand::valid( type, c.type ) ) {
				return "lop3 data operands must have 32-bit integer types";
			}
			if( immLut.addressMode != PTXOperand::Immediate
				|| immLut.imm_uint > 0xff ) {
				return "lop3 immLut must be an integer constant from 0 to 255";
			}
			if( predicateResult && (pq.type != PTXOperand::pred
				|| q.type != PTXOperand::pred
				|| (booleanOperator != BoolAnd && booleanOperator != BoolOr)) ) {
				return "lop3 predicate form requires .and or .or and predicates p, q";
			}
			break;
		}
		case Bra: {
			if( !( d.addressMode == PTXOperand::Label 
				|| d.addressMode == PTXOperand::Register ) ) {
				return "no support for types other than Label and Register";
			}
			break;
		}
		case Brev: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Brkpt: {
			break;
		}
		case Call: {
			if( a.addressMode != PTXOperand::Register
				&& a.addressMode != PTXOperand::FunctionName ) {
				return "operand A must be a function name or register.";
			}
			if( d.addressMode != PTXOperand::ArgumentList 
				&& d.addressMode != PTXOperand::Register 
				&& d.addressMode != PTXOperand::Invalid ) {
				return "operand D must be an argument/register "
					"list if it is specified.";
			}
			if( b.addressMode != PTXOperand::ArgumentList
				&& b.addressMode != PTXOperand::Register 
				&& b.addressMode != PTXOperand::Invalid ) {
				return "operand B must be an argument/register "
					"list if it is specified.";
			}
			if( a.addressMode == PTXOperand::Register
				&& c.addressMode != PTXOperand::FunctionName ) {
				return "operand C must be function name if A is a register.";
			}
			
			for( auto operand = d.array.begin();
				operand != d.array.end(); ++operand )
			{
				if (operand->addressMode != PTXOperand::Register &&
					operand->addressMode != PTXOperand::BitBucket &&
					operand->addressMode != PTXOperand::Address &&
					operand->addressMode != PTXOperand::Immediate) {
					return "return arguments must be registers, parameters, "
						"or immediates.";
				}
			}
			
			for( auto operand = b.array.begin();
				operand != b.array.end(); ++operand )
			{
				if (operand->addressMode != PTXOperand::Register &&
					operand->addressMode != PTXOperand::BitBucket &&
					operand->addressMode != PTXOperand::Address &&
					operand->addressMode != PTXOperand::Immediate) {
					return "function arguments must be registers"
						", parameters, or immediates.";
				}
			}
			
			break;
		}
		case Clz: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case CNot: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case CopySign : {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Cos: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( modifier != approx && modifier != ( approx | ftz ) ) {
				return "cos requires .approx with optional .ftz";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Cvt: {
			PTXOperand::DataType sourceType = a.relaxedType ==
				PTXOperand::TypeSpecifier_invalid ? a.type : a.relaxedType;
			const int cvtModifiers = rni | rzi | rmi | rpi | rn | rna | rz
				| rm | rp | ftz | sat | relu;
			if( modifier & ~cvtModifiers ) return "invalid cvt modifier";
			const bool packed = type == PTXOperand::f16x2
				|| type == PTXOperand::bf16x2;
			if (!packed && b.addressMode != PTXOperand::Invalid) {
				return "scalar cvt accepts only two operands";
			}
			if( packed ) {
				const PTXOperand::DataType bSourceType = b.relaxedType ==
					PTXOperand::TypeSpecifier_invalid ? b.type : b.relaxedType;
				if( sourceType != PTXOperand::f32
					|| bSourceType != PTXOperand::f32 ) {
					return "packed cvt requires two f32 source operands";
				}
				const int rounding = modifier & (rn | rz);
				if( rounding != rn && rounding != rz ) {
					return "packed cvt requires .rn or .rz";
				}
				if( modifier & ~(rn | rz | relu) ) {
					return "invalid packed cvt modifier";
				}
				if( type == PTXOperand::f16x2
					? !PTXOperand::relaxedValid(type, d.type)
					: d.type != PTXOperand::b32 ) {
					return "packed cvt requires a b32 destination";
				}
				break;
			}
			if( type == PTXOperand::tf32 ) {
				if( sourceType != PTXOperand::f32 || modifier != rna ) {
					return "sm_86 tf32 cvt requires cvt.rna.tf32.f32";
				}
				if( d.type != PTXOperand::b32 ) {
					return "tf32 cvt requires a b32 destination";
				}
				break;
			}
			if( type == PTXOperand::bf16 ) {
				const int rounding = modifier & (rn | rz);
				if( sourceType != PTXOperand::f32
					|| (modifier & ~(rn | rz | relu))
					|| (rounding != rn && rounding != rz) ) {
					return "sm_86 bf16 cvt requires cvt.{rn,rz}{.relu}.bf16.f32";
				}
				if( d.type != PTXOperand::b16 ) {
					return "bf16 cvt requires a b16 destination";
				}
				break;
			}
			if( sourceType == PTXOperand::bf16 ) {
				if( type != PTXOperand::f32 || modifier ) {
					return "sm_86 supports only cvt.f32.bf16";
				}
				if( !PTXOperand::relaxedValid(type, d.type) ) {
					return "cvt.f32.bf16 requires an f32-compatible destination";
				}
				break;
			}
			if( sourceType == PTXOperand::f16x2
				|| sourceType == PTXOperand::bf16x2
				|| sourceType == PTXOperand::tf32 ) {
				return "cvt source type is not supported on sm_86";
			}
			if( !PTXOperand::isInt(type) && !PTXOperand::isFloat(type)
				) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::isInt(sourceType)
				&& !PTXOperand::isFloat(sourceType) ) {
				return "invalid source type " + PTXOperand::toString(sourceType);
			}
			const bool destinationFloat = PTXOperand::isFloat(type);
			const bool sourceFloat = PTXOperand::isFloat(sourceType);
			const bool destinationInt = PTXOperand::isInt(type);
			const bool sourceInt = PTXOperand::isInt(sourceType);
			const int integerRounding = modifier & (rni | rzi | rmi | rpi);
			const int floatRounding = modifier & (rn | rz | rm | rp);
			const int rounding = integerRounding | floatRounding;
			if( rounding && (rounding & (rounding - 1)) ) {
				return "cvt accepts only one rounding modifier";
			}
			const bool narrowerFloat = sourceFloat && destinationFloat
				&& PTXOperand::bytes(type) < PTXOperand::bytes(sourceType);
			const bool sameSizeFloat = sourceFloat && destinationFloat
				&& PTXOperand::bytes(type) == PTXOperand::bytes(sourceType);
			if( integerRounding
				&& !(sourceFloat && (destinationInt || sameSizeFloat)) ) {
				return "integer rounding is invalid for this cvt conversion";
			}
			if( floatRounding
				&& !(destinationFloat && (sourceInt || narrowerFloat)) ) {
				return "floating-point rounding is invalid for this cvt conversion";
			}
			if( sourceFloat && destinationInt && !integerRounding ) {
				return "float-to-integer cvt requires integer rounding";
			}
			if( sourceInt && destinationFloat && !floatRounding ) {
				return "integer-to-float cvt requires floating-point rounding";
			}
			if( narrowerFloat && !floatRounding ) {
				return "narrowing float cvt requires floating-point rounding";
			}
			if( modifier & rna ) {
				return ".rna is valid only for cvt.rna.tf32.f32 on sm_86";
			}
			if( modifier & relu ) {
				const int rounding = modifier & (rn | rz);
				if( type != PTXOperand::f16 || sourceType != PTXOperand::f32
					|| (rounding != rn && rounding != rz)
					|| (modifier & ~(rn | rz | relu)) ) {
					return ".relu requires cvt.{rn,rz}.f16.f32";
				}
			}
			if( modifier & sat ) {
				if( !destinationInt && type != PTXOperand::f16
					&& type != PTXOperand::f32 && type != PTXOperand::f64 ) {
					return ".sat is invalid for this cvt destination type";
				}
				if( destinationInt && sourceInt ) {
					const unsigned int destinationBits = PTXOperand::bytes(type) * 8;
					const unsigned int sourceBits = PTXOperand::bytes(sourceType) * 8;
					const bool destinationSigned = PTXOperand::isSigned(type);
					const bool sourceSigned = PTXOperand::isSigned(sourceType);
					const bool destinationSuperset = destinationSigned == sourceSigned
						? destinationBits >= sourceBits
						: destinationSigned && destinationBits > sourceBits;
					if( destinationSuperset ) {
						return ".sat is invalid when the destination range contains the source range";
					}
				}
			}
			if( d.bytes() < PTXOperand::bytes( type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned from " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( type != PTXOperand::f32 && sourceType != PTXOperand::f32 ) {
					return ".ftz requires an f32 source or destination";
				}
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != d.addressMode 
				&& !PTXOperand::relaxedValid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions.";
			}
			break;
		}
		case Cvta: {
			if (!(type == PTXOperand::u32 || type == PTXOperand::u64)) {
				return "invalid instruction type " + PTXOperand::toString(type);
			}
			if (!(addressSpace == Global || addressSpace == Local
				|| addressSpace == Shared || addressSpace == Const
				|| addressSpace == Param)) {
				return "invalid address space " + toString(addressSpace);
			}
			break;
		}
		case Div: {
			if( ( modifier & sat ) ) {
				return "no support for saturating divide.";
			}
			if( modifier & ~(rn | rz | rm | rp | approx | full | ftz) ) {
				return "invalid divide modifier";
			}
			const int rounding = modifier & (rn | rz | rm | rp);
			const bool oneRounding = rounding == rn || rounding == rz
				|| rounding == rm || rounding == rp;
			if( type == PTXOperand::f32 ) {
				const int modes = ((modifier & approx) ? 1 : 0)
					+ ((modifier & full) ? 1 : 0) + (rounding ? 1 : 0);
				if( modes != 1 || (rounding && !oneRounding) ) {
					return "div.f32 requires exactly one of approx, full, or rounding";
				}
			}
			if( type == PTXOperand::f64 ) {
				if( modifier & (approx | full | ftz) ) {
					return "approx, full, and ftz are invalid for div.f64";
				}
				if( !oneRounding ) {
					return "div.f64 requires exactly one rounding modifier";
				}
			}
			if( PTXOperand::isInt( type ) && modifier ) {
				return "integer divide does not accept modifiers";
			}
			if( !( type == PTXOperand::u16 || type == PTXOperand::u32 
				|| type == PTXOperand::u64 || type == PTXOperand::s16 
				|| type == PTXOperand::s32 || type == PTXOperand::s64 
				|| type == PTXOperand::f32 || type == PTXOperand::f64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Dp2a:
		case Dp4a: {
			if( (type != PTXOperand::u32 && type != PTXOperand::s32)
				|| (bType != PTXOperand::u32 && bType != PTXOperand::s32) ) {
				return "dp2a/dp4a type qualifiers must be .u32 or .s32";
			}
			if( opcode == Dp2a && modifier != lo && modifier != hi ) {
				return "dp2a instruction requires .lo or .hi mode";
			}
			const PTXOperand::DataType resultType = type == PTXOperand::u32
				&& bType == PTXOperand::u32 ? PTXOperand::u32 : PTXOperand::s32;
			if( !PTXOperand::valid(type, a.type)
				|| !PTXOperand::valid(bType, b.type)
				|| !PTXOperand::valid(resultType, c.type)
				|| !PTXOperand::valid(resultType, d.type) ) {
				return "dp2a/dp4a operands must have compatible 32-bit integer types";
			}
			break;
		}
		case Ex2: {
			if( !( type == PTXOperand::f32 || type == PTXOperand::f16 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( type == PTXOperand::f32 && modifier != approx
				&& modifier != ( approx | ftz ) ) {
				return "ex2.f32 requires .approx with optional .ftz";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Exit: {
			break;
		}
		case Fma: {
			if (!(type == ir::PTXOperand::f16 || type == ir::PTXOperand::f16x2
				|| type == ir::PTXOperand::f32
				|| type == ir::PTXOperand::f64 || type == ir::PTXOperand::bf16
				|| type == ir::PTXOperand::bf16x2)) {
				return "invalid instruction type " + PTXOperand::toString( type );
			}
			const int rounding = modifier & (rn | rz | rm | rp);
			if (rounding != rn && rounding != rz
				&& rounding != rm && rounding != rp) {
				return "fma requires exactly one rounding modifier";
			}
			if (type == PTXOperand::f64 && (modifier & (ftz | sat))) {
				return "ftz and sat modifiers are invalid for fma.f64";
			}
			if ((type == PTXOperand::f16 || type == PTXOperand::f16x2)
				&& rounding != rn) {
				return "fma.f16 requires .rn";
			}
			if ((type == PTXOperand::f16 || type == PTXOperand::f16x2)
				&& (modifier & sat) && (modifier & relu)) {
				return "fma.f16 sat and relu are mutually exclusive";
			}
			if ((type == PTXOperand::bf16 || type == PTXOperand::bf16x2)
				&& (rounding != rn || (modifier & (ftz | sat)))) {
				return "fma.bf16 requires .rn without ftz or sat";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type ) ) {
				return "operand C type " + PTXOperand::toString( c.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Fns: {
			if( type != PTXOperand::b32 ) {
				return "fns instruction only supports .b32 type";
			}
			if( d.type != PTXOperand::b32 ) {
				return "fns destination operand must have .b32 type";
			}
			if( a.bytes() != 4 ) {
				return "fns mask operand must be 32-bit";
			}
			if( b.type != PTXOperand::b32 && b.type != PTXOperand::u32
				&& b.type != PTXOperand::s32 ) {
				return "fns base operand must have .b32, .u32, or .s32 type";
			}
			if( c.type != PTXOperand::s32 ) {
				return "fns offset operand must have .s32 type";
			}
			if( b.addressMode == PTXOperand::Immediate && b.imm_uint > 31 ) {
				return "fns base operand must be between 0 and 31";
			}
			break;
		}
		case Isspacep: {
			if (!(addressSpace == PTXInstruction::Const
				|| addressSpace == PTXInstruction::Global
				|| addressSpace == PTXInstruction::Param
				|| addressSpace == PTXInstruction::Shared
				|| addressSpace == PTXInstruction::Local)) {
				return "invalid address space " + toString(addressSpace);
			}
			if (!(d.addressMode == PTXOperand::Register
				&& d.type == PTXOperand::pred && d.vec == PTXOperand::v1
				&& a.addressMode == PTXOperand::Register && a.vec == PTXOperand::v1
				&& (a.type == PTXOperand::u32 || a.type == PTXOperand::u64))) {
				return "invalid address mode for operands";
			}
			break;
		}
		case Ldmatrix: {
			if (type != PTXOperand::b16 || modifier ||
				(addressSpace != Shared && addressSpace != Generic)) return "invalid ldmatrix type/modifier/space";
			if ((vec != PTXOperand::v1 && vec != PTXOperand::v2 && vec != PTXOperand::v4) ||
				d.addressMode != PTXOperand::Register || d.vec != vec || d.array.size() != unsigned(vec))
				return "ldmatrix requires one 32-bit destination register per matrix";
			for (const auto& element : d.array) {
				if (element.addressMode != PTXOperand::Register || element.bytes() != 4 ||
					element.vec != PTXOperand::v1 || !element.array.empty()) return "invalid ldmatrix destination leaf";
			}
			if ((a.addressMode != PTXOperand::Indirect && a.addressMode != PTXOperand::Register &&
				a.addressMode != PTXOperand::Address && a.addressMode != PTXOperand::Immediate) ||
				a.vec != PTXOperand::v1 || !a.array.empty() ||
				(a.addressMode != PTXOperand::Address && ((!PTXOperand::isInt(a.type) &&
				a.type != PTXOperand::b32 && a.type != PTXOperand::b64) ||
				(a.bytes() != 4 && a.bytes() != 8)))) return "ldmatrix requires a scalar 32/64-bit address";
			break;
		}
		case Ldu: // fall through
		case Ld: {
			if( !( a.addressMode == PTXOperand::Register 
				|| a.addressMode == PTXOperand::Address 
				|| a.addressMode == PTXOperand::Indirect 
				|| a.addressMode == PTXOperand::Immediate ) ) {
				return "invalid address mode " 
					+ PTXOperand::toString( a.addressMode ) 
					+ " for operand A ";
			}
			if( addressSpace == AddressSpace_Invalid ) {
				return "invalid address space";
			}
			if( addressSpace != Global && addressSpace != Shared 
				&& volatility == Volatile && addressSpace != Generic ) {
				return "only shared and global address spaces supported "
					"for volatile loads";
			}
			if( cacheOperation == Lu && volatility == Volatile ) {
				return "ld.lu cannot be combined with .volatile";
			}
			if( d.addressMode != PTXOperand::Register ) {
				return "operand D must be a register not a " 
					+ PTXOperand::toString( d.addressMode );
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != d.addressMode 
				&& !PTXOperand::relaxedValid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions: -\n" 
					+ toString() + "\nd.addressMode: "
					+ PTXOperand::toString(d.addressMode);
			}
			break;
		}
		case Lg2: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( modifier != approx && modifier != ( approx | ftz ) ) {
				return "lg2 requires .approx with optional .ftz";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Mad24: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Mad: {
			if( type == PTXOperand::f32 || type == PTXOperand::f64 ) {
				const int rounding = modifier & (rn | rz | rm | rp);
				if( rounding != rn && rounding != rz
					&& rounding != rm && rounding != rp ) {
					return "mad requires exactly one rounding modifier";
				}
				if( type == PTXOperand::f64 && (modifier & (ftz | sat)) ) {
					return "ftz and sat modifiers are invalid for mad.f64";
				}
				if( !PTXOperand::valid( type, a.type ) ) {
					return "operand A type " + PTXOperand::toString( a.type )
						+ " cannot be assigned to " + PTXOperand::toString( type );
				}
				if( !PTXOperand::valid( type, b.type ) ) {
					return "operand B type " + PTXOperand::toString( b.type )
						+ " cannot be assigned to " + PTXOperand::toString( type );
				}
				if( !PTXOperand::valid( type, c.type ) ) {
					return "operand C type " + PTXOperand::toString( c.type )
						+ " cannot be assigned to " + PTXOperand::toString( type );
				}
				if( !PTXOperand::valid( type, d.type ) ) {
					return "operand D type " + PTXOperand::toString( d.type )
						+ " cannot be assigned to " + PTXOperand::toString( type );
				}
				break;
			}
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( type == PTXOperand::f64 ) {
				if( !( modifier & rn ) && !( modifier & rz ) 
					&& !( modifier & rm ) && !( modifier & rp ) ) {
					return "requires a rounding modifier";
				}
			}
			if( !PTXOperand::valid(a.type, b.type) ) {
				return "type of operand A " + PTXOperand::toString( a.type ) 
					+ " is not interoperable with type of operand B "
					+ PTXOperand::toString( b.type )
					+ ", conversion is required";
			}
			if( !( c.bytes() == d.bytes() ) ) {
				std::stringstream stream;
				stream << "size of operand C " << c.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( ( modifier & hi ) || ( modifier & lo ) ) 
				&& ( a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "not wide and size of operand A " << a.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( modifier & wide ) && ( 2 * a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "wide and size of operand A " << a.bytes() 
					<< " does not equal half the size of operand D " 
					<< d.bytes();
				return stream.str();
			}
			if( ( modifier & sat ) && ( type != PTXOperand::s32
				&& type != PTXOperand::f32 ) ) {
				return "saturate only valid for s32/f32";
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Mma: {
			if (mmaSparse) {
				const bool sparseTf32 = (mmaShape == MmaM16N8K8 || mmaShape == MmaM16N8K16) && a.type == PTXOperand::tf32;
				const bool sparseInt8 = (mmaShape == MmaM16N8K32 || mmaShape == MmaM16N8K64) &&
					(a.type == PTXOperand::s8 || a.type == PTXOperand::u8);
				const bool sparseInt4 = (mmaShape == MmaM16N8K64 || mmaShape == MmaM16N8K128) &&
					(a.type == PTXOperand::s4 || a.type == PTXOperand::u4);
				const bool sparseHalf = (mmaShape == MmaM16N8K16 || mmaShape == MmaM16N8K32) &&
					(a.type == PTXOperand::f16 || a.type == PTXOperand::bf16);
				if (!sparseTf32 && !sparseHalf && !sparseInt8 && !sparseInt4) {
					return "unsupported sparse mma shape/type combination";
				}
				if (q.addressMode != PTXOperand::Register || q.vec != PTXOperand::v1 ||
					!q.array.empty() || (q.type != PTXOperand::b32 &&
					q.type != PTXOperand::u32 && q.type != PTXOperand::s32)) {
					return "sparse mma metadata must be a scalar 32-bit integer register";
				}
				const unsigned maxSelector = (sparseInt8 && mmaShape == MmaM16N8K64) ||
					(sparseInt4 && mmaShape == MmaM16N8K128) ? 0u :
					mmaShape == MmaM16N8K32 || sparseInt4 || (mmaShape == MmaM16N8K16 && sparseTf32) ? 1u : 3u;
				if (mmaSelector.addressMode != PTXOperand::Immediate ||
					!PTXOperand::isInt(mmaSelector.type) ||
					mmaSelector.imm_uint > maxSelector) {
					return "sparse mma selector must be a legal integer constant for its shape";
				}
				for (const auto* operand : {&c, &d}) {
					for (const auto& element : operand->array) {
						if (element.bytes() != 4) return "sparse mma C/D registers must be 32-bit";
					}
				}
			}
			for (const PTXOperand* operand : {&a, &b, &c, &d}) {
				if (operand->addressMode != PTXOperand::Register) {
					return "mma fragments must be register operands";
				}
				for (const auto& element : operand->array) {
					if (element.addressMode != PTXOperand::Register) {
						return "mma fragment elements must be registers";
					}
					if (element.vec != PTXOperand::v1 || !element.array.empty()) {
						return "mma fragment elements must be scalar registers";
					}
				}
			}
			if ((mmaAColumnMajor || !mmaBColumnMajor) &&
				(mmaShape != MmaM8N8K4 || a.type != PTXOperand::f16)) {
				return "this mma form requires row.col layouts";
			}
			if (a.type == PTXOperand::b1) {
				const bool small = mmaShape == MmaM8N8K128;
				const bool k256 = mmaShape == MmaM16N8K256;
				if ((!small && !k256 && mmaShape != MmaM16N8K128) ||
					type != PTXOperand::s32 || b.type != a.type || c.type != type || d.type != type || modifier ||
					(booleanOperator != BoolAnd && booleanOperator != BoolXor)) return "invalid binary mma shape/type/modifier";
				const unsigned aCount = small ? 1u : k256 ? 4u : 2u;
				const unsigned bCount = k256 ? 2u : 1u;
				const unsigned cdCount = small ? 2u : 4u;
				for (const PTXOperand* operand : {&a, &b, &c, &d}) {
					const bool input = operand == &a || operand == &b;
					const unsigned count = operand == &a ? aCount : operand == &b ? bCount : cdCount;
					if (operand->array.size() != count || operand->vec != static_cast<PTXOperand::Vec>(count)) return "invalid binary mma fragment size";
					for (const auto& element : operand->array)
						if (input ? element.type != PTXOperand::b32 : !PTXOperand::relaxedValid(type, element.type)) return "invalid binary mma register type";
				}
				break;
			}
			if (a.type == PTXOperand::f64) {
				if (mmaShape != MmaM8N8K4 || type != PTXOperand::f64 ||
					b.type != type || c.type != type || d.type != type ||
					(modifier & ~(rn | rz | rm | rp))) return "invalid f64 mma shape/type/modifier";
				if (modifier && (modifier & (modifier - 1))) {
					return "f64 mma accepts only one rounding modifier";
				}
				for (const PTXOperand* operand : {&a, &b, &c, &d}) {
					const int count = (operand == &a || operand == &b) ? 1 : 2;
					if (operand->array.size() != count || operand->vec != (count == 1 ? PTXOperand::v1 : PTXOperand::v2))
						return "invalid f64 mma fragment size";
					for (const auto& element : operand->array)
						if (!PTXOperand::relaxedValid(type, element.type)) return "invalid f64 mma register type";
				}
				break;
			}
			if (mmaShape == MmaM8N8K4) {
				if ((type != PTXOperand::f16 && type != PTXOperand::f32) || a.type != PTXOperand::f16 ||
					b.type != a.type || (c.type != PTXOperand::f16 && c.type != PTXOperand::f32) ||
					d.type != type || (c.type == PTXOperand::f32 && type != PTXOperand::f32) || modifier)
					return "unsupported m8n8k4 type/modifier combination";
				for (const PTXOperand* operand : {&a, &b, &c, &d}) {
					const bool input = operand == &a || operand == &b;
					const unsigned count = input ? 2 : operand->type == PTXOperand::f32 ? 8 : 4;
					if (operand->vec != static_cast<PTXOperand::Vec>(count) || operand->array.size() != count)
						return "invalid m8n8k4 f16 fragment size";
					for (const auto& element : operand->array)
						if (input || operand->type == PTXOperand::f16 ?
							(element.type != PTXOperand::b32 && element.type != PTXOperand::f16x2) :
							!PTXOperand::relaxedValid(PTXOperand::f32, element.type)) return "invalid mma fragment register type";
				}
				break;
			}
			if (a.type == PTXOperand::s4 || a.type == PTXOperand::u4) {
				const bool small = mmaShape == MmaM8N8K32;
				const bool k64 = mmaShape == MmaM16N8K64;
				const bool sparseK128 = mmaSparse && mmaShape == MmaM16N8K128;
				if ((!small && !k64 && !sparseK128 && mmaShape != MmaM16N8K32) || (b.type != PTXOperand::s4 && b.type != PTXOperand::u4) ||
					type != PTXOperand::s32 || c.type != type || d.type != type || (modifier & ~satfinite))
					return "unsupported sub-byte mma shape/type/modifier combination";
				const unsigned aCount = mmaSparse ? (sparseK128 ? 4u : 2u) : small ? 1u : k64 ? 4u : 2u;
				const unsigned bCount = mmaSparse ? aCount : k64 ? 2u : 1u;
				const unsigned cdCount = small ? 2u : 4u;
				for (const PTXOperand* operand : {&a, &b, &c, &d}) {
					const bool input = operand == &a || operand == &b;
					const unsigned count = operand == &a ? aCount : operand == &b ? bCount : cdCount;
					if (operand->vec != static_cast<PTXOperand::Vec>(count) || operand->array.size() != count)
						return "invalid sub-byte mma fragment size";
					for (const auto& element : operand->array)
						if (input ? element.type != PTXOperand::b32 : !PTXOperand::relaxedValid(type, element.type)) return "invalid sub-byte mma register type";
				}
				break;
			}
			const bool intInput = a.type == PTXOperand::s8 ||
				a.type == PTXOperand::u8;
			if (intInput) {
				if (modifier & ~satfinite) {
					return "integer mma accepts only satfinite";
				}
				if (mmaShape != MmaM16N8K16 && mmaShape != MmaM8N8K16 &&
					mmaShape != MmaM16N8K32 && !(mmaSparse && mmaShape == MmaM16N8K64)) {
					return "integer mma requires a supported shape";
				}
				if (type != PTXOperand::s32) {
					return "integer mma requires s32 accumulators";
				}
				if (b.type != PTXOperand::s8 && b.type != PTXOperand::u8) {
					return "integer mma B type must be s8 or u8";
				}
				if (d.type != PTXOperand::s32 || c.type != PTXOperand::s32) {
					return "integer mma C and D types must be s32";
				}
				const unsigned int aCount = mmaSparse ? (mmaShape == MmaM16N8K64 ? 4u : 2u) : mmaShape == MmaM16N8K32
					? 4u : mmaShape == MmaM8N8K16 ? 1u : 2u;
				const unsigned int bCount = mmaSparse ? aCount : mmaShape == MmaM8N8K16
					? 1u : mmaShape == MmaM16N8K32 ? 2u : 1u;
				const unsigned int cdCount = mmaShape == MmaM8N8K16 ? 2u : 4u;
				const auto aVec = static_cast<PTXOperand::Vec>(aCount), bVec = static_cast<PTXOperand::Vec>(bCount);
				const auto cdVec = static_cast<PTXOperand::Vec>(cdCount);
				if (d.vec != cdVec || c.vec != cdVec ||
					a.vec != aVec || b.vec != bVec) {
					return "integer mma has invalid fragment sizes";
				}
				if (d.array.size() != cdCount || c.array.size() != cdCount ||
					a.array.size() != aCount || b.array.size() != bCount) {
					return "integer mma has invalid fragment register counts";
				}
				for (PTXOperand::Array::const_iterator element = a.array.begin();
					element != a.array.end(); ++element) {
					if (element->type != PTXOperand::b32) {
						return "mma A fragment registers must be 32-bit "
							"packed values";
					}
				}
				for (PTXOperand::Array::const_iterator element = b.array.begin();
					element != b.array.end(); ++element) {
					if (element->type != PTXOperand::b32) {
						return "mma B fragment registers must be 32-bit "
							"packed values";
					}
				}
				for (PTXOperand::Array::const_iterator element = c.array.begin();
					element != c.array.end(); ++element) {
					if (!PTXOperand::relaxedValid(PTXOperand::s32, element->type)) {
						return "mma C fragment registers must be s32 or b32";
					}
				}
				for (PTXOperand::Array::const_iterator element = d.array.begin();
					element != d.array.end(); ++element) {
					if (!PTXOperand::relaxedValid(PTXOperand::s32, element->type)) {
						return "mma D fragment registers must be s32 or b32";
					}
				}
				break;
			}
			if (modifier) {
				return "f16/bf16/tf32 mma does not accept modifiers";
			}
			const bool m16n8k8 = mmaShape == MmaM16N8K8;
			const bool m16n8k4 = mmaShape == MmaM16N8K4;
			const bool sparseK32 = mmaSparse && mmaShape == MmaM16N8K32;
			const bool tf32Input = a.type == PTXOperand::tf32;
			const bool sparseTf32K16 = mmaSparse && tf32Input && mmaShape == MmaM16N8K16;
			const bool halfAccumulator = type == PTXOperand::f16;
			if ((m16n8k4 && !tf32Input) ||
				(!m16n8k4 && !m16n8k8 && mmaShape != MmaM16N8K16 && !sparseK32)) {
				return "unsupported floating-point mma shape/type combination";
			}
			if (!halfAccumulator && type != PTXOperand::f32) {
				return "mma requires f16 or f32 accumulators";
			}
			if (a.type != PTXOperand::f16 && a.type != PTXOperand::bf16 &&
				a.type != PTXOperand::tf32) {
				return "mma A type must be f16, bf16, or tf32";
			}
			if (tf32Input && ((!m16n8k8 && !m16n8k4 && !sparseTf32K16) || type != PTXOperand::f32 ||
				b.type != PTXOperand::tf32)) {
				return "tf32 mma requires a supported shape with f32 accumulators";
			}
			if (halfAccumulator && a.type != PTXOperand::f16) {
				return "f16 mma accumulators require f16 inputs";
			}
			if (b.type != a.type) {
				return "mma A and B types must match";
			}
			if (d.type != type || c.type != type) {
				return halfAccumulator ? "mma C and D types must be f16"
					: "mma C and D types must be f32";
			}
			const PTXOperand::Vec accumulatorVec = halfAccumulator
				? PTXOperand::v2 : PTXOperand::v4;
			const unsigned int accumulatorRegisters = halfAccumulator ? 2 : 4;
			const bool compactInputFragment = m16n8k4 || (m16n8k8 && !tf32Input);
			const bool compactA = compactInputFragment || (mmaSparse && !sparseK32 && !sparseTf32K16);
			const PTXOperand::Vec inputAVec = compactA
				? PTXOperand::v2 : PTXOperand::v4;
			const PTXOperand::Vec inputBVec = (sparseK32 || sparseTf32K16) ? PTXOperand::v4 : compactInputFragment
				? PTXOperand::v1 : PTXOperand::v2;
			if (d.vec != accumulatorVec || c.vec != accumulatorVec ||
				a.vec != inputAVec || b.vec != inputBVec) {
				return "mma has invalid fragment sizes";
			}
			if (d.array.size() != accumulatorRegisters ||
				c.array.size() != accumulatorRegisters ||
				a.array.size() != (compactA ? 2u : 4u) ||
				b.array.size() != (sparseK32 || sparseTf32K16 ? 4u : compactInputFragment ? 1u : 2u)) {
				return "mma has invalid fragment register counts";
			}
			for (PTXOperand::Array::const_iterator element = a.array.begin();
				element != a.array.end(); ++element) {
				if (element->type != PTXOperand::b32 &&
					(tf32Input || element->type != PTXOperand::f16x2)) {
					return "mma A fragment registers must be 32-bit packed values";
				}
			}
			for (PTXOperand::Array::const_iterator element = b.array.begin();
				element != b.array.end(); ++element) {
				if (element->type != PTXOperand::b32 &&
					(tf32Input || element->type != PTXOperand::f16x2)) {
					return "mma B fragment registers must be 32-bit packed values";
				}
			}
			for (PTXOperand::Array::const_iterator element = c.array.begin();
				element != c.array.end(); ++element) {
				if (halfAccumulator ? (element->type != PTXOperand::b32 &&
					element->type != PTXOperand::f16x2)
					: !PTXOperand::relaxedValid(PTXOperand::f32, element->type)) {
					return halfAccumulator ? "mma C fragment registers must be b32 or f16x2"
						: "mma C fragment registers must be f32 or b32";
				}
			}
			for (PTXOperand::Array::const_iterator element = d.array.begin();
				element != d.array.end(); ++element) {
				if (halfAccumulator ? (element->type != PTXOperand::b32 &&
					element->type != PTXOperand::f16x2)
					: !PTXOperand::relaxedValid(PTXOperand::f32, element->type)) {
					return halfAccumulator ? "mma D fragment registers must be b32 or f16x2"
						: "mma D fragment registers must be f32 or b32";
				}
			}
			break;
		}
		case MadC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !( ( modifier & hi ) || ( modifier & lo ) ) ) {
				return "modifier must be hi or lo";
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Max: {
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if (type == PTXOperand::b16 || type == PTXOperand::b32 ||
				type == PTXOperand::b64) {
				return "invalid instruction type " + PTXOperand::toString(type);
			}
			if ((type == PTXOperand::u16 || type == PTXOperand::u32 ||
				type == PTXOperand::u64 || type == PTXOperand::s16 ||
				type == PTXOperand::s32 || type == PTXOperand::s64) &&
				(modifier || carry != None)) {
				return "integer max does not support modifiers or carry out";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Membar: {
			break;
		}
		case Fence: {
			break;
		}
		case Min: {
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if (type == PTXOperand::b16 || type == PTXOperand::b32 ||
				type == PTXOperand::b64) {
				return "invalid instruction type " + PTXOperand::toString(type);
			}
			if ((type == PTXOperand::u16 || type == PTXOperand::u32 ||
				type == PTXOperand::u64 || type == PTXOperand::s16 ||
				type == PTXOperand::s32 || type == PTXOperand::s64) &&
				(modifier || carry != None)) {
				return "integer min does not support modifiers or carry out";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Match: {
			if ((type != PTXOperand::b32 && type != PTXOperand::b64) || modifier || vec != PTXOperand::v1) return "invalid match type/modifier";
			for (const auto* operand : {&d, &a, &b, &pq})
				if (operand->vec != PTXOperand::v1 || !operand->array.empty()) return "match requires scalar operands";
			if (!(d.addressMode == PTXOperand::Register && d.bytes() == 4) &&
				!(matchAll && d.addressMode == PTXOperand::BitBucket)) return "match requires a 32-bit destination";
			if ((a.addressMode != PTXOperand::Register && a.addressMode != PTXOperand::Immediate) ||
				a.bytes() != PTXOperand::bytes(type)) return "match source must have instruction width";
			if ((b.addressMode != PTXOperand::Register && b.addressMode != PTXOperand::Immediate) || b.bytes() != 4 ||
				(!PTXOperand::isInt(b.type) && b.type != PTXOperand::b32)) return "match requires a 32-bit integer member mask";
			if (pq.addressMode != PTXOperand::Invalid && (!matchAll ||
				!(pq.addressMode == PTXOperand::BitBucket || (pq.addressMode == PTXOperand::Register &&
				pq.type == PTXOperand::pred && pq.condition == PTXOperand::Pred)))) return "invalid match predicate destination";
			if (d.addressMode == PTXOperand::BitBucket && pq.addressMode == PTXOperand::BitBucket) return "match cannot discard both destinations";
			break;
		}
		case Redux: {
			const bool bitwise = reductionOperation == ReductionAnd || reductionOperation == ReductionOr || reductionOperation == ReductionXor;
			const bool arithmetic = reductionOperation == ReductionAdd || reductionOperation == ReductionMin || reductionOperation == ReductionMax;
			if (modifier || vec != PTXOperand::v1 || !(bitwise ? type == PTXOperand::b32 :
				arithmetic && (type == PTXOperand::s32 || type == PTXOperand::u32))) return "invalid redux operation/type/modifier";
			for (const auto* operand : {&d, &a, &b})
				if (operand->vec != PTXOperand::v1 || !operand->array.empty() || operand->bytes() != 4) return "redux requires scalar 32-bit operands";
			for (const auto* operand : {&d, &a})
				if (operand->addressMode != PTXOperand::Register || (!bitwise && !PTXOperand::valid(type, operand->type))) return "redux requires compatible destination/source registers";
			if ((b.addressMode != PTXOperand::Register && b.addressMode != PTXOperand::Immediate) ||
				(!PTXOperand::isInt(b.type) && b.type != PTXOperand::b32)) return "redux requires a 32-bit integer member mask";
			break;
		}
		case Activemask: {
			if (type != PTXOperand::b32 || modifier || vec != PTXOperand::v1 ||
				d.addressMode != PTXOperand::Register || d.bytes() != 4 ||
				d.vec != PTXOperand::v1 || !d.array.empty()) return "activemask requires a scalar 32-bit register";
			break;
		}
		case Movmatrix: {
			if (type != PTXOperand::b16 || modifier || !matrixTranspose || vec != PTXOperand::v1)
				return "invalid movmatrix type/modifier/transpose/count";
			for (const auto* operand : {&d, &a}) {
				if (operand->addressMode != PTXOperand::Register || operand->bytes() != 4 ||
					operand->vec != PTXOperand::v1 || !operand->array.empty())
					return "movmatrix requires scalar 32-bit register operands";
			}
			break;
		}
		case Mov: {
			if ( type != PTXOperand::b16 &&
				a.type == PTXOperand::f16 &&
				a.array.empty() &&
				a.addressMode != PTXOperand::Address &&
				a.addressMode != PTXOperand::Immediate ) {
				return "invalid type for operand A "
					+ PTXOperand::toString( a.type );
			}
			if ( !( d.type != PTXOperand::s8 && d.type != PTXOperand::u8 
				&& d.type != PTXOperand::b8 ) ) {
				return "invalid type for operand D " 
					+ PTXOperand::toString( d.type );
			}
			if( ( a.vec != PTXOperand::v1 || d.vec != PTXOperand::v1 ) 
				&& ( a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "at least one vector operand and A size " << a.bytes()
					<< " does not equal D size " << d.bytes();
				return stream.str();
			}
			break;
		}
		case Mul24: {
			if( type != PTXOperand::u32 && type != PTXOperand::s32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !( modifier & lo ) && !( modifier & hi ) ) {
				return "must be either lo or hi";
			}
			break;
		}
		case Mul: {
			if( type == PTXOperand::s8 || type == PTXOperand::u8 
				|| type == PTXOperand::b8
				|| type == PTXOperand::pred ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if ((type == PTXOperand::f16 || type == PTXOperand::f16x2)
				&& (modifier & ~(rn | ftz | sat))) {
				return "mul.f16 only supports .rn, .ftz, and .sat";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( PTXOperand::isInt( type ) && ( !( modifier & lo ) 
				&& !( modifier & hi ) && !( modifier & wide ) ) ) {
				return "int operations must be hi, lo, or wide";
			}
			if( ( ( modifier & lo ) || ( modifier & hi ) ) 
				&& a.bytes() != d.bytes() ) {
				std::stringstream stream;
				stream << "for lo and hi, size of operand A " << a.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( modifier & wide ) && ( 2 * a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "wide, size of operand A " << a.bytes() 
					<< " does not equal half the size of operand D " 
					<< d.bytes();
				return stream.str();
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Neg: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 && 
				type != PTXOperand::s64 && type != PTXOperand::f32 && 
				type != PTXOperand::f64 && type != PTXOperand::f16 &&
				type != PTXOperand::f16x2 && type != PTXOperand::bf16x2 &&
				type != PTXOperand::bf16 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if ((type == PTXOperand::s16 || type == PTXOperand::s32 ||
				type == PTXOperand::s64) && (modifier || carry != None)) {
				return "integer neg does not support modifiers or carry out";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Nop: {
			return "NOP is not a valid instruction.";
			break;
		}
		case Not: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Pmevent: {
			if( d.addressMode != PTXOperand::Immediate ) {
				return "only support Immediate targets";
			}
			break;
		}
		case Popc: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if (modifier || carry != None) {
				return "popc does not support modifiers or carry out";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Prefetch: {
			if (!(cacheLevel == L1 || cacheLevel == L2)) {
				return "cache level must be L1 or L2";
			}
			if (!(addressSpace == Local || addressSpace == Global)) {
				return "address space must be .local or .global, not " + toString(addressSpace);
			}
			if (!(d.addressMode == PTXOperand::Indirect || d.addressMode == PTXOperand::Address ||
				d.addressMode == PTXOperand::Immediate)) {
				
				return "address mode of destination operand must be Indirect, Address, or Immediate. Not " +
					PTXOperand::toString(d.addressMode);
			}
		}
		break;
		case Prefetchu: {
		
			if (!(cacheLevel == L1)) {
				return "cache level must be L1, not " + toString(cacheLevel);
			}
			if (!(d.addressMode == PTXOperand::Indirect || d.addressMode == PTXOperand::Address ||
				d.addressMode == PTXOperand::Immediate)) {
				
				return "address mode of destination operand must be Indirect, Address, or Immediate. Not " +
					PTXOperand::toString(d.addressMode);
			}
		}
		break;
		case Prmt: {
			if( type != PTXOperand::b32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Or: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Rcp: {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			const int rounding = modifier & (rn | rz | rm | rp);
			if( modifier & ~(approx | ftz | rn | rz | rm | rp) ) {
				return "invalid modifier for rcp";
			}
			if( type == PTXOperand::f64 && (modifier & approx) ) {
				if( modifier != (approx | ftz) ) {
					return "rcp.approx.f64 requires .ftz and no rounding modifier";
				}
			} else {
				if( modifier & approx ) {
					if( rounding ) return "rcp.approx.f32 cannot specify rounding";
				} else if( rounding != rn && rounding != rz
					&& rounding != rm && rounding != rp ) {
					return "rcp requires exactly one rounding modifier";
				}
				if( type == PTXOperand::f64 && (modifier & ftz) ) {
					return "rcp.rnd.f64 does not support .ftz";
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Red: {
			if( ( reductionOperation == ReductionAnd
				|| reductionOperation == ReductionOr
				|| reductionOperation == ReductionXor )
				&& type != PTXOperand::b32 && type != PTXOperand::b64 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for reduction "
					+ toString( reductionOperation );
			}
			if( reductionOperation == ReductionAdd
				&& ( type != PTXOperand::u32 && type != PTXOperand::s32
				&& type != PTXOperand::u64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for reduction "
					+ toString( reductionOperation );
			}
			if( ( reductionOperation == ReductionInc
				|| reductionOperation == ReductionDec )
				&& type != PTXOperand::u32 ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for reduction "
					+ toString( reductionOperation );
			}
			if( ( reductionOperation == ReductionMin
				|| reductionOperation == ReductionMax )
				&& ( type != PTXOperand::u32 && type != PTXOperand::s32
				&& type != PTXOperand::u64 && type != PTXOperand::s64 ) ) {
				return "invalid instruction type "
					+ PTXOperand::toString( type ) + " for reduction "
					+ toString( reductionOperation );
			}
			if( a.addressMode != PTXOperand::Address 
				&& a.addressMode != PTXOperand::Register 
				&& a.addressMode != PTXOperand::Indirect
				&& a.addressMode != PTXOperand::Immediate ) {
				return "operand A must be an address";
			}
			if( addressSpace != Shared && addressSpace != Global ) {
				return "address space much be either shared or global";
			}
			break;
		}
		case Rem: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;		
		}
		case Ret: {
			break;
		}
		case Rsqrt: {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( modifier != approx && modifier != (approx | ftz) ) {
				return "rsqrt requires .approx with optional .ftz";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;			
		}
		case Sad: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;		
		}
		case SelP: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( c.type != PTXOperand::pred ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			break;
		}
		case Set: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 && type != PTXOperand::f16 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.type != PTXOperand::s32 && d.type != PTXOperand::f32
				&& d.type != PTXOperand::u32 && d.type != PTXOperand::b32
				&& d.type != PTXOperand::b16 && d.type != PTXOperand::f16 ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " invalid (must be b16, f16, b32, u32, s32, or f32)";
			}
			if( c.type != PTXOperand::pred && 
				c.addressMode != PTXOperand::Invalid ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( a.type ) ) {
					return " .ftz only valid when source is .f32.";
				}
			}
			break;
		}
		case SetP: {
			if( d.type != PTXOperand::pred
				&& d.addressMode != PTXOperand::BitBucket ) {
				return "destination must be a predicate";
			}
			if( pq.type != PTXOperand::pred 
				&& pq.addressMode != PTXOperand::Invalid
				&& pq.addressMode != PTXOperand::BitBucket ) {
				return "Pq must be a predicate";
			}
			if( d.addressMode == PTXOperand::BitBucket
				&& pq.addressMode == PTXOperand::BitBucket ) {
				return "only one setp destination may be a sink";
			}
			if( c.type != PTXOperand::pred 
				&& c.addressMode != PTXOperand::Invalid ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 && type != PTXOperand::f16 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;			
		}
		case Shfl: {
			if( type != PTXOperand::b32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.bytes() != a.bytes() 
				&& a.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of operand D " << d.bytes();
				return stream.str(); 
			}
			if( b.bytes() != 4 && b.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand B " << b.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			if( c.bytes() != 4 && c.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand C " << b.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			break;
		}
		case Shf: {
			if ( type != PTXOperand::b32 ) {
				return "shf instruction only supports .b32 type";
			}
			if ( d.bytes() != 4 ) {
				std::stringstream stream;
				stream << "destination operand must be 32-bit, got " << d.bytes() << " bytes";
				return stream.str();
			}
			if ( a.bytes() != 4 && a.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "first source operand must be 32-bit, got " << a.bytes() << " bytes";
				return stream.str();
			}
			if ( b.bytes() != 4 && b.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "second source operand must be 32-bit, got " << b.bytes() << " bytes";
				return stream.str();
			}
			if ( c.addressMode != PTXOperand::Immediate
				&& !PTXOperand::valid( PTXOperand::u32, c.type ) ) {
				return "shift amount must have a 32-bit integer type";
			}
			break;
		}
		case Shl: {
			if( type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( a.addressMode != PTXOperand::Immediate
				&& !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( b.addressMode != PTXOperand::Immediate
				&& !PTXOperand::valid( PTXOperand::u32, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type )
					+ " cannot be assigned to u32";
			}
			break;
		}
		case Shr: {
			if( type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::s16 
				&& type != PTXOperand::s32 && type != PTXOperand::s64 
				&& type != PTXOperand::u16 && type != PTXOperand::u32 
				&& type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( a.addressMode != PTXOperand::Immediate
				&& !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type )
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( b.addressMode != PTXOperand::Immediate
				&& !PTXOperand::valid( PTXOperand::u32, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type )
					+ " cannot be assigned to u32";
			}
			break;
		}
		case Sin: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( modifier != approx && modifier != ( approx | ftz ) ) {
				return "sin requires .approx with optional .ftz";
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case SlCt: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			unsigned int bytes = PTXOperand::bytes( type );
			if( bytes != a.bytes() ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( bytes != b.bytes() ) {
				std::stringstream stream;
				stream << "size of operand B " << b.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( bytes != d.bytes() ) {
				std::stringstream stream;
				stream << "size of operand D " << d.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( !PTXOperand::valid( PTXOperand::f32, c.type ) 
				&& !PTXOperand::valid( PTXOperand::s32, c.type ) ) {
				return "operand C must be either s32 or f32 assignable";
			}
			if( modifier & ftz ) {
				if( c.type != PTXOperand::f32 ) {
					return toString( ftz ) 
						+ " only valid for an f32 comparison.";
				}
			}
			break;
			
		}
		case Sqrt: {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			const int rounding = modifier & (rn | rz | rm | rp);
			if( modifier & ~(approx | ftz | rn | rz | rm | rp) ) {
				return "invalid modifier for sqrt";
			}
			if( modifier & approx ) {
				if( type != PTXOperand::f32 ) {
					return "sqrt.approx only supports f32";
				}
				if( rounding ) return "sqrt.approx cannot specify rounding";
			} else if( rounding != rn && rounding != rz
				&& rounding != rm && rounding != rp ) {
				return "sqrt requires exactly one rounding modifier";
			}
			if( type == PTXOperand::f64 && (modifier & ftz) ) {
				return "sqrt.rnd.f64 does not support .ftz";
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;			
		}
		case St: {
			if( d.addressMode != PTXOperand::Register 
				&& d.addressMode != PTXOperand::Address 
				&& d.addressMode != PTXOperand::Indirect 
				&& d.addressMode != PTXOperand::Immediate ) {
				return "invalid address mode " 
					+ PTXOperand::toString( d.addressMode ) 
					+ " for operand D ";
			}
			if( addressSpace == AddressSpace_Invalid ) {
				return "invalid address space";
			}
			if( ( addressSpace != Global && addressSpace != Shared
				&& addressSpace != Generic ) 
				&& volatility == Volatile ) {
				return "only shared and global address spaces supported " 
					"for volatile stores";
			}
			if( a.addressMode != PTXOperand::Register
				&& a.addressMode != PTXOperand::Immediate ) {
				return "operand A must be a register or immediate";
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != a.addressMode 
				&& !PTXOperand::relaxedValid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions.";
			}
			break;
		}
		case Sub: {
			if ( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if ((type == PTXOperand::f16 || type == PTXOperand::f16x2)
				&& (modifier & ~(rn | ftz | sat))) {
				return "sub.f16 only supports .rn, .ftz, and .sat";
			}
			if( carry == CC ) {
				if( ( modifier & sat ) ) {
					return "saturate not supported with carry out";
				}
				if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
					return "invalid instruction type " 
						+ PTXOperand::toString( type );
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case SubC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Tanh: {
			if( !(type == PTXOperand::f32 || type == PTXOperand::f16
				|| type == PTXOperand::f16x2) || modifier != approx ) {
				return "tanh requires .approx with f32, f16, or f16x2";
			}
			if( !PTXOperand::valid(type, a.type)
				|| !PTXOperand::valid(type, d.type) ) {
				return "tanh operands must have compatible types";
			}
			break;
		}
		case TestP: {
			if( !( type == PTXOperand::f32 || type == PTXOperand::f64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( d.type != ir::PTXOperand::pred ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " should be a predicate instead.";
			}
			break;
		}
		case Tex: {
			if( geometry == _2d && c.vec == PTXOperand::v1 ) {
				return "for 2d tectures, C must be a at least a 2d vector";
			}
			if( geometry == _3d && c.vec != PTXOperand::v2 
				&& c.vec != PTXOperand::v4  ) {
				return "for 3d textures, C must be at least a 4d vector";
			}
			if( !PTXOperand::valid( PTXOperand::s32, c.type ) 
				&& !PTXOperand::valid( PTXOperand::f32, c.type ) ) {
				return "operand C must be assignable to f32 or s32";
			}
			if( !PTXOperand::valid( PTXOperand::s32, type ) 
				&& !PTXOperand::valid( PTXOperand::f32, type )
				&& !PTXOperand::valid( PTXOperand::s32, type ) ) {
				return "instruction must be be assignable to u32, f32, or s32";
			}
			if( geometry != _a2d && !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( geometry == _a2d && !PTXOperand::valid( type, d.type )  ) {//SI_FIX: temp, for SimpleLayeredTexture
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Tld4: {
			if( type != PTXOperand::f32 && type != PTXOperand::s32
				&& type != PTXOperand::u32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( c.type != PTXOperand::f32 ) {
				return "source C type must be f32 " 
					+ PTXOperand::toString( c.type );
			}
			if( c.vec != PTXOperand::v2 ) {
				return "operand C must be a 2-component vector ";
			}
			break;
		}
		case Txq: {
			if (type != ir::PTXOperand::b32) {
				return "data type must be .b32";
			}
			break;
		}
		case Suld: {
			if (formatMode == Formatted && !(type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::u32
				|| type == ir::PTXOperand::s32
				|| type == ir::PTXOperand::f32)) {
				return "sust.p - data type must be .b32, .u32, .s32, or .f32";
			}
			else if (formatMode == Unformatted && !(type == ir::PTXOperand::b8
				|| type == ir::PTXOperand::b16
				|| type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::b64)) {
				return "sust.b - data type must be .b8, .b16, .b32, or .b64";
			}
			if (cacheOperation == Nc || cacheOperation == Lu) {
				return "suld cache operator must be .ca, .cg, .cs, or .cv";
			}
			break;
		}
		case Suq: {
			if (!(surfaceQuery == Width || surfaceQuery == Height
				|| surfaceQuery == Depth)) {
				return "surface query must be .width, .height, or .depth";
			}
			if (type != ir::PTXOperand::b32) {
				return "data type must be .b32";
			}
			break;
		}
		case Szext: {
			if( type != PTXOperand::u32 && type != PTXOperand::s32 ) {
				return "szext instruction requires .u32 or .s32 type";
			}
			if( shiftMode != ShiftMode::Clamp && shiftMode != ShiftMode::Wrap ) {
				return "szext instruction requires .clamp or .wrap mode";
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "invalid szext destination type " + PTXOperand::toString(d.type);
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "invalid szext source type " + PTXOperand::toString(a.type);
			}
			if( b.type != PTXOperand::u32 ) {
				return "szext operand B must have .u32 type";
			}
			break;
		}
		case Sured: {
			if (!(reductionOperation == ReductionAdd
				|| reductionOperation == ReductionMin 
				|| reductionOperation == ReductionMax
				|| reductionOperation == ReductionAnd
				|| reductionOperation == ReductionOr)) {
				return "reduction operation must be .add, .min, .max, .and, or .or";
			}
			if (!(type == ir::PTXOperand::u32 || type == ir::PTXOperand::u64 
				|| type == ir::PTXOperand::s32 || type == ir::PTXOperand::b32)) {
				return "data type must be .u32, .u64, .s32, or .b32";
			}
			break;
		}
		case Sust: {
			if (formatMode == Formatted && !(type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::u32
				|| type == ir::PTXOperand::s32
				|| type == ir::PTXOperand::f32)) {
				return "sust.p - data type must be .b32, .u32, .s32, or .f32";
			}
			else if (formatMode == Unformatted && !(type == ir::PTXOperand::b8
				|| type == ir::PTXOperand::b16
				|| type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::b64)) {
				return "sust.b - data type must be .b8, .b16, .b32, or .b64";
			}
			if (cacheOperation == Nc || cacheOperation == Lu) {
				return "sust cache operator must be .ca, .cg, .cs, or .cv";
			}
			break;
		}
		case Trap: {
			break;
		}
		case Vote: {
			if (voteSync) {
				if (vote < All || vote > Ballot || modifier || vec != PTXOperand::v1 ||
					type != (vote == Ballot ? PTXOperand::b32 : PTXOperand::pred)) return "invalid vote.sync mode/type/modifier";
				for (const auto* operand : {&d, &a, &b})
					if (operand->vec != PTXOperand::v1 || !operand->array.empty()) return "vote.sync requires scalar operands";
				if (d.addressMode != PTXOperand::Register || a.addressMode != PTXOperand::Register ||
					(vote == Ballot && d.bytes() != 4) || (vote != Ballot && d.condition != PTXOperand::Pred))
					return "vote.sync requires scalar destination and source registers";
				if ((b.addressMode != PTXOperand::Register && b.addressMode != PTXOperand::Immediate) ||
					b.bytes() != 4 || (!PTXOperand::isInt(b.type) && b.type != PTXOperand::b32))
					return "vote.sync member mask must be a 32-bit integer";
			}
			if( vote != Ballot && d.type != PTXOperand::pred ) {
				return "destination must be a predicate";
			}
			else if( vote == Ballot &&
				!PTXOperand::valid(d.type, PTXOperand::b32) ) {
				return "destination must be assignable to b32 for .ballot";
			}
			if( a.type != PTXOperand::pred ) {
				return "operand A must be a predicate";
			}
			break;
		}
		case Xor: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		default: return "check not implemented for " + toString(opcode); break;
	}
	return "";
}

std::string ir::PTXInstruction::guard() const {
	switch( pg.condition ) {
		case PTXOperand::PT: return ""; break;
		case PTXOperand::nPT: return "@!%pt "; break;
		case PTXOperand::InvPred: // return "@!" + pg.toString() + " "; break;
		case PTXOperand::Pred: return "@" + pg.toString() + " " ; break;
	}
	return "";
}

std::string ir::PTXInstruction::toString() const {
	switch (opcode) {
		case Abs: {
			return guard() + "abs." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
		}
		case Add: {
			std::string result = guard() + "add.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
			return result;
		}
		case AddC: {
			std::string result = guard() + "addc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
			return result;
		}
		case And: {
			return guard() + "and." + PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
		}
		case Atom: {
			std::string result = guard() + "atom.";
			if( semantics != Relaxed ) {
				result += toString( semantics ) + ".";
			}
			if( scope != Level_Invalid ) {
				// same ".gpu" spelling quirk as fence's printer
				result += ( ( scope == GlobalLevel ) ? "gpu" : toString( scope ) )
					+ std::string( "." );
			}
			result += toString( addressSpace )
				+ "." + toString( atomicOperation ) + "."
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", [" + a.toString() + "], "
				+ b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Bar: {
			std::string result = guard() + "bar." + toString(barrierOperation);
			ir::PTXOperand ir::PTXInstruction::* instrMembers [] = { 
				&ir::PTXInstruction::d, 
				&ir::PTXInstruction::a, 
				&ir::PTXInstruction::b, 
				&ir::PTXInstruction::c,
				&ir::PTXInstruction::pq
			};
			switch (barrierOperation) {
				case BarReduction: 
				{
					result += toString(reductionOperation)
						+ PTXOperand::toString(type);
				}
				break;
				default: break;
			};
			for (int i = 0; instrMembers[i] != &ir::PTXInstruction::pq; i++) {
				if ((this->*instrMembers[i]).addressMode != ir::PTXOperand::Invalid) {
					result += " " + (this->*instrMembers[i]).toString();
				}
			}
			return result;
		}
        case Bfe: {
			return guard() + "bfe." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString()
				+ ", " + b.toString() + ", " + c.toString();
		}
		case Bfi: {
			return guard() + "bfi." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + pq.toString() + ", " + a.toString()
				+ ", " + b.toString() + ", " + c.toString();
		}
		case Bfind: {
			std::string result = guard() + "bfind.";
			if( shiftAmount ) result += "shiftamt.";
			result += PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
			return result;
		}
		case Bmsk: {
			return guard() + "bmsk." + toString(shiftMode) + ".b32 "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Lop3: {
			std::string result = guard() + "lop3";
			if( pq.addressMode != PTXOperand::Invalid ) {
				result += "." + toString(booleanOperator);
			}
			result += ".b32 " + d.toString();
			if( pq.addressMode != PTXOperand::Invalid ) result += "|" + pq.toString();
			result += ", " + a.toString() + ", " + b.toString() + ", "
				+ c.toString() + ", " + immLut.toString();
			if( pq.addressMode != PTXOperand::Invalid ) result += ", " + q.toString();
			return result;
		}
		case Bra: {
			std::string result = guard() + "bra";
			if( uni ) {
				result += ".uni";
			}
			return result + " " + d.toString();
		}
		case Brev: {
			return guard() + "brev." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
		}
		case Brkpt: {
			return guard() + "brkpt";
		}
		case Call: {
			std::string result = guard() + "call";
			if( uni ) {
				result += ".uni";
			}
			else if( tailCall ) {
				result += ".tail";
			}
			result += " ";
			if( d.addressMode != PTXOperand::Invalid ) {
				result += d.toString() + ", ";
			}
			result += a.toString();
			if( b.addressMode != PTXOperand::Invalid ) {
				result += ", " + b.toString();
			}
			if( a.addressMode == PTXOperand::Register ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Clz: {
			return guard() + "clz." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case CNot: {
			return guard() + "cnot." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case CopySign: {
			return guard() + "copysign." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Cos: {
			return guard() + "cos." + modifierString( modifier, carry )
				+ PTXOperand::toString( type ) + " " + d.toString()
				+ ", " + a.toString();
		}
		case Cvt: {
			std::string result = guard() + "cvt.";
			if( modifier & rni ) result += "rni.";
			else if( modifier & rzi ) result += "rzi.";
			else if( modifier & rmi ) result += "rmi.";
			else if( modifier & rpi ) result += "rpi.";
			result += modifierString(modifier, carry);
			
			PTXOperand::DataType sourceType = a.type;
	
			if( a.relaxedType != PTXOperand::TypeSpecifier_invalid ) {
				sourceType = a.relaxedType;
			}
			
			result += PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( sourceType ) + " " + d.toString() + ", " 
				+ a.toString();
			if( type == PTXOperand::f16x2 || type == PTXOperand::bf16x2 ) {
				result += ", " + b.toString();
			} else if (b.addressMode != PTXOperand::Invalid) {
				result += ", " + b.toString();
			}
			return result;
		}
		case Cvta: {
			std::string result = guard() + "cvta.";
			
			if (toAddrSpace) {
				result += "to.";
			}
			
			result += toString(addressSpace)
				+ "." + PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString();
			return result;
		}
		case Div: {
			std::string result = guard() + "div.";
			if( modifier & full ) {
				result += "full.";
			}
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Dp2a:
		case Dp4a: {
			std::string result = guard() + toString(opcode) + ".";
			if( opcode == Dp2a ) result += modifierString(modifier);
			return result + PTXOperand::toString(type) + "."
				+ PTXOperand::toString(bType) + " " + d.toString() + ", "
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
		}
		case Ex2: {
			std::string result = guard() + "ex2.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case Exit: {
			return "exit";
		}
		case Fma: {
			std::string result = guard() + "fma."
				+ modifierString(modifier, carry)
				+ PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Fns: {
			return guard() + "fns.b32 " + d.toString() + ", " + a.toString()
				+ ", " + b.toString() + ", " + c.toString();
		}
		case Isspacep: {
			std::string result = guard() + "isspacep." + toString(addressSpace) 
				+ " " + d.toString() + ", " + a.toString();
			return result;
		}
		case Ld: {
			std::string result = guard() + "ld.";
			if( volatility == Volatile ) {
				result += "volatile.";
			} else if( semantics != Weak ) {
				if( mmio ) {
					result += "mmio.";
				}
				result += toString( semantics ) + ".";
				if( scope != Level_Invalid ) {
					result += ( ( scope == GlobalLevel ) ? "gpu" : toString( scope ) )
						+ std::string( "." );
				}
			}
			if( cacheOperation != Ca ) {
				result += toStringLoad(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( d.vec != PTXOperand::v1 ) {
				result += toString( d.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + "]";
			return result;
		}
		case Ldmatrix: {
			return guard() + "ldmatrix.sync.aligned.m8n8.x" + std::to_string(unsigned(vec)) +
				(matrixTranspose ? ".trans" : "") + (addressSpace == Shared ? ".shared" : "") +
				".b16 " + d.toString() + ", [" + a.toString() + "]";
		}
		case Ldu: {
			std::string result = guard() + "ldu.";
			if( volatility == Volatile ) {
				result += "volatile.";
			}
			if( cacheOperation != Ca ) {
				result += toStringLoad(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( d.vec != PTXOperand::v1 ) {
				result += toString( d.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + "]";
			return result;
		}
		case Lg2: {
			std::string result = guard() + "lg2."; 
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case Mad24: {
			std::string result = guard() + "mad24.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Mad: {
			std::string result = guard() + "mad.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Mma: {
			std::string shapeName;
			switch (mmaShape) {
			case MmaM8N8K32:  shapeName = "8n8k32";  break;
			case MmaM16N8K64: shapeName = "16n8k64"; break;
			case MmaM8N8K128: shapeName = "8n8k128"; break;
			case MmaM16N8K128: shapeName = "16n8k128"; break;
			case MmaM16N8K256: shapeName = "16n8k256"; break;
			case MmaM8N8K4:   shapeName = "8n8k4";   break;
			case MmaM16N8K4:  shapeName = "16n8k4";  break;
			case MmaM16N8K8:  shapeName = "16n8k8";  break;
			case MmaM8N8K16:  shapeName = "8n8k16";  break;
			case MmaM16N8K32: shapeName = "16n8k32"; break;
			default:          shapeName = "16n8k16"; break;
			}
			std::string result = guard() +
				(mmaSparse ? "mma.sp.sync.aligned.m" : "mma.sync.aligned.m") +
				shapeName +
				(mmaAColumnMajor ? ".col" : ".row") +
				(mmaBColumnMajor ? ".col." : ".row.") +
				((modifier & satfinite) ? "satfinite." : "") +
				PTXOperand::toString(type) + "." +
				PTXOperand::toString(a.type) + "." +
				PTXOperand::toString(b.type) + "." +
				PTXOperand::toString(c.type) +
				(a.type == PTXOperand::b1 ? "." + toString(booleanOperator) + ".popc" : "") +
				(type == PTXOperand::f64 && modifier ? "." + toString((Modifier)modifier) : "") + " " +
				d.toString() + ", " + a.toString() + ", " +
				b.toString() + ", " + c.toString();
			if (mmaSparse) result += ", " + q.toString() + ", " + mmaSelector.toString();
			return result;
		}
		case MadC: {
			std::string result = guard() + "madc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Max: {
			return guard() + "max." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Membar: {
			return guard() + "membar." + toString( level );
		}
		case Fence: {
			// fence spells the device-wide scope ".gpu", membar spells
			// the same GlobalLevel value ".gl" -- toString(Level) can't be reused as-is
			std::string scope = ( level == GlobalLevel ) ? "gpu" : toString( level );
			return guard() + "fence." + toString( semantics ) + "." + scope;
		}
		case Min: {
			return guard() + "min." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Match:
			return guard() + "match." + (matchAll ? "all" : "any") + ".sync." + PTXOperand::toString(type) + " " +
				d.toString() + (pq.addressMode == PTXOperand::Invalid ? "" : "|" + pq.toString()) + ", " + a.toString() + ", " + b.toString();
		case Redux:
			return guard() + "redux.sync." + toString(reductionOperation) + "." + PTXOperand::toString(type) +
				" " + d.toString() + ", " + a.toString() + ", " + b.toString();
		case Activemask:
			return guard() + "activemask.b32 " + d.toString();
		case Movmatrix:
			return guard() + "movmatrix.sync.aligned.m8n8.trans.b16 " + d.toString() + ", " + a.toString();
		case Mov: {
			return guard() + "mov." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Mul24: {
			std::string result = guard() + "mul24.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Mul: {
			std::string result = guard() + "mul.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Neg: {
			return guard() + "neg." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Not: {
			return guard() + "not." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Pmevent: {
			return guard() + "pmevent." + toString( level );
		}
		case Popc: {
			return guard() + "popc." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Prefetch: {
			return guard() + "prefetch." + toString(addressSpace) + "." + 
				PTXInstruction::toString(cacheLevel) + " [" + d.toString() + "]";
		}
		case Prefetchu: {
			return guard() + "prefetchu.L1 [" + d.toString() + "]";
		}
		case Prmt: {
			std::string result = guard() + "prmt." 
				+ PTXOperand::toString( type );
			if( permuteMode != DefaultPermute )
			{
				result += "." + toString( permuteMode );
			}
			result += " " + d.toString() + ", " + a.toString() + ", " 
				+ b.toString() + ", " + c.toString();
			return result;
		}
		case Or: {
			return guard() + "or." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Rcp: {
			std::string result = guard() + "rcp.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
			return result;
		}
		case Red: {
			std::string result = guard() + "red.";
			if( semantics != Relaxed ) {
				result += toString( semantics ) + ".";
			}
			if( scope != Level_Invalid ) {
				result += ( ( scope == GlobalLevel ) ? "gpu" : toString( scope ) )
					+ std::string( "." );
			}
			result += toString( addressSpace ) + "."
				+ toString( reductionOperation ) + "."
				+ PTXOperand::toString( type ) + " " + d.toString() + ", "
				+ a.toString();
			return result;
		}
		case Rem: {
			return guard() + "rem." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Ret: {
			std::string result = guard() + "ret";
			if( uni ) {
				result += ".uni";
			}
			return result;
		}
		case Rsqrt: {
			std::string result = guard() + "rsqrt.";
			result += modifierString( modifier, carry );			
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();			
			return result;
		}
		case Sad: {
			return guard() + "sad." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString() 
				+ ", " + c.toString();
		}
		case SelP: {
			return guard() + "selp." + PTXOperand::toString( type ) 
				+ " " + d.toString() + ", " + a.toString() + ", " + b.toString()
				+ ", " + c.toString();
		}
		case Set: {
			std::string result = guard() + "set." 
				+ toString( comparisonOperator ) + ".";
			if( c.addressMode != PTXOperand::Invalid ) {
				result += toString( booleanOperator ) + ".";
			}
			if( ftz & modifier ) result += "ftz.";
			result += PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( a.type ) + " " + d.toString() 
				+ ", " + a.toString() + ", " + b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;				
		}
		case SetP: {
			std::string result = guard() + "setp." 
				+ toString( comparisonOperator ) + ".";
			if( c.addressMode != PTXOperand::Invalid ) {
				result += toString( booleanOperator ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString();
			if( pq.addressMode != PTXOperand::Invalid && !isPt( pq ) ) {
				result += "|" + pq.toString();
			}
			result += ", " + a.toString() + ", " + b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Shf: {
			return guard() + "shf." + toString( shiftDirection ) + '.'
				+ toString( shiftMode ) + '.' + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString()
				+ ',' + c.toString();
		}
		case Shfl: {
			std::string result = guard() + "shl." + toString( shuffleMode )
				+ "." +	PTXOperand::toString( type ) + " " + d.toString();

			if( pq.addressMode != PTXOperand::Invalid && !isPt( pq ) ) {
				result += "|" + pq.toString();
			}
				
			result += ", " + a.toString() + ", " + b.toString() + ", " +
				c.toString();
		
			return result;
		}
		case Shl: {
			return guard() + "shl." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Shr: {
			return guard() + "shr." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Sin: {
			std::string result = guard() + "sin.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case SlCt: {
			return guard() + "slct." + modifierString( modifier, carry )
				+ PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( c.type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
		}
		case Sqrt: {
			std::string result = guard() + "sqrt.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case St: {
			std::string result = guard() + "st.";
			if( volatility == Volatile ) {
				result += "volatile.";
			} else if( semantics != Weak ) {
				if( mmio ) {
					result += "mmio.";
				}
				result += toString( semantics ) + ".";
				if( scope != Level_Invalid ) {
					result += ( ( scope == GlobalLevel ) ? "gpu" : toString( scope ) )
						+ std::string( "." );
				}
			}
			if( cacheOperation != Wb ) {
				result += toStringStore(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( a.vec != PTXOperand::v1 ) {
				result += toString( a.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " [" + d.toString() + "], "
				+ a.toString();
			return result;
		}
		case Sub: {
			std::string result = guard() + "sub.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
			return result;
		}
		case SubC: {
			std::string result = guard() + "subc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
			return result;
		}
		case Suld: {
			return guard() + "suld" + toString(formatMode) + "." 
				+ toString(geometry) + "." + toString(vec) + "."
				+ PTXOperand::toString(type) + toString(clamp) +
				" " + d.toString() + ", [" + a.toString() + ", " 
				+ b.toString() + "]";
		}
		case Suq: {
			return guard() + "suq." + toString( surfaceQuery ) 
				+ "." + PTXOperand::toString(type) + " " + d.toString() 
				+ ", [" + a.toString() + "]";
		}
		case Szext: {
			return guard() + "szext." + toString(shiftMode) + "."
				+ PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString() + ", " + b.toString();
		}
		case Sured: {
			return guard() + "sured" + toString(formatMode) + "." 
				+ toString(reductionOperation) + "." +
				toString(geometry) + "." + PTXOperand::toString(type) 
				+  toString(clamp) + " [" + d.toString() + ", " + a.toString()
				+ "], " + b.toString();
		}
		case Sust: {
			return guard() + "sust" + toString(formatMode) + "." 
				+ toString(geometry) + "." 
				+ ((vec != Vec::v1)?toString(vec) + ".":"") + PTXOperand::toString(type) 
				+ toString(clamp) + " [" + d.toString() +", " + a.toString() 
				+ "], " + b.toString();
		}
		case Tanh: {
			return guard() + "tanh." + modifierString(modifier)
				+ PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString();
		}
		case TestP: {
			return guard() + "testp." + toString( floatingPointMode ) 
				+ "." + PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString();
		}
		case Tex: {			
			return guard() + "tex." + toString( geometry ) + ".v4." 
				+ PTXOperand::toString( d.type ) + "." 
				+ PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + ", " + c.toString() + "]"; 
		}
		case Tld4: {
			return guard() + "tld4." + toString( colorComponent ) + ".2d.v4." 
				+ PTXOperand::toString( d.type ) + "." 
				+ PTXOperand::toString( c.type ) + " " + d.toString() + ", [" 
				+ a.toString() + ", " + c.toString() + "]"; 
		}
		case Trap: {
			return guard() + "trap";
		}
		case Txq: {
			return guard() + "txq." + toString( surfaceQuery ) 
				+ "." + PTXOperand::toString(type) + " " + d.toString()
				+ ", [" + a.toString() + "]";
		}
		case Vote: {
			return guard() + (voteSync ? "vote.sync." : "vote.") + toString( vote ) + "."
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + (voteSync ? ", " + b.toString() : "");
		}
		case Xor: {
			return guard() + "xor." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Reconverge: {
			return "reconverge";
		}
		default: break;
	}
	assertM(false, "Instruction opcode " << toString(opcode) 
		<< " not implemented.");
		
	return "";
}

ir::Instruction* ir::PTXInstruction::clone(bool copy) const {
	if (copy) {
		return new PTXInstruction(*this);
	}
	else {
		return new PTXInstruction;
	}
}

bool ir::PTXInstruction::isBranch() const {
	return opcode == Bra || opcode == Call;
}

bool ir::PTXInstruction::isCall() const {
	return opcode == Call;
}

bool ir::PTXInstruction::isLoad() const {
	return opcode == Ld || opcode == Ldu || opcode == Ldmatrix;
}

bool ir::PTXInstruction::isStore() const {
	return opcode == St;
}

bool ir::PTXInstruction::mayHaveAddressableOperand() const {
	return opcode == Mov || opcode == Ld || opcode == St || opcode == Cvta
		|| opcode == Atom || opcode == Ldu || opcode == Ldmatrix;
}

bool ir::PTXInstruction::mayHaveRelaxedTypeDestination() const {
	return opcode == Ld || opcode == Cvt
		|| opcode == Ldu || opcode == SelP;
}

bool ir::PTXInstruction::hasSideEffects() const {
	return opcode == St || opcode == Atom || opcode == Bar
		|| opcode == Bra || opcode == Call || opcode == Exit || opcode == Ldu
		|| opcode == Membar || opcode == Tex || opcode == Tld4
		|| opcode == Prefetch || opcode == Sust || opcode == Suq
		|| opcode == Trap || opcode == Reconverge || opcode == Ret;
}

bool ir::PTXInstruction::canObserveSideEffects() const {
	if (opcode == Match || opcode == Redux || (opcode == Vote && voteSync)) return true;
	if (opcode == Atom || opcode == Ldmatrix || opcode == Movmatrix || opcode == Activemask) return true;

	if (opcode == Ld) {
		if (volatility == Volatile) {
			return true;
		}
		
		if (cacheOperation == Cv || cacheOperation == Cg) {
			return true;
		}
		
		return false;
	}
	
	return false;
}

bool ir::PTXInstruction::isMemoryInstruction() const {
	return opcode == St || opcode == Atom
		|| opcode == Ldu
		|| opcode == Tex || opcode == Tld4
		|| opcode == Prefetch || opcode == Sust || opcode == Ld || opcode == Ldmatrix;
}

bool ir::PTXInstruction::isExit() const {
	return opcode == Exit || opcode == Ret;
}
