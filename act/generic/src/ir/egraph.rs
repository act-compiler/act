use egg::{Analysis, DidMerge, EGraph, FromOp, FromOpError, Id, Language, LanguageChildren};

use crate::ir::dtype::Dtype;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum TensorOp {
    // ISA instructions
{{ISA_ENUM_VARIANTS}}
    // IR operators
    OpAbs([Id; 1]),
    OpAdd([Id; 2]),
    OpAnd([Id; 2]),
    OpBitcvt([Id; 1]),
    OpBroadcast(String, [Id; 1]),
    OpClamp([Id; 3]),
    OpCompare(String, [Id; 2]),
    OpConcat(String, [Id; 2]),
    OpConstant(String),
    OpConvert(String, [Id; 1]),
    OpCopy([Id; 1]),
    OpDivide([Id; 2]),
    OpDot([Id; 2]),
    OpExp([Id; 1]),
    OpEye(String),
    OpIota(String),
    OpLogPlusOne([Id; 1]),
    OpLogistic([Id; 1]),
    OpMaximum([Id; 2]),
    OpMinimum([Id; 2]),
    OpMultiply([Id; 2]),
    OpNegate([Id; 1]),
    OpOr([Id; 2]),
    OpPower([Id; 2]),
    OpReduceSum(String, [Id; 1]),
    OpReshape(String, [Id; 1]),
    OpReverse(String, [Id; 1]),
    OpSelect([Id; 3]),
    OpShiftLeft([Id; 2]),
    OpShiftRightArithmetic([Id; 2]),
    OpShiftRightLogical([Id; 2]),
    OpSlice(String, [Id; 1]),
    OpSubtract([Id; 2]),
    OpTanh([Id; 1]),
    OpUpdateSlice(String, [Id; 2]),
    OpXor([Id; 2]),
    OpTranspose(String, [Id; 1]),
    // other
    DetectedConst(String),
    Var(String),
}

impl TensorOp {
    pub fn num_children(&self) -> usize {
        match self {
{{ISA_NUM_CHILDREN_MATCH_ARMS}}
            TensorOp::OpAbs(..) => 1,
            TensorOp::OpAdd(..) => 2,
            TensorOp::OpAnd(..) => 2,
            TensorOp::OpBitcvt(..) => 1,
            TensorOp::OpBroadcast(..) => 1,
            TensorOp::OpClamp(..) => 3,
            TensorOp::OpCompare(..) => 2,
            TensorOp::OpConcat(..) => 2,
            TensorOp::OpConstant(..) => 0,
            TensorOp::OpConvert(..) => 1,
            TensorOp::OpCopy(..) => 1,
            TensorOp::OpDivide(..) => 2,
            TensorOp::OpDot(..) => 2,
            TensorOp::OpExp(..) => 1,
            TensorOp::OpEye(..) => 0,
            TensorOp::OpIota(..) => 0,
            TensorOp::OpLogPlusOne(..) => 1,
            TensorOp::OpLogistic(..) => 1,
            TensorOp::OpMaximum(..) => 2,
            TensorOp::OpMinimum(..) => 2,
            TensorOp::OpMultiply(..) => 2,
            TensorOp::OpNegate(..) => 1,
            TensorOp::OpOr(..) => 2,
            TensorOp::OpPower(..) => 2,
            TensorOp::OpReduceSum(..) => 1,
            TensorOp::OpReshape(..) => 1,
            TensorOp::OpReverse(..) => 1,
            TensorOp::OpSelect(..) => 3,
            TensorOp::OpShiftLeft(..) => 2,
            TensorOp::OpShiftRightArithmetic(..) => 2,
            TensorOp::OpShiftRightLogical(..) => 2,
            TensorOp::OpSlice(..) => 1,
            TensorOp::OpSubtract(..) => 2,
            TensorOp::OpTanh(..) => 1,
            TensorOp::OpUpdateSlice(..) => 2,
            TensorOp::OpXor(..) => 2,
            TensorOp::OpTranspose(..) => 1,
            TensorOp::DetectedConst(..) => 0,
            TensorOp::Var(..) => 0,
        }
    }

    pub fn is_detected_const(&self) -> bool {
        match self {
            TensorOp::DetectedConst(_) => true,
            _ => false,
        }
    }

    pub fn inplace_children(&self) -> Vec<bool> {
        match self {
{{ISA_INPLACE_MATCH_ARMS}}
            TensorOp::OpSlice(..) => vec![true],
            TensorOp::OpConcat(..) => vec![true, true],
            _ => vec![false; self.num_children()],
        }
    }

    pub fn set_metadata(&mut self, metadata: Option<String>) {
        match self {
{{ISA_SET_METADATA_MATCH_ARMS}}
            TensorOp::OpBroadcast(data, _) => {
                *data = metadata.expect("OpBroadcast needs metadata!")
            }
            TensorOp::OpCompare(data, _) => {
                *data = metadata.expect("OpCompare needs metadata!")
            }
            TensorOp::OpConcat(data, _) => *data = metadata.expect("OpConcat needs metadata!"),
            TensorOp::OpConstant(data) => *data = metadata.expect("OpConstant needs metadata!"),
            TensorOp::OpConvert(data, _) => *data = metadata.expect("OpConvert needs metadata!"),
            TensorOp::OpEye(data) => *data = metadata.expect("OpEye needs metadata!"),
            TensorOp::OpIota(data) => *data = metadata.expect("OpIota needs metadata!"),
            TensorOp::OpReduceSum(data, _) => {
                *data = metadata.expect("OpReduceSum needs metadata!")
            }
            TensorOp::OpReshape(data, _) => *data = metadata.expect("OpReshape needs metadata!"),
            TensorOp::OpReverse(data, _) => *data = metadata.expect("OpReverse needs metadata!"),
            TensorOp::OpSlice(data, _) => *data = metadata.expect("OpSlice needs metadata!"),
            TensorOp::OpUpdateSlice(data, _) => {
                *data = metadata.expect("OpUpdateSlice needs metadata!")
            }
            TensorOp::OpTranspose(data, _) => {
                *data = metadata.expect("OpTranspose needs metadata!")
            }
            _ => (),
        }
    }
}

impl Language for TensorOp {
    type Discriminant = std::mem::Discriminant<Self>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(self)
    }

    // All variants have a fixed number of children, so if self and other are the same variant,
    // then they must have the same arity.
    fn matches(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn children(&self) -> &[Id] {
        match self {
{{ISA_CHILDREN_MATCH_ARMS}}
            TensorOp::OpAbs(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpAdd(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpAnd(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpBitcvt(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpBroadcast(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpClamp(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpCompare(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpConcat(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpConstant(_) => &[],
            TensorOp::OpConvert(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpCopy(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpDivide(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpDot(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpExp(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpEye(_) => &[],
            TensorOp::OpIota(_) => &[],
            TensorOp::OpLogPlusOne(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpLogistic(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpMaximum(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpMinimum(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpMultiply(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpNegate(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpOr(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpPower(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReduceSum(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReshape(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpReverse(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpSelect(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpShiftLeft(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpShiftRightArithmetic(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpShiftRightLogical(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpSlice(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpSubtract(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpTanh(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpUpdateSlice(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpXor(ids) => LanguageChildren::as_slice(ids),
            TensorOp::OpTranspose(_, ids) => LanguageChildren::as_slice(ids),
            TensorOp::DetectedConst(_) => &[],
            TensorOp::Var(_) => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
{{ISA_CHILDREN_MUT_MATCH_ARMS}}
            TensorOp::OpAbs(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpAdd(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpAnd(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpBitcvt(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpBroadcast(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpClamp(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpCompare(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpConcat(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpConstant(_) => &mut [],
            TensorOp::OpConvert(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpCopy(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpDivide(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpDot(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpExp(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpEye(_) => &mut [],
            TensorOp::OpIota(_) => &mut [],
            TensorOp::OpLogPlusOne(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpLogistic(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpMaximum(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpMinimum(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpMultiply(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpNegate(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpOr(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpPower(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReduceSum(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReshape(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpReverse(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpSelect(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpShiftLeft(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpShiftRightArithmetic(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpShiftRightLogical(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpSlice(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpSubtract(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpTanh(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpUpdateSlice(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpXor(ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::OpTranspose(_, ids) => LanguageChildren::as_mut_slice(ids),
            TensorOp::DetectedConst(_) => &mut [],
            TensorOp::Var(_) => &mut [],
        }
    }
}

impl FromOp for TensorOp {
    type Error = FromOpError;

    // define_language picks the first variant where it is possible to parse data into type
    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
{{ISA_FROM_OP_MATCH_ARMS}}
            op if op == "abs" && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpAbs(children))
            }
            op if op == "add" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpAdd(children))
            }
            op if op == "and" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpAnd(children))
            }
            op if op == "bitcvt"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpBitcvt(children))
            }
            op if op.split('_').next().unwrap() == "broadcast"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpBroadcast(data.to_string(), children))
            }
            op if op == "clamp"
                && <[Id; 3] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 3] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpClamp(children))
            }
            op if op.split('_').next().unwrap() == "compare"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpCompare(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "concat"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpConcat(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "constant"
                && <[Id; 0] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                Ok(TensorOp::OpConstant(data.to_string()))
            }
            op if op.split('_').next().unwrap() == "convert"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpConvert(data.to_string(), children))
            }
            op if op == "copy" && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpCopy(children))
            }
            op if op == "divide"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpDivide(children))
            }
            op if op == "dot" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpDot(children))
            }
            op if op == "exponential"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpExp(children))
            }
            op if op.split('_').next().unwrap() == "eye"
                && <[Id; 0] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                Ok(TensorOp::OpEye(data.to_string()))
            }
            op if op.split('_').next().unwrap() == "iota"
                && <[Id; 0] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                Ok(TensorOp::OpIota(data.to_string()))
            }
            op if op == "log-plus-one"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpLogPlusOne(children))
            }
            op if op == "logistic"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpLogistic(children))
            }
            op if op == "maximum"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMaximum(children))
            }
            op if op == "minimum"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMinimum(children))
            }
            op if op == "multiply"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpMultiply(children))
            }
            op if op == "negate"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpNegate(children))
            }
            op if op == "or" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpOr(children))
            }
            op if op == "power" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpPower(children))
            }
            op if op.split('_').next().unwrap() == "reduce"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReduceSum(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "reshape"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReshape(data.to_string(), children))
            }
            op if op.split('_').next().unwrap() == "reverse"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpReverse(data.to_string(), children))
            }
            op if op == "select"
                && <[Id; 3] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 3] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpSelect(children))
            }
            op if op == "shift-left"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpShiftLeft(children))
            }
            op if op == "shift-right-arithmetic"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpShiftRightArithmetic(children))
            }
            op if op == "shift-right-logical"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpShiftRightLogical(children))
            }
            op if op.split('_').next().unwrap() == "slice"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpSlice(data.to_string(), children))
            }
            op if op == "subtract"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpSubtract(children))
            }
            op if op == "tanh"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpTanh(children))
            }
            op if op.split('_').next().unwrap() == "update-slice"
                && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpUpdateSlice(data.to_string(), children))
            }
            op if op == "xor" && <[Id; 2] as LanguageChildren>::can_be_length(children.len()) => {
                let children = <[Id; 2] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpXor(children))
            }
            op if op.split('_').next().unwrap() == "transpose"
                && <[Id; 1] as LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.split('_').last().unwrap();
                let children = <[Id; 1] as LanguageChildren>::from_vec(children);
                Ok(TensorOp::OpTranspose(data.to_string(), children))
            }
            op if op.starts_with('?') && children.is_empty() => Ok(TensorOp::Var(op.to_string())),
            _ => Err(FromOpError::new(op, children)),
        }
    }
}

impl std::fmt::Display for TensorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
{{ISA_DISPLAY_MATCH_ARMS}}
            TensorOp::OpAbs(_) => write!(f, "abs"),
            TensorOp::OpAdd(_) => write!(f, "add"),
            TensorOp::OpAnd(_) => write!(f, "and"),
            TensorOp::OpBitcvt(_) => write!(f, "bitcvt"),
            TensorOp::OpBroadcast(data, _) => write!(f, "broadcast{{{}}}", data),
            TensorOp::OpClamp(_) => write!(f, "clamp"),
            TensorOp::OpCompare(data, _) => write!(f, "compare{{{}}}", data),
            TensorOp::OpConcat(data, _) => write!(f, "concat{{{}}}", data),
            TensorOp::OpConstant(data) => write!(f, "constant{{{}}}", data),
            TensorOp::OpConvert(data, _) => write!(f, "convert{{{}}}", data),
            TensorOp::OpCopy(_) => write!(f, "copy"),
            TensorOp::OpDivide(_) => write!(f, "divide"),
            TensorOp::OpDot(_) => write!(f, "dot"),
            TensorOp::OpExp(_) => write!(f, "exponential"),
            TensorOp::OpEye(data) => write!(f, "eye{{{}}}", data),
            TensorOp::OpIota(data) => write!(f, "iota{{{}}}", data),
            TensorOp::OpLogPlusOne(_) => write!(f, "log-plus-one"),
            TensorOp::OpLogistic(_) => write!(f, "logistic"),
            TensorOp::OpMaximum(_) => write!(f, "maximum"),
            TensorOp::OpMinimum(_) => write!(f, "minimum"),
            TensorOp::OpMultiply(_) => write!(f, "multiply"),
            TensorOp::OpNegate(_) => write!(f, "negate"),
            TensorOp::OpOr(_) => write!(f, "or"),
            TensorOp::OpPower(_) => write!(f, "power"),
            TensorOp::OpReduceSum(data, _) => write!(f, "reduce{{{}}}", data),
            TensorOp::OpReshape(data, _) => write!(f, "reshape{{{}}}", data),
            TensorOp::OpReverse(data, _) => write!(f, "reverse{{{}}}", data),
            TensorOp::OpSelect(_) => write!(f, "select"),
            TensorOp::OpShiftLeft(_) => write!(f, "shift_left"),
            TensorOp::OpShiftRightArithmetic(_) => write!(f, "shift_right_arithmetic"),
            TensorOp::OpShiftRightLogical(_) => write!(f, "shift_right_logical"),
            TensorOp::OpSlice(data, _) => write!(f, "slice{{{}}}", data),
            TensorOp::OpSubtract(_) => write!(f, "subtract"),
            TensorOp::OpTanh(_) => write!(f, "tanh"),
            TensorOp::OpUpdateSlice(data, _) => write!(f, "update-slice{{{}}}", data),
            TensorOp::OpXor(_) => write!(f, "xor"),
            TensorOp::OpTranspose(data, _) => write!(f, "transpose{{{}}}", data),
            TensorOp::DetectedConst(id) => write!(f, "DCC{{{}}}", id),
            TensorOp::Var(v) => write!(f, "Var{{{}}}", v),
        }
    }
}

// E-class metadata
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    pub is_const: bool,
}

impl Default for TensorInfo {
    fn default() -> Self {
        TensorInfo {
            shape: vec![],
            dtype: Dtype::U8,
            is_const: false,
        }
    }
}

impl PartialEq for TensorInfo {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}

impl Analysis<TensorOp> for TensorInfo {
    type Data = TensorInfo;

    fn make(_egraph: &mut EGraph<TensorOp, Self>, enode: &TensorOp) -> Self::Data {
        let mut data = TensorInfo::default();
        data.is_const = match enode {
            TensorOp::DetectedConst(_) => true,
            _ => false,
        };
        data
    }

    // TODO: ensure that the two eclasses have the same shape
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let x = a.is_const;
        a.is_const |= b.is_const;
        DidMerge(a.is_const != x, false)
    }
}
