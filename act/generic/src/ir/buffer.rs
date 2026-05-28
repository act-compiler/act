use crate::ir::egraph::TensorOp;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Buffer {
    HBM,
{{BUFFER_VARIANTS}}
    ANY,
}

impl std::fmt::Display for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Buffer::HBM => write!(f, "HBM"),
{{BUFFER_DISPLAY_MATCH_ARMS}}
            Buffer::ANY => panic!("Buffer::ANY does not have a string representation"),
        }
    }
}

// Return buffer assignment for an instruction enode.
// Returns Some(vec![out_buf, in_buf1, in_buf2, ...]) or None if not applicable.
pub fn buffer_assignment(en: &TensorOp) -> Option<Vec<Buffer>> {
    match en {
{{ISA_BUFFER_ASSIGNMENT_MATCH_ARMS}}
        TensorOp::OpSlice(_, _) => Some(vec![Buffer::ANY, Buffer::ANY]),
        TensorOp::OpConcat(_, _) => Some(vec![Buffer::ANY, Buffer::ANY, Buffer::ANY]),
        TensorOp::DetectedConst(_) => Some(vec![Buffer::HBM]),
        TensorOp::Var(_) => Some(vec![Buffer::HBM]),
        _ => None,
    }
}
