use egg::{EGraph, Id};
use itertools::Itertools;

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

/// precond_*(): Return if the rewrite should be applied.
/// metadata_*(): Return a list of metadata strings to use for RHS enodes
/// set_shapes_*(): Set the TensorInfo for each RHS eclass
/// TODO: change name to set_metadata_*()

pub fn precond_0(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let i = lhs_metadata[0].shape[0] / 16;
    if (i % 2 != 0) || (i < 2) {
        return false;
    }
    if lhs_metadata[0].shape != vec![i * 16, 16] || lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].shape != vec![16, 16] || lhs_metadata[1].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[2] != (TensorInfo { shape: vec![i * 16, 16], dtype: Dtype::I32, is_const: false, }) {
        return false;
    }
    if lhs_metadata[3] != (TensorInfo { shape: vec![i * 16, 16], dtype: Dtype::I32, is_const: false, }) {
        return false;
    }
    if lhs_metadata[4] != (TensorInfo { shape: vec![i * 16, 16], dtype: Dtype::I32, is_const: false, }) {
        return false;
    }
    true
}

pub fn metadata_0(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let i = lhs_metadata[0].shape[0] / 16;

    // TODO: fix string to slice the first dim and keep all others
    let mut rhs_metadata = vec![None; 12];
    rhs_metadata[1] = Some(format!("{}:{}", 0, (i / 2) * 16));
    rhs_metadata[5] = Some(format!("{}:{}", 0, (i / 2) * 16));
    rhs_metadata[7] = Some(format!("{}:{}", (i / 2) * 16, i * 16));
    rhs_metadata[9] = Some(format!("{}:{}", (i / 2) * 16, i * 16));
    rhs_metadata[11] = Some("1".to_string());
    rhs_metadata
}

pub fn set_shapes_0(
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    rhs_eclasses: &Vec<Id>
) {
    assert_eq!(lhs_eclasses.len(), 5);
    assert_eq!(rhs_eclasses.len(), 12);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let i = lhs_metadata[0].shape[0] / 16;

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[3],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[5],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[6],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[7],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[8],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[9],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[10],
        TensorInfo { shape: vec![(i / 2) * 16, 16], dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[11],
        TensorInfo { shape: vec![i * 16, 16], dtype: Dtype::I32, is_const: false, });
}

pub fn precond_1(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    true
}

pub fn metadata_1(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let slice_data = match &lhs_enodes[2] {
        Some(TensorOp::OpSlice(data, _)) => data.clone(),
        _ => panic!("OpSlice should have metadata!"),
    };
    let ty = lhs_metadata[1].dtype;

    let mut rhs_metadata = vec![None; 3];
    rhs_metadata[1] = Some(slice_data);
    rhs_metadata[2] = Some(format!("{:?}", ty));
    rhs_metadata
}

pub fn set_shapes_1(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: lhs_metadata[2].shape.clone(), dtype: lhs_metadata[0].dtype, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[2],
        TensorInfo { shape: lhs_metadata[2].shape.clone(), dtype: lhs_metadata[1].dtype, is_const: false, });
}

pub fn precond_2(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0] != (TensorInfo { shape: vec![16, 16], dtype: Dtype::I8, is_const: false, }) {
        return false;
    }
    true
}

pub fn metadata_2(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let mut metadata = lhs_metadata[0].shape.iter().join(",");
    metadata.push_str(format!(",{:?}", lhs_metadata[0].dtype).as_str());

    let mut rhs_metadata = vec![None; 3];
    rhs_metadata[1] = Some(metadata);
    rhs_metadata
}

pub fn set_shapes_2(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    egraph.set_analysis_data(rhs_eclasses[1], lhs_metadata[0].clone());
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_3(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].shape.len() <= 1 || lhs_metadata[0].dtype != Dtype::U8 {
        return false;
    }
    true
}

pub fn metadata_3(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rs2_shape = &lhs_metadata[0].shape;
    let rs1_shape = rs2_shape.iter().fold(1, |acc, &x| acc * x);

    let mut rhs_metadata = vec![None; 3];
    rhs_metadata[1] = Some(rs1_shape.to_string());
    rhs_metadata[2] = Some(rs2_shape.iter().join(","));
    rhs_metadata
}

pub fn set_shapes_3(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let rs2_shape = &lhs_metadata[0].shape;
    let rs1_shape = rs2_shape.iter().fold(1, |acc, &x| acc * x);

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![rs1_shape], dtype: lhs_metadata[0].dtype, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_4(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].dtype == Dtype::U8 {
        return false;
    }
    true
}

pub fn metadata_4(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 1);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 3];
    rhs_metadata
}

pub fn set_shapes_4(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let dtype_size = lhs_metadata[0].dtype.size_in_bytes();
    let bitcvt_shape = match dtype_size {
        1 => lhs_metadata[0].shape.clone(),
        _ => {
            let mut shape = lhs_metadata[0].shape.clone();
            shape.extend(vec![dtype_size]);
            shape
        }
    };

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: bitcvt_shape, dtype: Dtype::U8, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
}

pub fn precond_5(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].dtype != Dtype::I8 {
        return false;
    }
    true
}

pub fn metadata_5(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let mut rhs_metadata = vec![None; 6];
    rhs_metadata[1] = Some("I32".to_string());
    rhs_metadata[3] = Some("I32".to_string());
    rhs_metadata[5] = Some("I8".to_string());
    rhs_metadata
}

pub fn set_shapes_5(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 6);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[3],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[4],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::I32, is_const: false, });
    egraph.set_analysis_data(rhs_eclasses[5],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::I8, is_const: false, });
}

pub fn precond_6(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    if lhs_metadata[1].dtype != Dtype::I8 {
        return false;
    }
    if lhs_metadata[2].dtype != Dtype::I32 {
        return false;
    }
    true
}

pub fn metadata_6(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 1];
    rhs_metadata
}

pub fn set_shapes_6(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 1);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
}

pub fn precond_7(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].dtype == Dtype::U8 {
        return false;
    }

    true
}

pub fn metadata_7(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 1);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 2];
    rhs_metadata
}

pub fn set_shapes_7(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    egraph.set_analysis_data(rhs_eclasses[1], lhs_metadata[0].clone());
}

pub fn precond_8(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[1].shape.len() != lhs_metadata[0].shape.len() {
        return false;
    }
    for (i, &dim) in lhs_metadata[1].shape.iter().enumerate() {
        if dim != lhs_metadata[0].shape[i] {
            return false;
        }
    }

    true
}

pub fn metadata_8(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 2);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 1];
    rhs_metadata
}

pub fn set_shapes_8(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 2);
    assert_eq!(rhs_eclasses.len(), 1);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
}

pub fn precond_9(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    if lhs_metadata[0].dtype != Dtype::BF16 {
        return false;
    }

    if lhs_metadata[0].shape.len() != lhs_metadata[2].shape.len() {
        return false;
    }
    for (i, &dim) in lhs_metadata[0].shape.iter().enumerate() {
        if dim != lhs_metadata[2].shape[i] {
            return false;
        }
    }

    true
}

pub fn metadata_9(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let rhs_metadata = vec![None; 1];
    rhs_metadata
}

pub fn set_shapes_9(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 1);
    let _lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
}

// Rule 10: (?x) => (multiply ?x (broadcast_? (constant_?)))
// Fires on I8 and I32 tensors. The constant value encodes the dtype (e.g. "1:I8")
// so that I8 and I32 constants remain distinct in the egraph.
pub fn precond_10(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype != Dtype::I8 && lhs_metadata[0].dtype != Dtype::I32 {
        return false;
    }
    true
}

pub fn metadata_10(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    let mut rhs_metadata = vec![None; 4];
    // Encode dtype in constant value so I8 and I32 constants stay distinct
    rhs_metadata[1] = Some(format!("1:{:?}", lhs_metadata[0].dtype));
    let shape_str = lhs_metadata[0].shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    rhs_metadata[2] = Some(shape_str);
    rhs_metadata
}

pub fn set_shapes_10(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![1], dtype: lhs_metadata[0].dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[0].clone());
}

// Rule 11: (reverse_? ?x) => (dot ?x (reverse_? eye_?))
pub fn precond_11(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype == Dtype::U8 { return false; }
    if lhs_metadata[0].shape != vec![16, 16] { return false; }
    true
}

pub fn metadata_11(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let mut metadata = lhs_metadata[0].shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    metadata.push_str(&format!(",{:?}", lhs_metadata[0].dtype));

    // RHS postorder: ?x(0), eye_?(1), reverse_?(eye_?)(2), dot(?x, reverse_?)(3)
    let mut rhs_metadata = vec![None; 4];
    rhs_metadata[1] = Some(metadata);
    rhs_metadata[2] = Some("0".to_string());
    rhs_metadata
}

pub fn set_shapes_11(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 2);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();

    // rhs_eclasses: ?x(0), eye(1), reverse(eye)(2), dot(3)
    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: lhs_metadata[0].dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[2],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: lhs_metadata[0].dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[1].clone());
}

// Rule 12: (broadcast_? ?x) => (dot (reshape_? ?x) (broadcast_? (constant_?)))
// Broadcast via outer product: broadcast(x[N,1]) = dot(x, ones[1,M])
// Requires: input is 2D with one dim == 1, output is 2D.
pub fn precond_12(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype == Dtype::U8 { return false; }
    if lhs_metadata[0].is_const { return false; }
    let input = &lhs_metadata[0];
    let output = &lhs_metadata[1];
    if output.shape.len() != 2 { return false; }
    // Input must be 2D with at least one dimension being 1 (column or row vector)
    if input.shape.len() == 2 {
        if input.shape[0] != 1 && input.shape[1] != 1 { return false; }
    } else {
        return false;
    }
    true
}

pub fn metadata_12(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let input = &lhs_metadata[0];
    let output = &lhs_metadata[1];

    // Input is [N,1] column vector, output is [N,M]
    let k = if input.shape.len() == 2 { input.shape[1] } else { 1 };
    let m = output.shape[1];

    // RHS postorder: ?x(0), constant(1), broadcast(2), dot(3)
    let mut rhs_metadata = vec![None; 4];
    rhs_metadata[1] = Some(format!("1:{:?}", input.dtype));
    rhs_metadata[2] = Some(format!("{},{}", k, m));
    rhs_metadata
}

pub fn set_shapes_12(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 2);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let dtype = lhs_metadata[0].dtype;
    let input = &lhs_metadata[0];
    let output = &lhs_metadata[1];

    let k = if input.shape.len() == 2 { input.shape[1] } else { 1 };
    let m = output.shape[1];

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![1], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[2],
        TensorInfo { shape: vec![k, m], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[1].clone());
}

// Rule 13: (reduce_? ?x) => (reshape_? (dot ?x (broadcast_? (constant_?))))
// Reduce-sum via contraction: reduce(A[N,K], dim=1) = dot(A, ones[K,1]) -> [N,1]
// Requires: input is 2D, output is 2D with one dim == 1.
// LHS postorder: ?x(0), reduce(1), reshape(2) -> 3 eclasses
pub fn precond_13(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype == Dtype::U8 { return false; }
    if lhs_metadata[0].is_const { return false; }
    if lhs_metadata[0].shape.len() != 2 { return false; }
    if lhs_metadata[2].shape.len() != 2 { return false; }
    true
}

pub fn metadata_13(
    egraph: &EGraph<TensorOp, TensorInfo>,
    lhs_eclasses: &Vec<Id>,
    _lhs_enodes: &Vec<Option<TensorOp>>,
) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let k = lhs_metadata[0].shape[1];

    // RHS postorder: ?x(0), constant(1), broadcast(2), dot(3)
    let mut rhs_metadata = vec![None; 4];
    rhs_metadata[1] = Some(format!("1:{:?}", lhs_metadata[0].dtype));
    rhs_metadata[2] = Some(format!("{},1", k));
    rhs_metadata
}

pub fn set_shapes_13(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let dtype = lhs_metadata[0].dtype;
    let k = lhs_metadata[0].shape[1];

    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![1], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[2],
        TensorInfo { shape: vec![k, 1], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[2].clone());
}

// Rule 14: (subtract ?a ?b) => (add ?a (negate ?b))
pub fn precond_14(_egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    true
}
pub fn metadata_14(_egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    vec![None; 4]
}
pub fn set_shapes_14(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[1].clone());
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[2].clone());
}

// Rule 15: (negate ?x) => (multiply ?x (broadcast_? (constant_?)))
// Creates multiply with constant -1 for mvin_spad(scale=-1).
pub fn precond_15(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype != Dtype::I8 { return false; }
    true
}
pub fn metadata_15(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 2);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_str = lhs_metadata[0].shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    let mut rhs_metadata = vec![None; 4];
    rhs_metadata[1] = Some(format!("-1:{:?}", lhs_metadata[0].dtype));
    rhs_metadata[2] = Some(shape_str);
    rhs_metadata
}
pub fn set_shapes_15(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 2);
    assert_eq!(rhs_eclasses.len(), 4);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: vec![1], dtype: lhs_metadata[0].dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[0].clone());
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[1].clone());
}

// Rule 16: (clamp ?lo ?x ?hi) => (maximum ?lo (minimum ?x ?hi))
pub fn precond_16(_egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 4);
    true
}
pub fn metadata_16(_egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 4);
    vec![None; 5]
}
pub fn set_shapes_16(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 4);
    assert_eq!(rhs_eclasses.len(), 5);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    egraph.set_analysis_data(rhs_eclasses[3], lhs_metadata[3].clone());
    egraph.set_analysis_data(rhs_eclasses[4], lhs_metadata[3].clone());
}

// Rule 17: (maximum ?a ?b) => (add (maximum (subtract ?a ?b) (broadcast_? (constant_?))) ?b)
// max(A,B) = relu(A-B) + B where relu = maximum(x, 0) matched by mvout_*_relu.
pub fn precond_17(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype == Dtype::U8 { return false; }
    if lhs_metadata[0].is_const && lhs_metadata[1].is_const { return false; }
    true
}
pub fn metadata_17(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_str = lhs_metadata[0].shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    // RHS postorder: ?a(0), ?b(1), subtract(2), constant(3), broadcast(4), maximum(5), add(6)
    let mut rhs_metadata = vec![None; 7];
    rhs_metadata[3] = Some(format!("0:{:?}", lhs_metadata[0].dtype));
    rhs_metadata[4] = Some(shape_str);
    rhs_metadata
}
pub fn set_shapes_17(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let dtype = lhs_metadata[0].dtype;
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[2].clone());
    egraph.set_analysis_data(rhs_eclasses[3],
        TensorInfo { shape: vec![1], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[4],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[5], lhs_metadata[2].clone());
    egraph.set_analysis_data(rhs_eclasses[6], lhs_metadata[2].clone());
}

// Rule 18: (minimum ?a ?b) => (subtract ?a (maximum (subtract ?a ?b) (broadcast_? (constant_?))))
// min(A,B) = A - relu(A-B)
pub fn precond_18(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    if lhs_metadata[0].dtype == Dtype::U8 { return false; }
    if lhs_metadata[0].is_const && lhs_metadata[1].is_const { return false; }
    true
}
pub fn metadata_18(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    assert_eq!(lhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let shape_str = lhs_metadata[0].shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
    // RHS postorder: ?a(0), ?b(1), subtract(2), constant(3), broadcast(4), maximum(5), subtract_outer(6)
    let mut rhs_metadata = vec![None; 7];
    rhs_metadata[3] = Some(format!("0:{:?}", lhs_metadata[0].dtype));
    rhs_metadata[4] = Some(shape_str);
    rhs_metadata
}
pub fn set_shapes_18(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 3);
    assert_eq!(rhs_eclasses.len(), 7);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    let dtype = lhs_metadata[0].dtype;
    egraph.set_analysis_data(rhs_eclasses[2], lhs_metadata[2].clone());
    egraph.set_analysis_data(rhs_eclasses[3],
        TensorInfo { shape: vec![1], dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[4],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype, is_const: true });
    egraph.set_analysis_data(rhs_eclasses[5], lhs_metadata[2].clone());
    egraph.set_analysis_data(rhs_eclasses[6], lhs_metadata[2].clone());
}

// Rule 19: (?x) => (convert_? (convert_? ?x))
// Like rule 4 (bitcvt pairs) but for pred types: introduces convert(f32, convert(pred, x))
// Enables compare_f32 to absorb compare→convert and select_f32 to absorb convert→select
pub fn precond_19(egraph: &EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>) -> bool {
    assert_eq!(lhs_eclasses.len(), 1);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    lhs_metadata[0].dtype == Dtype::Pred
}
pub fn metadata_19(_egraph: &EGraph<TensorOp, TensorInfo>, _lhs_eclasses: &Vec<Id>, _lhs_enodes: &Vec<Option<TensorOp>>) -> Vec<Option<String>> {
    // RHS postorder: ?x(0), convert_?(inner, pred→f32)(1), convert_?(outer, f32→pred)(2)
    let mut rhs_metadata = vec![None; 3];
    rhs_metadata[1] = Some("f32".to_string());
    rhs_metadata[2] = Some("pred".to_string());
    rhs_metadata
}
pub fn set_shapes_19(egraph: &mut EGraph<TensorOp, TensorInfo>, lhs_eclasses: &Vec<Id>, rhs_eclasses: &Vec<Id>) {
    assert_eq!(lhs_eclasses.len(), 1);
    assert_eq!(rhs_eclasses.len(), 3);
    let lhs_metadata: Vec<TensorInfo> = lhs_eclasses.iter().map(|id| egraph[*id].data.clone()).collect();
    // rhs[1] = inner convert: pred→f32 (same shape, f32 dtype)
    egraph.set_analysis_data(rhs_eclasses[1],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::FP32, is_const: false });
    // rhs[2] = outer convert: f32→pred (same shape, pred dtype) — unions back to original
    egraph.set_analysis_data(rhs_eclasses[2],
        TensorInfo { shape: lhs_metadata[0].shape.clone(), dtype: Dtype::Pred, is_const: false });
}
