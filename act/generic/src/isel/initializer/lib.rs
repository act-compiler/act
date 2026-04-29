use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::metadata::{MetaData, MetaDataInfo};
use crate::isel::initializer::parser::HLOExpr;

use crate::isel::initializer::parser::{
    parse_hlo_module_with_regions, parse_type_shape, HloModuleHeader,
};

pub type EggSymbolTable = HashMap<String, Id>;

macro_rules! binary_op_match {
    ($op_name:expr, $lhs:expr, $rhs:expr, $table:expr, $egraph:expr, {
        $($key:literal => $variant:path),* $(,)?
    }) => {
        match $op_name {
            $(
                $key => {
                    $egraph.add($variant([
                        *$table.get($lhs).unwrap(),
                        *$table.get($rhs).unwrap(),
                    ]))
                }
            )*
            _ => panic!("Unsupported op."),
        }
    };
}

pub fn parse_hlo_module_to_egraph<P: AsRef<Path>>(
    path: P,
) -> Result<
    (
        EGraph<TensorOp, TensorInfo>,
        Vec<(Option<Id>, i32)>,
        Id,
        HashSet<Id>,
        MetaData,
    ),
    io::Error,
> {
    let mut egraph = EGraph::<TensorOp, TensorInfo>::default();
    let mut inputs = HashSet::default();
    let mut table = EggSymbolTable::default();
    let mut offsets: Vec<(Option<Id>, i32)> = vec![];
    let mut metadata: MetaData = MetaData::default();

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read all lines into a vector
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

    // Parse the module with regions
    if let Some((header, regions, entry_instructions)) = parse_hlo_module_with_regions(&line_refs) {
        // Fill offsets from the header
        fill_offsets_from_header(&header, &mut offsets, &mut metadata);

        // Process regions (store them for future use, but don't add to egraph yet)
        // For now, we just store the regions - they'll be used when referenced in reduce operations
        let _regions = regions; // Store regions for future use

        // Process entry instructions
        let mut root: Option<Id> = None;
        let total_instructions = entry_instructions.len();

        for (i, expr) in entry_instructions.into_iter().enumerate() {
            // The last instruction is typically the ROOT operation
            let is_root = i == total_instructions - 1;

            add_expr_to_egraph(
                &expr,
                &mut egraph,
                &mut inputs,
                &mut table,
                &mut offsets,
                &mut root,
                is_root,
            );
        }

        if let Some(last) = offsets.last_mut() {
            last.0 = root;
        }

        Ok((egraph, offsets, root.unwrap(), inputs, metadata))
    } else {
        panic!("Failed to parse HLO module with regions");
    }
}

fn _should_skip_line(line: String) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("}")
        || trimmed.starts_with("ENTRY ")
    {
        return true;
    }
    false
}

fn fill_offsets_from_header(
    header: &HloModuleHeader,
    offsets: &mut Vec<(Option<Id>, i32)>,
    metadata: &mut MetaData,
) {
    metadata.module_name = header.module_name.clone();

    let mut curr_bytes: i32 = 0;
    for shape in header.input_types.iter() {
        let (ttype, tshape, _ttile) = parse_type_shape(shape);
        let dtype: Dtype = ttype.as_str().into();
        let shape_vec = shape_to_vec(&tshape);

        let mut bytes: i32 = dtype.size_in_bytes();
        shape_vec.iter().for_each(|dim| bytes *= dim);

        offsets.push((None, curr_bytes));
        metadata.input.push(MetaDataInfo {
            addr: curr_bytes,
            shape: shape_vec,
            dtype: dtype,
        });

        curr_bytes += bytes;
    }

    let output_shape = &header.output_type;
    let (ttype, tshape, _ttile) = parse_type_shape(output_shape);
    let dtype: Dtype = ttype.as_str().into();
    let shape_vec = shape_to_vec(&tshape);

    offsets.push((None, curr_bytes));
    metadata.output.push(MetaDataInfo {
        addr: curr_bytes,
        shape: shape_vec.clone(),
        dtype: dtype,
    });
}

fn add_expr_to_egraph(
    expr: &HLOExpr,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    inputs: &mut HashSet<Id>,
    table: &mut EggSymbolTable,
    offsets: &mut Vec<(Option<Id>, i32)>,
    root: &mut Option<Id>,
    parsing_root: bool,
) {
    match expr {
        HLOExpr::Parameter {
            tsymbol,
            tshape,
            ttile,
            ttype,
            tidx,
        } => {
            assert!(parsing_root == false); // parameters should not be in root
            if offsets[tidx.clone() as usize].0.is_none() {
                offsets[tidx.clone() as usize].0 = Some(add_parameter(
                    tsymbol, tshape, ttile, ttype, egraph, inputs, table,
                ));
            } else {
                panic!("Duplicate parameter index: {}", tidx);
            }
        }

        HLOExpr::Constant {
            tsymbol,
            ttype,
            tshape,
            value,
        } => {
            assert!(parsing_root == false); // parameters should not be in root
            add_constant(tsymbol, ttype, tshape, value, egraph, table);
        }

        HLOExpr::Broadcast {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_broadcast(tsymbol, ttype, tshape, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::BinaryOp {
            tsymbol,
            ttype,
            tshape,
            op_name,
            ttile,
            lhs,
            rhs,
        } => {
            add_binary_op(tsymbol, ttype, tshape, op_name, lhs, rhs, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Slice {
            tsymbol,
            ttype,
            tshape,
            source,
            ttile,
            slices,
        } => {
            add_slice(tsymbol, ttype, tshape, source, slices, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Concatenate {
            tsymbol,
            ttype,
            tshape,
            ttile,
            tinputs,
            dimensions,
        } => {
            add_concatenate(tsymbol, ttype, tshape, tinputs, dimensions, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Reshape {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_reshape(tsymbol, ttype, tshape, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Transpose {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
            dimensions,
        } => {
            add_transpose(tsymbol, ttype, tshape, source, dimensions, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Convert {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_convert(tsymbol, ttype, tshape, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::ReduceSum {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
            constant,
            dimensions,
            to_apply,
        } => {
            add_reduce(
                tsymbol, ttype, tshape, ttile, source, constant, dimensions, to_apply, egraph,
                table,
            );
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Exponential {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_exponential(tsymbol, ttype, tshape, ttile, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Negate {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_negate(tsymbol, ttype, tshape, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Clamp {
            tsymbol,
            ttype,
            tshape,
            ttile,
            lo,
            operand,
            hi,
        } => {
            add_clamp(tsymbol, ttype, tshape, lo, operand, hi, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Reverse {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
            dimensions,
        } => {
            add_reverse(tsymbol, ttype, tshape, source, dimensions, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Copy {
            tsymbol,
            ttype,
            tshape,
            ttile,
            source,
        } => {
            add_copy(tsymbol, ttype, tshape, source, egraph, table);
            if parsing_root {
                add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
            }
        }

        HLOExpr::Abs { tsymbol, ttype, tshape, ttile, source } => {
            add_unary_op(tsymbol, ttype, tshape, source, TensorOp::OpAbs, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::Logistic { tsymbol, ttype, tshape, ttile, source } => {
            add_unary_op(tsymbol, ttype, tshape, source, TensorOp::OpLogistic, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::Tanh { tsymbol, ttype, tshape, ttile, source } => {
            add_unary_op(tsymbol, ttype, tshape, source, TensorOp::OpTanh, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::LogPlusOne { tsymbol, ttype, tshape, ttile, source } => {
            add_unary_op(tsymbol, ttype, tshape, source, TensorOp::OpLogPlusOne, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::Compare { tsymbol, ttype, tshape, ttile, lhs, rhs, direction } => {
            add_compare(tsymbol, ttype, tshape, &lhs, &rhs, &direction, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::Select { tsymbol, ttype, tshape, ttile, pred, on_true, on_false } => {
            add_select(tsymbol, ttype, tshape, &pred, &on_true, &on_false, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::Iota { tsymbol, ttype, tshape, ttile, dimension } => {
            add_iota(tsymbol, ttype, tshape, &dimension, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::DynamicSlice { tsymbol, ttype, tshape, ttile, source, starts, sizes } => {
            add_dynamic_slice(tsymbol, ttype, tshape, &source, &starts, &sizes, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
        HLOExpr::DynamicUpdateSlice { tsymbol, ttype, tshape, ttile, source, update, starts } => {
            add_dynamic_update_slice(tsymbol, ttype, tshape, &source, &update, &starts, egraph, table);
            if parsing_root { add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root); }
        }
    }
}

pub fn add_parameter(
    tsymbol: &str,
    tshape: &str,
    ttile: &str,
    ttype: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    inputs: &mut HashSet<Id>,
    table: &mut EggSymbolTable,
) -> Id {
    let dtype: Dtype = ttype.into();
    let dtype_size = dtype.size_in_bytes();

    let bytes = get_bytes(tshape, dtype_size);
    if ttile == "" {
        // no tile supplied case
        let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
        egraph.set_analysis_data(
            var_node,
            TensorInfo {
                shape: vec![bytes],
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let mut shape_with_size = tshape.to_string();
        if dtype_size != 1 {
            shape_with_size += &format!(",{}", dtype_size);
        }

        let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
        let reshape_vec = shape_to_vec(&shape_with_size);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: reshape_vec.clone(),
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let node = egraph.add(TensorOp::OpBitcvt([node]));
        let bitcvt_vec = shape_to_vec(tshape);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: bitcvt_vec.clone(),
                dtype: dtype,
                is_const: false,
            },
        );

        table.insert(tsymbol.to_string(), node);
        inputs.insert(var_node);
        return var_node;
    } else {
        let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
        egraph.set_analysis_data(
            var_node,
            TensorInfo {
                shape: vec![bytes],
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let tiled_shape = get_tiled_shape(tshape, ttile);

        let mut shape_with_size = tiled_shape.to_string();
        if dtype_size != 1 {
            shape_with_size += &format!(",{}", dtype_size);
        }

        let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
        let reshape_vec = shape_to_vec(&shape_with_size);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: reshape_vec.clone(),
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let node = egraph.add(TensorOp::OpBitcvt([node]));
        let bitcvt_vec = shape_to_vec(&tiled_shape);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: bitcvt_vec.clone(),
                dtype: dtype,
                is_const: false,
            },
        );

        let transpose_order = get_transpose_order_param(tshape.split(',').count() as i32);
        let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: bitcvt_vec.clone(),
                dtype: dtype,
                is_const: false,
            },
        );

        let mut shape_with_size = tshape.to_string();
        if dtype_size != 1 {
            shape_with_size += &format!(",{}", dtype_size);
        }

        let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [node]));
        let reshape_vec = shape_to_vec(&shape_with_size);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: reshape_vec.clone(),
                dtype: dtype,
                is_const: false,
            },
        );

        table.insert(tsymbol.to_string(), node);
        inputs.insert(var_node);
        return var_node;
    }
}

fn add_constant(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    value: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpConstant(value.to_string()));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: true,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_broadcast(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpBroadcast(
        tshape.to_string(),
        [*table.get(source).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_binary_op(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    op_name: &str,
    lhs: &str,
    rhs: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = binary_op_match!(op_name, lhs, rhs, table, egraph, {
        "add" => TensorOp::OpAdd,
        "subtract" => TensorOp::OpSubtract,
        "multiply" => TensorOp::OpMultiply,
        "minimum" => TensorOp::OpMinimum,
        "maximum" => TensorOp::OpMaximum,
        "xor" => TensorOp::OpXor,
        "shift-right-logical" => TensorOp::OpShiftRightLogical,
        "shift-right-arithmetic" => TensorOp::OpShiftRightArithmetic,
        "or" => TensorOp::OpOr,
        "and" => TensorOp::OpAnd,
        "shift-left" => TensorOp::OpShiftLeft,
        "dot" => TensorOp::OpDot,
        "divide" => TensorOp::OpDivide,
        "power" => TensorOp::OpPower
    });

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_slice(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    slices: &Vec<String>,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let slice_str = slices
        .iter()
        .map(|s| {
            let trimmed = s.trim_matches(|c| c == '[' || c == ']');
            // remove stride component - only keep start:end
            let parts: Vec<&str> = trimmed.split(':').collect();
            if parts.len() >= 2 {
                format!("{}:{}", parts[0], parts[1])
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<String>>()
        .join(";");

    let node = egraph.add(TensorOp::OpSlice(
        slice_str.to_string(),
        [*table.get(source).unwrap()],
    ));
    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_concatenate(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    inputs: &Vec<String>,
    dimensions: &Vec<u32>,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let dim_str = dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<String>>()
        .join(",");
    let node = egraph.add(TensorOp::OpConcat(
        dim_str.clone(),
        [
            *table.get(&inputs[0]).unwrap(),
            *table.get(&inputs[1]).unwrap(),
        ],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_reshape(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpReshape(
        tshape.to_string(),
        [*table.get(source).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_transpose(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    dimensions: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpTranspose(
        dimensions.to_string(),
        [*table.get(source).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_convert(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpConvert(
        ttype.to_string(),
        [*table.get(source).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_reduce(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    _ttile: &str,
    source: &str,
    _constant: &str,
    dimensions: &str,
    _to_apply: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpReduceSum(
        dimensions.to_string(),
        [*table.get(source).unwrap()],
    ));
    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_exponential(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    _ttile: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpExp([*table.get(source).unwrap()]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_negate(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpNegate([*table.get(source).unwrap()]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_copy(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpCopy([*table.get(source).unwrap()]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_unary_op(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    op_constructor: fn([Id; 1]) -> TensorOp,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(op_constructor([*table.get(source).unwrap()]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_compare(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    lhs: &str,
    rhs: &str,
    direction: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpCompare(
        direction.to_string(),
        [*table.get(lhs).unwrap(), *table.get(rhs).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_select(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    pred: &str,
    on_true: &str,
    on_false: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpSelect([
        *table.get(pred).unwrap(),
        *table.get(on_true).unwrap(),
        *table.get(on_false).unwrap(),
    ]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_iota(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    dimension: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpIota(dimension.to_string()));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn resolve_constant_value(
    egraph: &EGraph<TensorOp, TensorInfo>,
    table: &EggSymbolTable,
    name: &str,
) -> i64 {
    let id = *table.get(name).expect(&format!("Symbol {} not found in table", name));
    for en in egraph[id].nodes.iter() {
        if let TensorOp::OpConstant(val) = en {
            return val.parse::<i64>().unwrap_or(0);
        }
    }
    0
}

fn add_dynamic_slice(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    starts: &[String],
    sizes: &[String],
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let mut slice_parts = Vec::new();
    for (start_name, size_str) in starts.iter().zip(sizes.iter()) {
        let start = resolve_constant_value(egraph, table, start_name);
        let size: i64 = size_str.parse().unwrap_or(0);
        slice_parts.push(format!("{}:{}", start, start + size));
    }
    let slice_str = slice_parts.join(";");

    let node = egraph.add(TensorOp::OpSlice(
        slice_str,
        [*table.get(source).unwrap()],
    ));
    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_dynamic_update_slice(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    update: &str,
    starts: &[String],
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let update_shape = egraph[*table.get(update).unwrap()].data.shape.clone();
    let mut slice_parts = Vec::new();
    for (i, start_name) in starts.iter().enumerate() {
        let start = resolve_constant_value(egraph, table, start_name);
        let size = update_shape.get(i).copied().unwrap_or(0) as i64;
        slice_parts.push(format!("{}:{}", start, start + size));
    }
    let slice_str = slice_parts.join(";");

    let node = egraph.add(TensorOp::OpUpdateSlice(
        slice_str,
        [*table.get(source).unwrap(), *table.get(update).unwrap()],
    ));
    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_clamp(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    lo: &str,
    operand: &str,
    hi: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpClamp([
        *table.get(lo).unwrap(),
        *table.get(operand).unwrap(),
        *table.get(hi).unwrap(),
    ]));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_reverse(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    source: &str,
    dimensions: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
) {
    let dtype: Dtype = ttype.into();
    let shape = shape_to_vec(tshape);

    let node = egraph.add(TensorOp::OpReverse(
        dimensions.to_string(),
        [*table.get(source).unwrap()],
    ));

    egraph.set_analysis_data(
        node,
        TensorInfo {
            shape: shape.clone(),
            dtype,
            is_const: false,
        },
    );

    table.insert(tsymbol.to_string(), node);
}

fn add_root_operations(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    ttile: &str,
    egraph: &mut EGraph<TensorOp, TensorInfo>,
    table: &mut EggSymbolTable,
    root: &mut Option<Id>,
) {
    let dtype: Dtype = ttype.into();
    let dtype_size = dtype.size_in_bytes();
    let bytes = get_bytes(tshape, dtype_size);

    if ttile == "" {
        let mut shape_with_size = tshape.to_string();
        if dtype_size != 1 {
            shape_with_size += &format!(",{}", dtype_size);
        }

        let node = table.get(tsymbol);

        let node = egraph.add(TensorOp::OpBitcvt([*node.unwrap()]));
        let bitcvt_vec = shape_to_vec(&shape_with_size);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: bitcvt_vec.clone(),
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: vec![bytes],
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        table.insert(tsymbol.to_string(), node);
        *root = Some(node);
    } else {
        let node = table.get(tsymbol);
        let tiled_shape = get_result_shape(tshape, ttile);

        let node = egraph.add(TensorOp::OpReshape(tiled_shape.clone(), [*node.unwrap()]));
        let reshape_vec = shape_to_vec(&tiled_shape);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: reshape_vec.clone(),
                dtype: dtype,
                is_const: false,
            },
        );

        let transpose_order = get_transpose_order_result(tshape.split(',').count() as i32);
        let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));

        let transposed_reshape_str = apply_transpose_order(&tiled_shape, &transpose_order);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: shape_to_vec(&transposed_reshape_str),
                dtype: dtype,
                is_const: false,
            },
        );

        let mut shape_with_size = transposed_reshape_str.to_string();
        if dtype_size != 1 {
            shape_with_size += &format!(",{}", dtype_size);
        }

        let node = egraph.add(TensorOp::OpBitcvt([node]));
        let bitcvt_vec = shape_to_vec(&shape_with_size);
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: bitcvt_vec.clone(),
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
        egraph.set_analysis_data(
            node,
            TensorInfo {
                shape: vec![bytes],
                dtype: Dtype::U8,
                is_const: false,
            },
        );

        table.insert(tsymbol.to_string(), node);
        *root = Some(node);
    }
}

fn shape_to_vec(shape_str: &str) -> Vec<i32> {
    let cleaned = shape_str.trim();

    if cleaned.is_empty() || cleaned == "[]" {
        return vec![1];
    }

    let trimmed = cleaned.trim_matches(|c| c == '[' || c == ']');

    if trimmed.trim().is_empty() {
        return vec![1];
    }

    trimmed
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|x| x.trim().parse::<i32>().expect("Invalid shape dimension"))
        .collect()
}

fn get_bytes(shape: &str, dtype: i32) -> i32 {
    let mut bytes = 1;
    shape_to_vec(shape).iter().for_each(|dim| bytes *= dim);
    bytes * dtype
}

fn get_tiled_shape(shape: &str, tile: &str) -> String {
    let shape_vec: Vec<u32> = shape
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid shape number"))
        .collect();

    let tile_vec: Vec<u32> = tile
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid tile number"))
        .collect();

    if shape_vec.len() != tile_vec.len() {
        panic!("Shape and tile must have the same number of dimensions");
    }

    let divided: Vec<String> = shape_vec
        .iter()
        .zip(tile_vec.iter())
        .map(|(s, t)| (s / t).to_string())
        .collect();

    let mut result = divided.join(",");
    result.push(',');
    result.push_str(
        &tile_vec
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );

    result
}

fn get_result_shape(tshape: &str, ttile: &str) -> String {
    let dims: Vec<i32> = tshape
        .split(',')
        .map(|x| x.trim().parse::<i32>().unwrap())
        .collect();
    let tiles: Vec<i32> = ttile
        .split(',')
        .map(|x| x.trim().parse::<i32>().unwrap())
        .collect();

    assert_eq!(dims.len(), tiles.len(), "Shape and tile length mismatch");

    let mut result = Vec::new();
    for (d, t) in dims.iter().zip(tiles.iter()) {
        result.push((d / t).to_string()); // integer division
        result.push(t.to_string());
    }

    result.join(",")
}

fn get_transpose_order_param(num_dims: i32) -> String {
    let mut order = Vec::with_capacity((num_dims * 2) as usize);
    for i in 0..num_dims {
        order.push(i + 1); // outer dimension (1-based)
        order.push(i + 1 + num_dims); // tile dimension (1-based + offset)
    }

    order
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(",")
}

fn get_transpose_order_result(n: i32) -> String {
    let mut odds = (1..=2 * n).step_by(2).collect::<Vec<_>>();
    let evens = (2..=2 * n).step_by(2).collect::<Vec<_>>();

    odds.extend(evens);

    odds.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn apply_transpose_order(reshape_str: &str, order_str: &str) -> String {
    let dims: Vec<&str> = reshape_str.split(',').map(|x| x.trim()).collect();
    let order: Vec<usize> = order_str
        .split(',')
        .map(|x| x.trim().parse::<usize>().unwrap())
        .collect();

    let new_dims: Vec<&str> = order.iter().map(|&i| dims[i - 1]).collect();
    new_dims.join(",")
}
