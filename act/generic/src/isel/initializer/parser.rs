pub fn parse_type_shape(type_shape: &str) -> (String, String, String) {
    let ttype_end = type_shape.find('[').unwrap_or(type_shape.len());
    let ttype = type_shape[..ttype_end].to_string();

    let mut tshape;
    let mut ttile = String::new();
    if let Some(bracket_start) = type_shape.find('[') {
        // Find end of shape (could be before tile annotation)
        if let Some(bracket_end) = type_shape[bracket_start..].find(']') {
            let bracket_end = bracket_start + bracket_end;
            tshape = type_shape[bracket_start + 1..=bracket_end - 1].to_string();
            // Check for tile annotation after shape
            if let Some(tile_start) = type_shape[bracket_end + 1..].find('{') {
                let tile_start = bracket_end + 1 + tile_start;
                let tile_str = &type_shape[tile_start..];
                if let Some(start_paren) = tile_str.find('(') {
                    if let Some(end_paren) = tile_str.find(')') {
                        ttile = tile_str[start_paren + 1..end_paren].to_string();
                    } else {
                        // No closing ')', fallback
                        // ttile = tile_str[start_paren + 1..].to_string();
                        // DJ: this should never happen
                        panic!("Invalid ttype string: {}", type_shape);
                    }
                }
            }
        } else {
            // No closing ']', fallback
            // tshape = type_shape[bracket_start..].to_string();
            // DJ: this should never happen
            panic!("Invalid ttype string: {}", type_shape);
        }
    } else {
        // tshape = "[]".to_string();
        // DJ: this should never happen
        panic!("Invalid ttype string: {}", type_shape);
    }
    if tshape.is_empty() {
        // Handle scalar types like bf16[] - they have empty shape
        tshape = "[]".to_string();
    }
    (ttype, tshape, ttile)
}

pub fn parse_line(line: &str) -> Option<(String, String, String, String, String, String, String)> {
    let line = line.trim();

    // 0. Ignore empty lines or comments
    if line.is_empty() || line.starts_with("//") {
        return None;
    }

    // 1. Split at '=' (exactly one)
    let mut parts = line.splitn(2, '=');
    let mut tsymbol = parts.next()?.trim().to_string();

    // Check if this is a ROOT operation and trim the prefix
    let is_root = tsymbol.starts_with("ROOT ");
    if is_root {
        tsymbol = tsymbol.replacen("ROOT ", "", 1).trim().to_string();
    }

    let rest = parts.next()?.trim();

    // 2. rest starts with type + shape + operator(operands) + rest

    // Find the first space to get type+shape
    let first_space = rest.find(' ')?;
    let (type_shape, after_type_shape) = rest.split_at(first_space);
    let after_type_shape = after_type_shape.trim_start();

    // 3. Extract type, shape, and tile annotation from type_shape (e.g. u64[3,8]{;T(16,4)})
    let (ttype, tshape, ttile) = parse_type_shape(type_shape);

    // 4. Extract operator name before '('
    let op_start = after_type_shape.find('(')?;
    let toperator = after_type_shape[..op_start].trim().to_string();

    // 5. Extract operands inside matching parentheses (find ')' after op_start)
    let operands_start = op_start + 1;
    let mut paren_count = 1;
    let mut operands_end = operands_start;

    for (i, c) in after_type_shape[operands_start..].chars().enumerate() {
        match c {
            '(' => paren_count += 1,
            ')' => {
                paren_count -= 1;
                if paren_count == 0 {
                    operands_end = operands_start + i;
                    break;
                }
            }
            _ => {}
        }
    }

    let toperands = after_type_shape[operands_start..operands_end]
        .trim()
        .to_string();

    // 6. Everything after the closing ')' is the rest
    let trest = after_type_shape[operands_end + 1..].trim().to_string();

    Some((tsymbol, ttype, tshape, ttile, toperator, toperands, trest))
}

#[derive(Debug, Clone)]
pub enum HLOExpr {
    Parameter {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        tidx: u32,
    },
    Constant {
        tsymbol: String,
        ttype: String,
        tshape: String,
        value: String,
    },
    Broadcast {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Slice {
        tsymbol: String,
        ttype: String,
        tshape: String,
        source: String,
        ttile: String,
        slices: Vec<String>,
    },
    BinaryOp {
        tsymbol: String,
        ttype: String,
        tshape: String,
        op_name: String,
        ttile: String,
        lhs: String,
        rhs: String,
    },
    Concatenate {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        tinputs: Vec<String>,
        dimensions: Vec<u32>,
    },
    Reshape {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Transpose {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
        dimensions: String,
    },
    Convert {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    ReduceSum {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
        constant: String,
        dimensions: String,
        to_apply: String,
    },
    Exponential {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Negate {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Clamp {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        lo: String,
        operand: String,
        hi: String,
    },
    Reverse {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
        dimensions: String,
    },
    Copy {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Abs {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Logistic {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Tanh {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    LogPlusOne {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
    },
    Compare {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        lhs: String,
        rhs: String,
        direction: String,
    },
    Select {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        pred: String,
        on_true: String,
        on_false: String,
    },
    Iota {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        dimension: String,
    },
    DynamicSlice {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
        starts: Vec<String>,
        sizes: Vec<String>,
    },
    DynamicUpdateSlice {
        tsymbol: String,
        ttype: String,
        tshape: String,
        ttile: String,
        source: String,
        update: String,
        starts: Vec<String>,
    },
}

pub fn validate_region_exists(
    region_name: &str,
    regions: &std::collections::HashMap<String, HLORegion>,
) -> bool {
    if !regions.contains_key(region_name) {
        panic!("Region {} does not exist", region_name);
    }

    // region only contains parameter and add instructions
    // @todo check shape here and type
    for instruction in regions.get(region_name).unwrap().instructions.iter() {
        match instruction {
            HLOExpr::Parameter { .. } => {}
            HLOExpr::BinaryOp { op_name, .. } if op_name == "add" => {}
            _ => {
                panic!(
                    "Region {} contains non-parameter and non-add instructions",
                    region_name
                );
            }
        }
    }
    true
}

pub fn parse_hlo_expr(
    tsymbol: &str,
    ttype: &str,
    tshape: &str,
    ttile: &str,
    toperator: &str,
    toperands: &str,
    trest: &str,
    regions: &std::collections::HashMap<String, HLORegion>,
) -> Option<HLOExpr> {
    match toperator {
        "parameter" => {
            // Parameter(0) -- parse the index from operands
            let tidx = toperands.trim().parse::<u32>().ok()?;
            Some(HLOExpr::Parameter {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                tidx,
            })
        }

        "constant" => {
            // constant(1002)
            let value = toperands.trim().to_string();
            Some(HLOExpr::Constant {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                value,
            })
        }

        "broadcast" => {
            // broadcast(Arg_0.6)
            let source = toperands.trim().to_string();
            Some(HLOExpr::Broadcast {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }

        "slice" => {
            // slice(Arg_0.0), slice={[0:1:1], [0:8:1]}
            // Operands before comma are source, after comma look for slice spec in rest
            let source = toperands.trim().to_string();

            // Parse slices from rest string like "slice={[0:1:1], [0:8:1]}"
            // We'll parse what's inside {...}
            let slices = if let Some(start) = trest.find('{') {
                if let Some(end) = trest.find('}') {
                    let slice_str = &trest[start + 1..end];
                    slice_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

            Some(HLOExpr::Slice {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                slices,
            })
        }

        "add"
        | "subtract"
        | "multiply"
        | "minimum"
        | "maximum"
        | "xor"
        | "shift-right-logical"
        | "shift-right-arithmetic"
        | "mul"
        | "div"
        | "and"
        | "or"
        | "shift-left"
        | "dot"
        | "divide"
        | "power" => {
            // binary operators with two operands separated by comma
            let mut parts = toperands.split(',').map(|s| s.trim().to_string());
            let lhs = parts.next().unwrap_or_default();
            let rhs = parts.next().unwrap_or_default();

            Some(HLOExpr::BinaryOp {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                op_name: toperator.to_string(),
                lhs,
                rhs,
            })
        }

        "concatenate" => {
            // concatenate({a, b, c}), dimension=0 or similar, parse inputs and dims

            // Inputs are comma-separated in operands, remove braces if any
            let inputs = toperands
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>();

            // Parse dimension from rest e.g. dimension=0 or dims=[0,1]
            let dimensions = trest
                .chars()
                .filter_map(|c| c.to_digit(10))
                .collect::<Vec<_>>();

            Some(HLOExpr::Concatenate {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                tinputs: inputs,
                dimensions,
            })
        }

        "reshape" => {
            // reshape(Arg_0.6), new_shape=[1, 2, 3]
            let source = toperands.trim().to_string();

            Some(HLOExpr::Reshape {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }

        "transpose" => {
            let dim_str = trest
                .split_once('{')
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(inner, _)| inner.trim())
                .unwrap_or("");

            let source = toperands.trim().to_string();

            Some(HLOExpr::Transpose {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                dimensions: dim_str.to_string(),
            })
        }

        "convert" => {
            let source = toperands.trim().to_string();
            Some(HLOExpr::Convert {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }

        "reduce" => {
            let _source = toperands.trim().to_string();

            // Parse dimensions from something like: dimensions={1}
            let dimensions = trest
                .split("dimensions=")
                .nth(1)
                .and_then(|s| s.split(',').next()) // Up to next comma or end
                .map(|s| s.trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace()))
                .unwrap_or("")
                .to_string();

            // Parse reduction operation from something like: to_apply=%region_0.10, metadata=...
            let to_apply = trest
                .split("to_apply=")
                .nth(1)
                .map(|s| {
                    // Find the next comma that's not inside brackets/parentheses
                    let mut depth = 0;
                    let mut in_brackets = false;
                    let mut in_parens = false;

                    for (i, ch) in s.char_indices() {
                        match ch {
                            '[' => in_brackets = true,
                            ']' => in_brackets = false,
                            '(' => {
                                if !in_brackets {
                                    in_parens = true;
                                    depth += 1;
                                }
                            }
                            ')' => {
                                if !in_brackets {
                                    depth -= 1;
                                    if depth == 0 {
                                        in_parens = false;
                                    }
                                }
                            }
                            ',' if !in_brackets && !in_parens => {
                                // Found a top-level comma, take everything before it
                                return s[..i].trim().to_string();
                            }
                            _ => {}
                        }
                    }
                    // No comma found, take the whole string
                    s.trim().to_string()
                })
                .unwrap_or("".to_string())
                .to_string();

            validate_region_exists(&to_apply, regions);

            // Parse the operands to separate source and constant
            let operands: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            if operands.len() != 2 {
                panic!(
                    "Reduce operation expects exactly 2 operands, got {}: {}",
                    operands.len(),
                    toperands
                );
            }

            let source = operands[0].to_string();
            let constant = operands[1].to_string();

            Some(HLOExpr::ReduceSum {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                constant,
                dimensions,
                to_apply,
            })
        }
        "exponential" => {
            let source = toperands.trim().to_string();
            Some(HLOExpr::Exponential {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }
        "negate" => {
            let source = toperands.trim().to_string();
            Some(HLOExpr::Negate {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }
        "copy" => {
            let source = toperands.trim().to_string();
            Some(HLOExpr::Copy {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
            })
        }
        "clamp" => {
            let parts: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            if parts.len() != 3 {
                panic!(
                    "Clamp operation expects exactly 3 operands, got {}: {}",
                    parts.len(),
                    toperands
                );
            }
            Some(HLOExpr::Clamp {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                lo: parts[0].to_string(),
                operand: parts[1].to_string(),
                hi: parts[2].to_string(),
            })
        }
        "reverse" => {
            let source = toperands.trim().to_string();
            let dim_str = trest
                .split_once('{')
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(inner, _)| inner.trim())
                .unwrap_or("");
            Some(HLOExpr::Reverse {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                dimensions: dim_str.to_string(),
            })
        }
        "abs" | "logistic" | "tanh" | "log-plus-one" => {
            let source = toperands.trim().to_string();
            let variant = match toperator {
                "abs" => HLOExpr::Abs {
                    tsymbol: tsymbol.to_string(), ttype: ttype.to_string(),
                    tshape: tshape.to_string(), ttile: ttile.to_string(), source,
                },
                "logistic" => HLOExpr::Logistic {
                    tsymbol: tsymbol.to_string(), ttype: ttype.to_string(),
                    tshape: tshape.to_string(), ttile: ttile.to_string(), source,
                },
                "tanh" => HLOExpr::Tanh {
                    tsymbol: tsymbol.to_string(), ttype: ttype.to_string(),
                    tshape: tshape.to_string(), ttile: ttile.to_string(), source,
                },
                "log-plus-one" => HLOExpr::LogPlusOne {
                    tsymbol: tsymbol.to_string(), ttype: ttype.to_string(),
                    tshape: tshape.to_string(), ttile: ttile.to_string(), source,
                },
                _ => unreachable!(),
            };
            Some(variant)
        }
        "compare" => {
            let parts: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            let direction = trest
                .split("direction=")
                .nth(1)
                .and_then(|s| s.split(|c: char| c == ',' || c.is_whitespace()).next())
                .unwrap_or("")
                .trim()
                .to_string();
            Some(HLOExpr::Compare {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                lhs: parts[0].to_string(),
                rhs: parts[1].to_string(),
                direction,
            })
        }
        "select" => {
            let parts: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            Some(HLOExpr::Select {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                pred: parts[0].to_string(),
                on_true: parts[1].to_string(),
                on_false: parts[2].to_string(),
            })
        }
        "iota" => {
            let dimension = trest
                .split("iota_dimension=")
                .nth(1)
                .and_then(|s| s.split(|c: char| c == ',' || c.is_whitespace()).next())
                .unwrap_or("0")
                .trim()
                .to_string();
            Some(HLOExpr::Iota {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                dimension,
            })
        }
        "dynamic-slice" => {
            // dynamic-slice(input, start0, start1), dynamic_slice_sizes={128,128}
            let parts: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            let source = parts[0].to_string();
            let starts: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
            let sizes_str = trest
                .split("dynamic_slice_sizes=")
                .nth(1)
                .and_then(|s| {
                    let s = s.trim_start_matches('{');
                    s.split('}').next()
                })
                .unwrap_or("");
            let sizes: Vec<String> = sizes_str.split(',').map(|s| s.trim().to_string()).collect();
            Some(HLOExpr::DynamicSlice {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                starts,
                sizes,
            })
        }
        "dynamic-update-slice" => {
            // dynamic-update-slice(input, update, start0, start1)
            let parts: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
            let source = parts[0].to_string();
            let update = parts[1].to_string();
            let starts: Vec<String> = parts[2..].iter().map(|s| s.to_string()).collect();
            Some(HLOExpr::DynamicUpdateSlice {
                tsymbol: tsymbol.to_string(),
                ttype: ttype.to_string(),
                tshape: tshape.to_string(),
                ttile: ttile.to_string(),
                source,
                update,
                starts,
            })
        }
        _ => {
            panic!("Unknown operator: {}", toperator);
        }
    }
}

#[derive(Debug, Clone)]
pub struct HLORegion {
    pub name: String,
    pub instructions: Vec<HLOExpr>,
}

#[derive(Debug)]
pub struct HloModuleHeader {
    pub module_name: String,
    pub input_types: Vec<String>,
    pub output_type: String,
}

pub fn parse_hlo_module_header_hand(line: &str) -> Option<HloModuleHeader> {
    let line = line.trim();

    // Step 1: Extract module name and entry computation layout
    let prefix = "HloModule ";
    if !line.starts_with(prefix) {
        return None;
    }
    let rest = &line[prefix.len()..];

    // Get module name (first part before comma)
    let module_name = rest.split(',').next().unwrap_or("").trim().to_string();

    // Find entry_computation_layout parameter
    let layout_start = "entry_computation_layout={";
    let layout_part = if let Some(start_pos) = rest.find(layout_start) {
        &rest[start_pos + layout_start.len()..]
    } else {
        return None;
    };

    // Find matching closing brace
    let mut brace_count = 1;
    let mut layout_end = 0;
    for (i, ch) in layout_part.char_indices() {
        match ch {
            '{' => brace_count += 1,
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    layout_end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    if brace_count != 0 {
        return None;
    }

    let entry_layout = &layout_part[..layout_end];

    // Step 2: Remove layout specifications like {1,0}
    let mut layout_cleaned = entry_layout.to_string();
    while let Some(start) = layout_cleaned.find('{') {
        if let Some(end) = layout_cleaned[start..].find('}') {
            let end_pos = start + end;
            layout_cleaned = format!(
                "{}{}",
                &layout_cleaned[..start],
                &layout_cleaned[end_pos + 1..]
            );
        } else {
            break;
        }
    }

    // Step 3: Process the cleaned layout
    let layout_no_spaces = layout_cleaned.replace(" ", "");

    // Assume format: (input_types) -> output_type (no parentheses around output)
    let mut normalized_layout = layout_no_spaces.clone();
    if let Some(close_paren_pos) = normalized_layout.find(')') {
        if let Some(arrow_pos) = normalized_layout[close_paren_pos..].find("->") {
            let actual_arrow_pos = close_paren_pos + arrow_pos;
            normalized_layout = format!(
                "{}|{}",
                &normalized_layout[..close_paren_pos],
                &normalized_layout[actual_arrow_pos + 2..] // Skip the "->"
            );
        } else {
            panic!(
                "Expected format: (input_types) -> output_type. Found: {}",
                layout_no_spaces
            );
        }
    } else {
        panic!(
            "Expected format: (input_types) -> output_type. Found: {}",
            layout_no_spaces
        );
    }

    let io_parts: Vec<&str> = normalized_layout.split('|').collect();
    if io_parts.len() != 2 {
        return None;
    }

    // Extract inputs and output
    let inputs_str = io_parts[0].strip_prefix('(').unwrap_or(io_parts[0]);
    let output_type = io_parts[1].strip_suffix(')').unwrap_or(io_parts[1]);

    // 6) split inputs on top-level commas (ignore commas inside brackets)
    fn split_top_level_commas(s: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut buf = String::new();
        let mut depth = 0usize; // bracket depth for '[' and ']'

        for ch in s.chars() {
            match ch {
                '[' => {
                    depth += 1;
                    buf.push(ch);
                }
                ']' => {
                    if depth > 0 {
                        depth -= 1;
                    }
                    buf.push(ch);
                }
                ',' if depth == 0 => {
                    let piece = buf.trim();
                    if !piece.is_empty() {
                        parts.push(piece.to_string());
                    }
                    buf.clear();
                }
                _ => buf.push(ch),
            }
        }
        let piece = buf.trim();
        if !piece.is_empty() {
            parts.push(piece.to_string());
        }
        parts
    }

    let input_types = if inputs_str.is_empty() {
        Vec::new()
    } else {
        split_top_level_commas(inputs_str)
    };

    Some(HloModuleHeader {
        module_name,
        input_types,
        output_type: output_type.to_string(),
    })
}

pub fn parse_region(
    lines: &[&str],
    start_idx: usize,
    regions: &std::collections::HashMap<String, HLORegion>,
) -> Option<(HLORegion, usize)> {
    if start_idx >= lines.len() {
        return None;
    }

    let first_line = lines[start_idx].trim();

    // Check if this is a region definition (starts with region name and has opening brace)
    let brace_pos = first_line.find('{');
    if brace_pos.is_none() {
        return None;
    }

    let region_name_with_signature = first_line[..brace_pos.unwrap()].trim();

    // Extract just the region name (before the function signature)
    // Format: %region_name (params) -> return_type
    let region_name = if let Some(paren_pos) = region_name_with_signature.find('(') {
        region_name_with_signature[..paren_pos].trim().to_string()
    } else {
        region_name_with_signature.to_string()
    };
    let mut instructions = Vec::new();
    let mut idx = start_idx + 1;
    let mut brace_count = 1; // We already found the opening brace

    // Parse until we find the matching closing brace
    while idx < lines.len() && brace_count > 0 {
        let line = lines[idx].trim();

        // Count braces to handle nested regions
        for ch in line.chars() {
            match ch {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => {}
            }
        }

        // If this line contains an instruction (not just braces), parse it
        if !line.is_empty()
            && !line
                .chars()
                .all(|c| c == '{' || c == '}' || c.is_whitespace())
        {
            // Remove the closing brace if it's on this line
            let instruction_line = if line.ends_with('}') && brace_count == 0 {
                line.strip_suffix('}').unwrap_or(line).trim()
            } else {
                line
            };

            if !instruction_line.is_empty() {
                if let Some((tsymbol, ttype, tshape, ttile, toperator, toperands, trest)) =
                    parse_line(instruction_line)
                {
                    if let Some(expr) = parse_hlo_expr(
                        &tsymbol, &ttype, &tshape, &ttile, &toperator, &toperands, &trest, regions,
                    ) {
                        instructions.push(expr);
                    }
                }
            }
        }

        idx += 1;
    }

    Some((
        HLORegion {
            name: region_name,
            instructions,
        },
        idx,
    ))
}

pub fn parse_hlo_module_with_regions(
    lines: &[&str],
) -> Option<(
    HloModuleHeader,
    std::collections::HashMap<String, HLORegion>,
    Vec<HLOExpr>,
)> {
    if lines.is_empty() {
        return None;
    }
    // Parse the module header from the first line
    let header = match parse_hlo_module_header_hand(lines[0]) {
        Some(h) => h,
        None => return None,
    };

    let mut regions = std::collections::HashMap::new();
    let mut entry_instructions = Vec::new();
    let mut idx = 1;

    // Parse regions and entry computation
    while idx < lines.len() {
        let line = lines[idx].trim();

        if line.is_empty() {
            idx += 1;
            continue;
        }

        if line.starts_with("ENTRY") {
            // Parse entry computation
            idx += 1; // Skip the ENTRY line
            while idx < lines.len() {
                let instruction_line = lines[idx].trim();
                if instruction_line.is_empty() {
                    idx += 1;
                    continue;
                }

                if instruction_line == "}" {
                    break; // End of entry computation
                }

                if let Some((tsymbol, ttype, tshape, ttile, toperator, toperands, trest)) =
                    parse_line(instruction_line)
                {
                    if let Some(expr) = parse_hlo_expr(
                        &tsymbol, &ttype, &tshape, &ttile, &toperator, &toperands, &trest, &regions,
                    ) {
                        entry_instructions.push(expr);
                    }
                }
                idx += 1;
            }
        } else if line.contains('{') {
            // This is a region definition
            if let Some((region, new_idx)) = parse_region(lines, idx, &regions) {
                regions.insert(region.name.clone(), region);
                idx = new_idx;
            } else {
                idx += 1;
            }
        } else {
            idx += 1;
        }
    }

    Some((header, regions, entry_instructions))
}
