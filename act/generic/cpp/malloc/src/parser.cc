#include "parser.h"

namespace parser {

ParsedLine parse_line(const std::string &line) {
  ParsedLine pl;

  // Helper lambdas
  // (i) panic with error message
  auto panic = [&](const std::string &msg) -> ParsedLine {
    std::cerr << "parse_line error: " << msg << "\n  line: `" << line << "`"
              << std::endl;
    assert(false && "unreachable");
    ParsedLine pl;
    return pl;
  };
  // (ii) trim whitespace from both ends of a string
  auto trim = [](std::string &s) {
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
  };

  // parse line: lhs: rest
  // find first ':'
  auto colon_pos = line.find(':');
  if (colon_pos == std::string::npos)
    return panic("missing ':' separator");

  std::string lhs = line.substr(0, colon_pos);
  std::string rest = line.substr(colon_pos + 1);
  trim(lhs);
  trim(rest);

  pl.lhs_variable_name = lhs; // Processed

  // parse rest: left = right
  // find first '='
  auto eq_pos = rest.find('=');
  if (eq_pos == std::string::npos)
    return panic("missing '=' separator");

  std::string left = rest.substr(0, eq_pos);
  std::string right = rest.substr(eq_pos + 1);
  trim(left);
  trim(right);

  // parse left: storage[offset]
  // find first '[' and ']'
  auto br_open = left.find('[');
  auto br_close = left.find(']');
  if (br_open == std::string::npos || br_close == std::string::npos ||
      br_close < br_open)
    return panic("missing '[' or ']' for storage offset");

  std::string storage_name = left.substr(0, br_open);
  std::string addr_str = left.substr(br_open + 1, br_close - br_open - 1);
  trim(storage_name);
  trim(addr_str);

  pl.lhs_storage_name = storage_name; // Processed

  int64 offset = -1;
  try {
    offset = std::stoll(addr_str);
  } catch (...) {
    return panic("failed to parse storage offset as integer");
  }

  pl.lhs_offset = offset; // Processed

  // parse right: datatype[shape] <operation>
  // find first space
  auto first_space = right.find(' ');
  if (first_space == std::string::npos)
    return panic("missing type/shape before operation (no space found)");

  std::string type_shape = right.substr(0, first_space);
  std::string op_and_rest = right.substr(first_space + 1);
  trim(type_shape);
  trim(op_and_rest);

  // parse type_shape: dtype[shape]
  // find first '[' and ']'
  auto s_brkt = type_shape.find('[');
  auto e_brkt = type_shape.find(']');
  if (s_brkt == std::string::npos || e_brkt == std::string::npos ||
      e_brkt < s_brkt)
    return panic("missing '[' or ']' for type shape");

  // std::string dtype = type_shape.substr(0, s_brkt); // Not needed
  std::string shape_str = type_shape.substr(s_brkt + 1, e_brkt - s_brkt - 1);
  // trim(dtype); // Not needed
  trim(shape_str);

  std::vector<int64> shape_vec;
  if (!shape_str.empty()) {
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      trim(tok);
      if (!tok.empty()) {
        try {
          shape_vec.push_back(std::stoll(tok));
        } catch (...) {
          return panic("failed to parse shape as integer vector");
        }
      }
    }
  }

  pl.lhs_shape = shape_vec; // Processed

  // parse op_and_rest: op_name_attr(operands)
  // find last '(' and ')'
  auto s_prn = op_and_rest.rfind('(');
  auto e_prn = op_and_rest.rfind(')');
  if (s_prn == std::string::npos || e_prn == std::string::npos ||
      e_prn <= s_prn)
    return panic("missing '(' or ')' for operands");

  std::string op_name_and_attr = op_and_rest.substr(0, s_prn);
  std::string operands_str = op_and_rest.substr(s_prn + 1, e_prn - s_prn - 1);
  trim(op_name_and_attr);
  trim(operands_str);

  std::vector<std::string> operands_vec;
  if (!operands_str.empty()) {
    std::stringstream ss(operands_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
      trim(item);
      if (!item.empty())
        operands_vec.push_back(item);
    }
  }

  pl.operand_names = operands_vec; // Processed

  // parse op_name_and_attr: op_name[attr]
  // find first '[' and ']'
  auto s_attr = op_name_and_attr.find('{');
  auto e_attr = op_name_and_attr.rfind('}');

  std::string op_name;
  std::string attr_str;
  if (s_attr != std::string::npos && e_attr != std::string::npos &&
      e_attr > s_attr) {
    op_name = op_name_and_attr.substr(0, s_attr);
    attr_str = op_name_and_attr.substr(s_attr + 1, e_attr - s_attr - 1);
  } else {
    op_name = op_name_and_attr;
  }
  trim(op_name);
  trim(attr_str);

  pl.op_name = op_name;  // Processed
  pl.op_attr = attr_str; // Processed

  return pl;
}

Instruction *
create_instruction(const ParsedLine &pl,
                   std::unordered_map<std::string, Tensor *> &tensor_map,
                   std::unordered_map<Tensor *, int64> &known_offsets,
                   std::unordered_map<Tensor *, std::string> &known_constants) {

  // Helper lambdas
  // (i) panic with error message
  auto panic = [&](const std::string &msg) -> Instruction * {
    std::cerr << "create_instruction error: " << msg << std::endl;
    assert(false && "unreachable");
    return nullptr;
  };

  const std::string &lhs = pl.lhs_variable_name;
  const std::string &storage_name = pl.lhs_storage_name;
  int64 offset = pl.lhs_offset; // -1 if not specified
  const std::vector<int64> &shape_vec = pl.lhs_shape;
  const std::string &op_name = pl.op_name;
  const std::string &attr_str = pl.op_attr;
  const std::vector<std::string> &operands_names = pl.operand_names;

  // Find storage pointer
  Storage *storage = nullptr;
  if (g_storage.find(storage_name) != g_storage.end())
    storage = g_storage[storage_name];
  else
    return panic("unknown storage: " + storage_name);

  // Create Tensor object for lhs variable
  Tensor *lhs_tensor = new Tensor(lhs, storage, shape_vec);

  // insert into tensor_map
  tensor_map[lhs] = lhs_tensor;

  // if offset is known, insert into known_offsets
  if (offset != -1) {
    assert(storage->get_name() == "HBM" &&
           "only HBM storage supports fixed offset specification");
    assert(offset >= 0 && "storage offset must be non-negative");
    known_offsets[lhs_tensor] = offset;
  }

  // Handle Var and DCC as declarations with no instruction
  if (op_name.rfind("Var", 0) == 0) {
    lhs_tensor->type = Tensor::node_type::INPUT;
    // variable declaration -- no instruction created
    return nullptr;
  }
  if (op_name.rfind("DCC", 0) == 0) {
    lhs_tensor->type = Tensor::node_type::CONSTANT;
    // store constant value
    known_constants[lhs_tensor] = attr_str;
    // constant declaration -- no instruction created
    return nullptr;
  }
  lhs_tensor->type = Tensor::node_type::INTERMEDIATE;

  // For instructions, lookup operand tensors
  std::vector<Tensor *> rhs_tensors;
  for (auto &operand : operands_names) {
    if (tensor_map.find(operand) == tensor_map.end()) {
      return panic("unknown operand: " + operand);
    }
    rhs_tensors.push_back(tensor_map[operand]);
  }

  // Create instruction object based on op_name
  if (op_name == "slice") {
    // currently only implemented for one attribute: slices
    std::vector<std::pair<int64, int64>> slices;
    if (!attr_str.empty()) {
      std::stringstream ss(attr_str);
      std::string tok;
      while (std::getline(ss, tok, ';')) {
        if (!tok.empty()) {
          auto dash_pos = tok.find(':');
          if (dash_pos == std::string::npos)
            return panic("missing ':' in slice specification");
          std::string start_str = tok.substr(0, dash_pos);
          std::string end_str = tok.substr(dash_pos + 1);
          int64 s = 0, e = 0;
          try {
            s = std::stoll(start_str);
            e = std::stoll(end_str);
          } catch (...) {
            return panic("failed to parse slice indices as integers");
          }
          slices.push_back({s, e});
        }
      }
    } else {
      return panic("missing slices attribute");
    }

    return new Slice(lhs_tensor, rhs_tensors, slices);
  }
  else if (op_name == "concat") {
    // currently only implemented for one attribute: dimension
    int64 dimension = 0;
    if (!attr_str.empty()) {
      try {
        dimension = std::stoll(attr_str);
      } catch (...) {
        return panic("failed to parse dimension attribute as integer");
      }
    } else {
      return panic("missing dimension attribute");
    }

    return new Concat(lhs_tensor, rhs_tensors, dimension);
  }
{{INSTRUCTION_CASES}}

  return panic("unknown instruction: " + op_name);
}

} // namespace parser
