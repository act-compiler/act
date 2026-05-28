#include "emitter.h"

namespace emitter {

// Recursive DCC expression → Python/jnp translator.
//
// DCC strings are produced by the Rust egraph Display impl and op_repr.
// Format: op(child1, child2)       -- no attribute
//         op{attr}(child1)         -- with attribute
//         op{attr}()               -- leaf (e.g. constant{-16}())
//
// This parser walks the string left-to-right with a shared position cursor.
// Each call to dcc_to_python() consumes one sub-expression and returns the
// equivalent jnp Python code.
static std::string dcc_to_python(const std::string &expr, size_t &pos);

// Consume a {}-delimited attribute string. pos must point at '{'.
// Returns the content between the braces (e.g. "16,16" from "{16,16}").
static std::string parse_attr(const std::string &expr, size_t &pos) {
  if (pos >= expr.size() || expr[pos] != '{') return "";
  pos++;
  int depth = 1;
  size_t start = pos;
  while (pos < expr.size() && depth > 0) {
    if (expr[pos] == '{') depth++;
    else if (expr[pos] == '}') depth--;
    pos++;
  }
  return expr.substr(start, pos - start - 1);
}

// Consume a ()-delimited argument list. pos must point at '('.
// Recursively calls dcc_to_python() for each comma-separated child.
static std::vector<std::string> parse_args(const std::string &expr, size_t &pos) {
  std::vector<std::string> args;
  if (pos >= expr.size() || expr[pos] != '(') return args;
  pos++;
  while (pos < expr.size() && expr[pos] != ')') {
    if (expr[pos] == ' ' || expr[pos] == ',') { pos++; continue; }
    args.push_back(dcc_to_python(expr, pos));
    if (pos < expr.size() && expr[pos] == ',') pos++;
  }
  if (pos < expr.size()) pos++;
  return args;
}

// Parse one DCC sub-expression starting at pos and return jnp Python code.
static std::string dcc_to_python(const std::string &expr, size_t &pos) {
  // 1. Read op name (letters, digits, underscore) up to '{' or '('
  size_t start = pos;
  while (pos < expr.size() && expr[pos] != '(' && expr[pos] != '{' &&
         expr[pos] != ')' && expr[pos] != ',')
    pos++;
  std::string op = expr.substr(start, pos - start);

  // 2. Optional attribute in braces: {attr}
  std::string attr;
  if (pos < expr.size() && expr[pos] == '{')
    attr = parse_attr(expr, pos);

  // 3. Arguments in parens: (child1, child2, ...)
  auto args = parse_args(expr, pos);

  // 4. Map DCC op to jnp Python expression.

  // --- Leaves (no children) ---
  if (op == "constant") {
    // attr may be "value" or "value:dtype" (e.g. "1:I8", "-16:I32")
    auto colon = attr.find(':');
    std::string val = (colon != std::string::npos) ? attr.substr(0, colon) : attr;
    std::string dtype = (colon != std::string::npos) ? attr.substr(colon + 1) : "I8";
    std::string jnp_type = "jnp.int8";
    if (dtype == "I32" || dtype == "s32") jnp_type = "jnp.int32";
    else if (dtype == "U8" || dtype == "u8") jnp_type = "jnp.uint8";
    else if (dtype == "BF16" || dtype == "bf16") jnp_type = "jnp.bfloat16";
    return jnp_type + "(" + val + ")";
  } else if (op == "eye") {
    auto comma1 = attr.find(',');                                 // eye{16,16,I8}()
    return "jnp.eye(" + attr.substr(0, comma1) + ", dtype=jnp.int8)";

  // --- Layout / type ops ---
  } else if (op == "broadcast") {
    if (args.empty()) return "jnp.zeros((" + attr + "), dtype=jnp.int8)";
    return "jnp.broadcast_to(" + args[0] + ", (" + attr + "))";  // broadcast{16,16}(x)
  } else if (op == "reshape") {
    if (args.empty()) return "jnp.zeros((" + attr + ",), dtype=jnp.int8)";
    return "jnp.reshape(" + args[0] + ", (" + attr + ",))";      // reshape{256}(x)
  } else if (op == "bitcvt") {
    if (args.empty()) return "jnp.int8(0)";
    return args[0] + ".view(jnp.uint8)";                         // bitcvt(x)
  } else if (op == "copy") {
    return args.empty() ? "jnp.int8(0)" : args[0];               // copy(x)
  } else if (op == "convert") {
    if (args.empty()) return "jnp.int8(0)";                      // convert{s32}(x)
    std::string target = "jnp.int8";
    if (attr == "I32" || attr == "s32") target = "jnp.int32";
    else if (attr == "I16" || attr == "s16") target = "jnp.int16";
    else if (attr == "U8" || attr == "u8") target = "jnp.uint8";
    else if (attr == "U16" || attr == "u16") target = "jnp.uint16";
    else if (attr == "U32" || attr == "u32") target = "jnp.uint32";
    else if (attr == "BF16" || attr == "bf16") target = "jnp.bfloat16";
    else if (attr == "FP16" || attr == "f16") target = "jnp.float16";
    else if (attr == "FP32" || attr == "f32") target = "jnp.float32";
    return args[0] + ".astype(" + target + ")";
  } else if (op == "reverse") {
    return "jnp.flip(" + args[0] + ", axis=(" + attr + ",))";    // reverse{0,1}(x)
  } else if (op == "transpose") {
    return "jnp.transpose(" + args[0] + ", (" + attr + ",))";    // transpose{1,0}(x)

  // --- Arithmetic (binary) ---
  } else if (op == "add") {
    return "jnp.add(" + args[0] + ", " + args[1] + ")";
  } else if (op == "subtract") {
    return "jnp.subtract(" + args[0] + ", " + args[1] + ")";
  } else if (op == "multiply") {
    return "jnp.multiply(" + args[0] + ", " + args[1] + ")";
  } else if (op == "divide") {
    return "jnp.divide(" + args[0] + ", " + args[1] + ")";
  } else if (op == "dot") {
    return "jnp.dot(" + args[0] + ", " + args[1] + ")";

  // --- Arithmetic (unary) ---
  } else if (op == "negate") {
    return "jnp.negative(" + args[0] + ")";
  } else if (op == "exponential") {
    return "jnp.exp(" + args[0] + ")";

  // --- Min/max/clamp ---
  } else if (op == "maximum") {
    return "jnp.maximum(" + args[0] + ", " + args[1] + ")";
  } else if (op == "minimum") {
    return "jnp.minimum(" + args[0] + ", " + args[1] + ")";
  } else if (op == "clamp") {
    return "jnp.clip(" + args[1] + ", " + args[0] + ", " + args[2] + ")";

  // --- Reduction ---
  } else if (op == "reduce") {
    return "jnp.sum(" + args[0] + ", axis=" + attr + ")";        // reduce{1}(x)
  }

  return "None  # unsupported: " + op;
}

MetaDataInfo::MetaDataInfo(nlohmann::json &j) {
  addr = 0;
  if (j.contains("addr") && j["addr"].is_number_integer())
    addr = j["addr"].get<int64>();
  else
    throw std::runtime_error("addr is required in metadata.json");

  shape.clear();
  if (j.contains("shape") && j["shape"].is_array()) {
    for (const auto &s : j["shape"]) {
      if (s.is_number_integer())
        shape.push_back(s.get<int64>());
      else
        throw std::runtime_error("shape must be an array of integers");
    }
  } else
    throw std::runtime_error("shape is required in metadata.json");

  if (j.contains("dtype") && j["dtype"].is_string())
    dtype = j["dtype"].get<std::string>();
  else
    throw std::runtime_error("dtype is required in metadata.json");
}

std::string MetaDataInfo::str() const {
  std::ostringstream oss;
  oss << "{'addr': " << addr << ", 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i != shape.size() - 1)
      oss << ", ";
  }
  oss << "), 'dtype': " << dtype;
  if (value != "")
    oss << ", 'value': " << value;
  oss << "}";
  return oss.str();
}

MetaData::MetaData(
    std::ifstream &metadata_path,
    const std::unordered_map<std::string, Tensor *> &tensor_map,
    const std::unordered_map<Tensor *, std::string> &known_constants)
    : max_hbm_size(HBM_SIZE) {
  try {
    if (!metadata_path)
      throw std::runtime_error("failed to open metadata.json");

    nlohmann::json j;
    metadata_path.clear();
    metadata_path.seekg(0);
    metadata_path >> j;

    if (j.contains("module_name") && j["module_name"].is_string()) {
      this->module_name = j["module_name"].get<std::string>();
    } else
      throw std::runtime_error("module_name is required in metadata.json");

    if (j.contains("input") && j["input"].is_array()) {
      for (const auto &entry : j["input"]) {
        input_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("input is required in metadata.json");

    if (j.contains("output") && j["output"].is_array()) {
      if (j["output"].size() != 1)
        throw std::runtime_error("supports only one output in metadata.json");

      for (const auto &entry : j["output"]) {
        output_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("output is required in metadata.json");
  } catch (const std::exception &e) {
    std::cerr << "Warning: failed to parse metadata.json: " << e.what()
              << std::endl;
    assert(false && "nlohmann json parse error");
  }

  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->type == Tensor::CONSTANT) {
      if (known_constants.find(tensor) == known_constants.end()) {
        std::cerr << "Warning: constant tensor " << tensor->get_name()
                  << " not found in known constants." << std::endl;
        assert(false && "unknown constant tensor");
      }
      const std::string &dcc_expr = known_constants.at(tensor);
      size_t pos = 0;
      std::string py_value = dcc_to_python(dcc_expr, pos);

      constant_info.emplace_back(MetaDataInfo(
          tensor->get_offsets()[0]->Min(), tensor->get_sizes(), "jnp.uint8",
          py_value));
    }
  }

  this->max_hbm_size = 0;
  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->get_storage()->get_name() == "HBM") {
      int64 offset = tensor->get_offsets()[0]->Min();
      int64 size = tensor->get_sizes()[0];
      if (offset + size > this->max_hbm_size)
        this->max_hbm_size = offset + size;
    }
  }
}

bool assembly_dump(char *outpath,
                   const std::vector<Instruction *> &instructions,
                   const MetaData &metadata) {
  std::string indent1 = "    ";
  std::string indent2 = indent1 + indent1;
  std::string indent3 = indent2 + indent1;
  std::string indent4 = indent3 + indent1;

  // Process hlo_name and pii_number from outpath.
  // Accept any path with a filename stem; use parent directory as hlo_name
  // when available, otherwise fall back to metadata.module_name.
  std::string path_str(outpath);
  size_t lslash = path_str.find_last_of('/');
  size_t fname_start = (lslash == std::string::npos) ? 0 : lslash + 1;
  if (fname_start >= path_str.size()) {
    std::cerr
        << "Warning: output path " << outpath
        << " has no filename component. Defaulting to no solution."
        << std::endl;
    return false;
  }

  size_t ldot = path_str.find_last_of('.');
  size_t stem_end = path_str.size();
  if (ldot != std::string::npos && ldot > fname_start) {
    stem_end = ldot;
  }

  if (stem_end <= fname_start) {
    std::cerr
        << "Warning: output path " << outpath
        << " does not contain a valid filename stem. "
           "Defaulting to no solution."
        << std::endl;
    return false;
  }

  std::string pii_number = path_str.substr(fname_start, stem_end - fname_start);
  std::string hlo_name = metadata.module_name;
  if (lslash != std::string::npos && lslash > 0) {
    size_t l2slash = path_str.find_last_of('/', lslash - 1);
    size_t parent_start = (l2slash == std::string::npos) ? 0 : l2slash + 1;
    if (lslash > parent_start) {
      hlo_name = path_str.substr(parent_start, lslash - parent_start);
    }
  }

  std::ofstream outfile(outpath);
  if (!outfile) {
    std::cerr << "Warning: failed to open output file: " << outpath
              << "; defaulting to no solution." << std::endl;
    return false;
  }

  // Process kernel function name
  std::string kernel_name = metadata.module_name;

  // HEADER
  outfile << "# Input file: " << hlo_name << ".hlo" << std::endl;
  outfile << "# Kernel name: " << kernel_name << std::endl;
  outfile << "# PII number: " << pii_number << std::endl;
  outfile << "# Do not edit!" << std::endl << std::endl;

  outfile << "import jax.numpy as jnp" << std::endl << std::endl << std::endl;

  // Kernel function metadata
  outfile << "def " << kernel_name << "(kernel, api):" << std::endl;

  outfile << indent1 << "@kernel(hbm=" << metadata.max_hbm_size << ","
          << std::endl;

  outfile << indent3 << "input=[" << std::endl;
  for (const auto &info : metadata.input_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]," << std::endl;

  if (!metadata.constant_info.empty()) {
    outfile << indent3 << "constant=[" << std::endl;
    for (const auto &info : metadata.constant_info) {
      outfile << indent4 << info.str() << "," << std::endl;
    }
    outfile << indent3 << "]," << std::endl;
  } else {
    outfile << indent3 << "constant=[]," << std::endl;
  }

  outfile << indent3 << "output=[" << std::endl;
  for (const auto &info : metadata.output_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]" << std::endl;

  outfile << indent3 << ")" << std::endl;
  outfile << indent1 << "def " << kernel_name << "_" << "():" << std::endl;

  // Assembly code
  for (auto *instr : instructions) {
    outfile << indent2 << "api." << instr->str() << std::endl;
  }

  outfile << std::endl;
  outfile << indent1 << "return " << kernel_name << "_" << std::endl;

  outfile.close();
  return true;
}

} // namespace emitter
