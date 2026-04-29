#include <fstream>
#include <iostream>
#include <string>

#include "act_malloc.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " input.pii metadata.txt" << std::endl;
    return 1;
  }

  std::string pii_path = argv[1];
  std::string metadata_path = argv[2];
  std::string asm_path;
  size_t extpos = std::string::npos;
  if ((extpos = pii_path.rfind(".pii")) != std::string::npos &&
      extpos + 4 == pii_path.size()) {
    asm_path = pii_path.substr(0, extpos) + ".py";
  } else {
    std::cerr << "Input file must have .pii extension" << std::endl;
    return 1;
  }

  return act_malloc(argv[1], argv[2], (char *)asm_path.c_str());
}
