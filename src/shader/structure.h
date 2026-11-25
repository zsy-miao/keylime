#pragma once
#include <cstdint>
#include <vector>

namespace kl::shader::cfg {

enum class Exit : uint8_t { Next, End, Jump, Branch };
struct Term {
    Exit kind = Exit::Next;
    uint32_t t = 0, f = 0;
};
struct Region {
    uint32_t lo, hi;
};
struct Node {
    enum Kind : uint8_t { Block, If, IfElse, Loop, Range } kind;
    uint32_t a = 0, b = 0;
    std::vector<Node> body, other;
};

std::vector<Node> plan(const std::vector<Term>& terms, const std::vector<Region>& regions);
const char* build_stamp();

}  // namespace kl::shader::cfg
