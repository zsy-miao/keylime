// keylime shaders: a small SPIR-V module builder (sections, ids, cached
// types and constants). Opcode and enum values from Khronos' spirv.hpp.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "spirv.hpp"

namespace kl::shader::sp {

using Id = uint32_t;

class Builder {
public:
    Builder();
    Id id() { return next_++; }

    // Sections.
    void capability(spv::Capability c);
    void extension(const char* name);
    Id import_glsl();
    void entry(spv::ExecutionModel model, Id fn, const char* name, const std::vector<Id>& interface);
    void mode(Id fn, spv::ExecutionMode m, std::initializer_list<uint32_t> args = {});
    void name(Id target, const std::string& n);
    void decorate(Id target, spv::Decoration d, std::initializer_list<uint32_t> args = {});
    void member_decorate(Id type, uint32_t member, spv::Decoration d, std::initializer_list<uint32_t> args = {});

    // Types (cached).
    Id t_void();
    Id t_bool();
    Id t_int(uint32_t width, bool sign);
    Id t_u32() { return t_int(32, false); }
    Id t_i32() { return t_int(32, true); }
    Id t_u64() { return t_int(64, false); }
    Id t_float(uint32_t width = 32);
    Id t_f32() { return t_float(32); }
    Id t_vec(Id component, uint32_t n);
    Id t_array(Id element, uint32_t n);
    Id t_runtime_array(Id element);
    Id t_struct(const std::vector<Id>& members);  // not cached
    Id t_ptr(spv::StorageClass sc, Id type);
    Id t_fn(Id ret, const std::vector<Id>& params);
    Id t_image(Id sampled, spv::Dim dim, bool depth, bool arrayed, bool ms, uint32_t sampled_kind, spv::ImageFormat fmt);
    Id t_sampler();
    Id t_sampled_image(Id image);

    // Constants (cached).
    Id c_u32(uint32_t v);
    Id c_i32(int32_t v);
    Id c_u64(uint64_t v);
    Id c_f32(float v);
    Id c_bool(bool v);
    Id c_composite(Id type, const std::vector<Id>& parts);
    Id c_null(Id type);

    // Globals.
    Id variable(Id ptr_type, spv::StorageClass sc, Id init = 0);

    // Function body.
    void begin_function(Id fn, Id ret, Id type);
    void end_function();
    Id label();          // emits OpLabel with a new id
    void label(Id l);    // emits OpLabel with id l
    Id local(Id ptr_type, Id init = 0);  // Function-storage variable (hoisted to the entry block)
    // Generic instruction: result type and result id (0 for none), then operands.
    Id op(spv::Op o, Id type, std::initializer_list<uint32_t> operands);
    Id op(spv::Op o, Id type, const std::vector<uint32_t>& operands);
    void op_void(spv::Op o, std::initializer_list<uint32_t> operands);
    void op_void(spv::Op o, const std::vector<uint32_t>& operands);

    std::vector<uint32_t> finish();

private:
    static void emit(std::vector<uint32_t>& s, spv::Op o, const std::vector<uint32_t>& w);
    static std::vector<uint32_t> str(const std::string& s);
    Id next_ = 1;
    Id glsl_ = 0;
    std::vector<uint32_t> caps_, exts_, imports_, memmodel_, entries_, modes_, debug_, annotations_, globals_, functions_;
    std::vector<uint32_t> fn_vars_;  // locals of the current function (go after its first label)
    std::vector<uint32_t> fn_body_;
    size_t fn_header_ = 0;
    std::map<std::vector<uint32_t>, Id> type_cache_;
    std::map<std::tuple<Id, uint64_t>, Id> const_cache_;
    std::map<uint32_t, bool> caps_seen_;
    std::map<std::string, bool> exts_seen_;
};

}  // namespace kl::shader::sp
