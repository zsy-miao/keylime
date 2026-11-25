#include "shader/spirv.h"

#include <cstring>

namespace kl::shader::sp {

Builder::Builder() {
    memmodel_ = {};
}

void Builder::emit(std::vector<uint32_t>& s, spv::Op o, const std::vector<uint32_t>& w) {
    s.push_back(uint32_t(w.size() + 1) << 16 | uint32_t(o));
    s.insert(s.end(), w.begin(), w.end());
}

std::vector<uint32_t> Builder::str(const std::string& s) {
    std::vector<uint32_t> w((s.size() + 4) / 4, 0);
    memcpy(w.data(), s.data(), s.size());
    return w;
}

void Builder::capability(spv::Capability c) {
    if (caps_seen_[uint32_t(c)]) return;
    caps_seen_[uint32_t(c)] = true;
    emit(caps_, spv::OpCapability, {uint32_t(c)});
}
void Builder::extension(const char* n) {
    if (exts_seen_[n]) return;
    exts_seen_[n] = true;
    emit(exts_, spv::OpExtension, str(n));
}
Id Builder::import_glsl() {
    if (!glsl_) {
        glsl_ = id();
        auto w = str("GLSL.std.450");
        w.insert(w.begin(), glsl_);
        emit(imports_, spv::OpExtInstImport, w);
    }
    return glsl_;
}
void Builder::entry(spv::ExecutionModel m, Id fn, const char* n, const std::vector<Id>& iface) {
    std::vector<uint32_t> w = {uint32_t(m), fn};
    auto s = str(n);
    w.insert(w.end(), s.begin(), s.end());
    w.insert(w.end(), iface.begin(), iface.end());
    emit(entries_, spv::OpEntryPoint, w);
}
void Builder::mode(Id fn, spv::ExecutionMode m, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> w = {fn, uint32_t(m)};
    w.insert(w.end(), args);
    emit(modes_, spv::OpExecutionMode, w);
}
void Builder::name(Id t, const std::string& n) {
    auto w = str(n);
    w.insert(w.begin(), t);
    emit(debug_, spv::OpName, w);
}
void Builder::decorate(Id t, spv::Decoration d, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> w = {t, uint32_t(d)};
    w.insert(w.end(), args);
    emit(annotations_, spv::OpDecorate, w);
}
void Builder::member_decorate(Id t, uint32_t m, spv::Decoration d, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> w = {t, m, uint32_t(d)};
    w.insert(w.end(), args);
    emit(annotations_, spv::OpMemberDecorate, w);
}

// ---------------------------------------------------------------- types ----

#define KL_TYPE(key_words, op, ...)                     \
    do {                                                \
        std::vector<uint32_t> key = key_words;          \
        auto it = type_cache_.find(key);                \
        if (it != type_cache_.end()) return it->second; \
        Id r = id();                                    \
        emit(globals_, op, {r, __VA_ARGS__});           \
        type_cache_[key] = r;                           \
        return r;                                       \
    } while (0)

Id Builder::t_void() {
    std::vector<uint32_t> key = {uint32_t(spv::OpTypeVoid)};
    if (auto it = type_cache_.find(key); it != type_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpTypeVoid, {r});
    return type_cache_[key] = r;
}
Id Builder::t_bool() {
    std::vector<uint32_t> key = {uint32_t(spv::OpTypeBool)};
    if (auto it = type_cache_.find(key); it != type_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpTypeBool, {r});
    return type_cache_[key] = r;
}
Id Builder::t_int(uint32_t w, bool s) {
    if (w == 64) capability(spv::CapabilityInt64);
    if (w == 16) capability(spv::CapabilityInt16);
    if (w == 8) capability(spv::CapabilityInt8);
    KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeInt), w, uint32_t(s)}), spv::OpTypeInt, w, uint32_t(s));
}
Id Builder::t_float(uint32_t w) {
    if (w == 16) capability(spv::CapabilityFloat16);
    if (w == 64) capability(spv::CapabilityFloat64);
    KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeFloat), w}), spv::OpTypeFloat, w);
}
Id Builder::t_vec(Id c, uint32_t n) { KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeVector), c, n}), spv::OpTypeVector, c, n); }
Id Builder::t_array(Id e, uint32_t n) {
    Id len = c_u32(n);
    KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeArray), e, len}), spv::OpTypeArray, e, len);
}
Id Builder::t_array_fresh(Id e, uint32_t n) {
    Id len = c_u32(n);
    Id r = id();
    emit(globals_, spv::OpTypeArray, {r, e, len});
    return r;
}
Id Builder::t_runtime_array(Id e) { KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeRuntimeArray), e}), spv::OpTypeRuntimeArray, e); }
Id Builder::t_struct(const std::vector<Id>& members) {
    Id r = id();
    std::vector<uint32_t> w = {r};
    w.insert(w.end(), members.begin(), members.end());
    emit(globals_, spv::OpTypeStruct, w);
    return r;
}
Id Builder::t_ptr(spv::StorageClass sc, Id t) {
    KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypePointer), uint32_t(sc), t}), spv::OpTypePointer, uint32_t(sc), t);
}
Id Builder::t_fn(Id ret, const std::vector<Id>& params) {
    std::vector<uint32_t> key = {uint32_t(spv::OpTypeFunction), ret};
    key.insert(key.end(), params.begin(), params.end());
    if (auto it = type_cache_.find(key); it != type_cache_.end()) return it->second;
    Id r = id();
    std::vector<uint32_t> w = {r, ret};
    w.insert(w.end(), params.begin(), params.end());
    emit(globals_, spv::OpTypeFunction, w);
    return type_cache_[key] = r;
}
Id Builder::t_image(Id sampled, spv::Dim dim, bool depth, bool arrayed, bool ms, uint32_t kind, spv::ImageFormat fmt) {
    KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeImage), sampled, uint32_t(dim), uint32_t(depth), uint32_t(arrayed),
                                   uint32_t(ms), kind, uint32_t(fmt)}),
            spv::OpTypeImage, sampled, uint32_t(dim), uint32_t(depth), uint32_t(arrayed), uint32_t(ms), kind, uint32_t(fmt));
}
Id Builder::t_sampler() {
    std::vector<uint32_t> key = {uint32_t(spv::OpTypeSampler)};
    if (auto it = type_cache_.find(key); it != type_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpTypeSampler, {r});
    return type_cache_[key] = r;
}
Id Builder::t_sampled_image(Id img) { KL_TYPE((std::vector<uint32_t>{uint32_t(spv::OpTypeSampledImage), img}), spv::OpTypeSampledImage, img); }

// ------------------------------------------------------------ constants ----

Id Builder::c_u32(uint32_t v) {
    Id t = t_u32();
    auto k = std::make_tuple(t, uint64_t(v));
    if (auto it = const_cache_.find(k); it != const_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpConstant, {t, r, v});
    return const_cache_[k] = r;
}
Id Builder::c_i32(int32_t v) {
    Id t = t_i32();
    auto k = std::make_tuple(t, uint64_t(uint32_t(v)));
    if (auto it = const_cache_.find(k); it != const_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpConstant, {t, r, uint32_t(v)});
    return const_cache_[k] = r;
}
Id Builder::c_u64(uint64_t v) {
    Id t = t_u64();
    auto k = std::make_tuple(t, v);
    if (auto it = const_cache_.find(k); it != const_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpConstant, {t, r, uint32_t(v), uint32_t(v >> 32)});
    return const_cache_[k] = r;
}
Id Builder::c_f32(float f) {
    Id t = t_f32();
    uint32_t v;
    memcpy(&v, &f, 4);
    auto k = std::make_tuple(t, uint64_t(v));
    if (auto it = const_cache_.find(k); it != const_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, spv::OpConstant, {t, r, v});
    return const_cache_[k] = r;
}
Id Builder::c_bool(bool b) {
    Id t = t_bool();
    auto k = std::make_tuple(t, uint64_t(b));
    if (auto it = const_cache_.find(k); it != const_cache_.end()) return it->second;
    Id r = id();
    emit(globals_, b ? spv::OpConstantTrue : spv::OpConstantFalse, {t, r});
    return const_cache_[k] = r;
}
Id Builder::c_composite(Id type, const std::vector<Id>& parts) {
    Id r = id();
    std::vector<uint32_t> w = {type, r};
    w.insert(w.end(), parts.begin(), parts.end());
    emit(globals_, spv::OpConstantComposite, w);
    return r;
}
Id Builder::c_null(Id type) {
    Id r = id();
    emit(globals_, spv::OpConstantNull, {type, r});
    return r;
}

Id Builder::variable(Id ptr_type, spv::StorageClass sc, Id init) {
    Id r = id();
    if (init) emit(globals_, spv::OpVariable, {ptr_type, r, uint32_t(sc), init});
    else emit(globals_, spv::OpVariable, {ptr_type, r, uint32_t(sc)});
    return r;
}

// ------------------------------------------------------------ functions ----

void Builder::begin_function(Id fn, Id ret, Id type) {
    fn_body_.clear();
    fn_vars_.clear();
    emit(fn_body_, spv::OpFunction, {ret, fn, uint32_t(spv::FunctionControlMaskNone), type});
    fn_header_ = 0;
}
void Builder::end_function() {
    emit(fn_body_, spv::OpFunctionEnd, {});
    // Locals go right after the first OpLabel.
    size_t first_label = 5;  // OpFunction is 5 words
    size_t after = first_label + 2;  // OpLabel is 2 words
    fn_body_.insert(fn_body_.begin() + ptrdiff_t(after), fn_vars_.begin(), fn_vars_.end());
    functions_.insert(functions_.end(), fn_body_.begin(), fn_body_.end());
    fn_body_.clear();
    fn_vars_.clear();
}
Id Builder::label() {
    ++labels;
    Id l = id();
    emit(fn_body_, spv::OpLabel, {l});
    return l;
}
void Builder::label(Id l) {
    ++labels;
    emit(fn_body_, spv::OpLabel, {l});
}
Id Builder::local(Id ptr_type, Id init) {
    Id r = id();
    if (init) emit(fn_vars_, spv::OpVariable, {ptr_type, r, uint32_t(spv::StorageClassFunction), init});
    else emit(fn_vars_, spv::OpVariable, {ptr_type, r, uint32_t(spv::StorageClassFunction)});
    return r;
}
Id Builder::op(spv::Op o, Id type, std::initializer_list<uint32_t> operands) {
    return op(o, type, std::vector<uint32_t>(operands));
}
Id Builder::op(spv::Op o, Id type, const std::vector<uint32_t>& operands) {
    Id r = id();
    std::vector<uint32_t> w = {type, r};
    w.insert(w.end(), operands.begin(), operands.end());
    emit(fn_body_, o, w);
    return r;
}
void Builder::op_void(spv::Op o, std::initializer_list<uint32_t> operands) { op_void(o, std::vector<uint32_t>(operands)); }
void Builder::op_void(spv::Op o, const std::vector<uint32_t>& operands) {
    switch (o) {
        case spv::OpBranch: case spv::OpBranchConditional: case spv::OpSwitch: case spv::OpReturn:
        case spv::OpReturnValue: case spv::OpKill: case spv::OpUnreachable: case spv::OpTerminateInvocation:
        case spv::OpSelectionMerge: case spv::OpLoopMerge:
            if (before_end) before_end();
            break;
        default: break;
    }
    emit(fn_body_, o, operands);
}

std::vector<uint32_t> Builder::finish() {
    std::vector<uint32_t> m = {0x07230203u, 0x00010500u /* SPIR-V 1.5 */, 0x4b4c0001u /* "KL" generator */, next_, 0};
    auto add = [&](const std::vector<uint32_t>& s) { m.insert(m.end(), s.begin(), s.end()); };
    add(caps_), add(exts_), add(imports_);
    emit(m, spv::OpMemoryModel, {uint32_t(spv::AddressingModelPhysicalStorageBuffer64), uint32_t(spv::MemoryModelGLSL450)});
    add(entries_), add(modes_), add(debug_), add(annotations_), add(globals_), add(functions_);
    return m;
}

}  // namespace kl::shader::sp
