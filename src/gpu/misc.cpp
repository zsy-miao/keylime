// keylime GPU: small device children: query heaps and command signatures.
#include <array>
#include <vector>

#include "gpu/device.h"
#include "gpu/fence.h"
#include "log.h"

namespace kl::gpu::misc {
namespace {

const Iid QUERY_HEAP_IID = guid(0x0d9658ae, 0xed45, 0x469e, {0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4});
const Iid SIGNATURE_IID = guid(0xc36a797c, 0xec80, 0x4f0a, {0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1});

bool pageable(const Iid* iid) {
    for (auto& g : fence::PAGEABLE_IIDS)
        if (same(iid, g)) return true;
    return false;
}

struct QueryDesc {
    uint32_t kind, count, nodes;
};
struct QueryHeap {
    Object object;
    QueryDesc desc;
};
HRESULT WINAPI qh_query(QueryHeap* h, const Iid* iid, void** out) {
    return answer(Kind::QuerySet, iid, (same(iid, QUERY_HEAP_IID) || pageable(iid)) ? h : nullptr, out);
}
void** qh_table() {
    static void** t = [] {
        void** t = make_table(Kind::QuerySet, 24);
        fill_object(t, (void*)&qh_query);
        fill_child(t);
        return t;
    }();
    return t;
}
HRESULT WINAPI create_query_heap(void*, const QueryDesc* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    return qh_query(new QueryHeap{{qh_table(), 0, {0}}, *d}, iid, out);
}

// Command signatures: the record layout of indirect execution (type + up to 3 fields each).
struct SignatureDesc {
    uint32_t stride, count;
    const uint32_t (*arguments)[4];
    uint32_t nodes;
};
struct Signature {
    Object object;
    uint32_t stride;
    std::vector<std::array<uint32_t, 4>> arguments;
};
HRESULT WINAPI sig_query(Signature* s, const Iid* iid, void** out) {
    return answer(Kind::CommandSignature, iid, (same(iid, SIGNATURE_IID) || pageable(iid)) ? s : nullptr, out);
}
void** sig_table() {
    static void** t = [] {
        void** t = make_table(Kind::CommandSignature, 24);
        fill_object(t, (void*)&sig_query);
        fill_child(t);
        return t;
    }();
    return t;
}
HRESULT WINAPI create_signature(void*, const SignatureDesc* d, void*, const Iid* iid, void** out) {
    if (!d || (!d->arguments && d->count)) return E_INVALIDARG;
    auto* s = new Signature{{sig_table(), 0, {0}}, d->stride, {}};
    for (uint32_t i = 0; i < d->count; ++i) s->arguments.push_back({d->arguments[i][0], d->arguments[i][1], d->arguments[i][2], d->arguments[i][3]});
    return sig_query(s, iid, out);
}

struct Init {
    Init() {
        device_set(39, (void*)&create_query_heap);  // 0x138
        device_set(41, (void*)&create_signature);   // 0x148
    }
} g_init;

}  // namespace
}  // namespace kl::gpu::misc
