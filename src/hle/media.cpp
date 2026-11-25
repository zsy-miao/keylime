#include <windows.h>

#include <atomic>
#include <mutex>
#include <string>

#include "hle/files.h"
#include "hle/hle.h"
#include "log.h"

namespace kl::media {
namespace {

std::wstring host_url(const wchar_t* url) {
    if (!url) return {};
    const wchar_t* p = url;
    bool file = !_wcsnicmp(p, L"file:///", 8);
    if (file) p += 8;
    std::wstring out;
    if (files::translate(p, out)) {
        static std::atomic<int> n{0};
        if (n++ < 8) KL_INFO("media", "media file %ls -> %ls", url, out.c_str());
        return out;
    }
    return url;
}

using CreateFromUrl = HRESULT(WINAPI*)(void*, const wchar_t*, DWORD, void*, DWORD*, void**);
using BeginFromUrl = HRESULT(WINAPI*)(void*, const wchar_t*, DWORD, void*, void**, void*, void*);
CreateFromUrl g_create = nullptr;
BeginFromUrl g_begin = nullptr;
void* g_table[10];

HRESULT WINAPI create_from_url(void* self, const wchar_t* url, DWORD flags, void* props, DWORD* type, void** out) {
    std::wstring u = host_url(url);
    HRESULT hr = g_create(self, url ? u.c_str() : nullptr, flags, props, type, out);
    if (FAILED(hr)) KL_WARN("media", "opening media %ls failed: %08lx", url ? url : L"(none)", (unsigned long)hr);
    return hr;
}
HRESULT WINAPI begin_from_url(void* self, const wchar_t* url, DWORD flags, void* props, void** cancel, void* cb, void* state) {
    std::wstring u = host_url(url);
    return g_begin(self, url ? u.c_str() : nullptr, flags, props, cancel, cb, state);
}

HRESULT WINAPI MFCreateSourceResolver(void** out) {
    static auto host = reinterpret_cast<HRESULT(WINAPI*)(void**)>(hle::host_function("mfplat.dll", "MFCreateSourceResolver"));
    if (!host) return E_NOTIMPL;
    HRESULT hr = host(out);
    if (FAILED(hr) || !out || !*out) return hr;
    static std::once_flag once;
    void** vt = *static_cast<void***>(*out);
    std::call_once(once, [vt] {
        for (int i = 0; i < 10; ++i) g_table[i] = vt[i];
        g_create = reinterpret_cast<CreateFromUrl>(vt[3]);
        g_begin = reinterpret_cast<BeginFromUrl>(vt[5]);
        g_table[3] = reinterpret_cast<void*>(&create_from_url);
        g_table[5] = reinterpret_cast<void*>(&begin_from_url);
    });
    if (vt[3] == reinterpret_cast<void*>(g_create)) *static_cast<void***>(*out) = g_table;
    return hr;
}

HRESULT WINAPI MFCreateSourceReaderFromURL(const wchar_t* url, void* attrs, void** out) {
    static auto host = reinterpret_cast<HRESULT(WINAPI*)(const wchar_t*, void*, void**)>(
        hle::host_function("mfreadwrite.dll", "MFCreateSourceReaderFromURL"));
    if (!host) return E_NOTIMPL;
    std::wstring u = host_url(url);
    HRESULT hr = host(url ? u.c_str() : nullptr, attrs, out);
    if (FAILED(hr)) KL_WARN("media", "a source reader for %ls failed: %08lx", url ? url : L"(none)", (unsigned long)hr);
    return hr;
}

HRESULT WINAPI MFCreateFile(int access, int open, int flags, const wchar_t* url, void** out) {
    static auto host = reinterpret_cast<HRESULT(WINAPI*)(int, int, int, const wchar_t*, void**)>(
        hle::host_function("mfplat.dll", "MFCreateFile"));
    if (!host) return E_NOTIMPL;
    std::wstring u = host_url(url);
    return host(access, open, flags, url ? u.c_str() : nullptr, out);
}

}  // namespace

KL_MODULE("media", (std::vector<std::string>{"mfplat.dll", "mfreadwrite.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(MFCreateSourceResolver), KL_EXPORT(MFCreateSourceReaderFromURL),
                                    KL_EXPORT(MFCreateFile)}));

}  // namespace kl::media
