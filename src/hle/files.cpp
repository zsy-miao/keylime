// keylime: the console's fixed drives over host folders. G: is the package
// (the game folder), T: temporary storage. Every title import taking a path
// is bound here: a path on a virtual drive is rewritten to the host folder
// and handed to the host's wide function; any other path passes unchanged.
#include "hle/files.h"

#include <windows.h>

#include <cstdio>
#include <cstdarg>
#include <fcntl.h>
#include <io.h>
#include <mutex>
#include <sys/stat.h>
#include <vector>

#include "hle/hle.h"
#include "log.h"

namespace kl::files {
namespace {

std::mutex g_lock;
std::wstring g_roots[26];  // by letter; empty = not mounted

std::wstring widen(const char* s) {
    if (!s) return {};
    int n = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    std::wstring w(size_t(n > 0 ? n - 1 : 0), L'\0');
    if (n > 1) MultiByteToWideChar(CP_ACP, 0, s, -1, w.data(), n);
    return w;
}
std::string narrow(const std::wstring& w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n > 0 ? n - 1 : 0), '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

}  // namespace

void mount(char letter, const std::filesystem::path& root) {
    std::lock_guard g(g_lock);
    auto r = std::filesystem::absolute(root).wstring();
    while (r.size() > 3 && (r.back() == L'\\' || r.back() == L'/')) r.pop_back();
    g_roots[toupper(letter) - 'A'] = r;
    KL_INFO("files", "drive %c: -> %ls", toupper(letter), r.c_str());
}

bool translate(const wchar_t* path, std::wstring& out) {
    if (!path) return false;
    const wchar_t* rest = path;
    bool verbatim = false;
    if (!wcsncmp(rest, L"\\\\?\\", 4)) rest += 4, verbatim = true;
    if (!iswalpha(rest[0]) || rest[1] != L':') return false;
    int letter = towupper(rest[0]) - L'A';
    if (letter < 0 || letter >= 26) return false;
    std::wstring root;
    {
        std::lock_guard g(g_lock);
        root = g_roots[letter];
    }
    if (root.empty()) return false;
    const wchar_t* tail = rest + 2;
    // Resolve "." and ".." ourselves, '/' -> '\', so the \\?\ form (no host clean-up) is safe.
    std::vector<std::wstring> parts;
    std::wstring cur;
    for (const wchar_t* p = tail;; ++p) {
        if (*p == L'\\' || *p == L'/' || *p == 0) {
            if (cur == L"..") {
                if (!parts.empty()) parts.pop_back();
            } else if (!cur.empty() && cur != L".") {
                parts.push_back(cur);
            }
            cur.clear();
            if (!*p) break;
        } else {
            cur.push_back(*p);
        }
    }
    out = L"\\\\?\\" + root;
    for (auto& s : parts) out += L"\\" + s;
    if (parts.empty()) out += L"\\";
    (void)verbatim;
    KL_TRACE("files", "%ls -> %ls", path, out.c_str());
    return true;
}

namespace {

// The title's wide path or its host rewrite; `keep` owns the rewrite.
const wchar_t* host(const wchar_t* path, std::wstring& keep) { return translate(path, keep) ? keep.c_str() : path; }
// A narrow path on a virtual drive becomes a wide host path; else nullptr (use the narrow host API).
const wchar_t* host_a(const char* path, std::wstring& keep) {
    if (!path) return nullptr;
    std::wstring w = widen(path);
    return translate(w.c_str(), keep) ? keep.c_str() : nullptr;
}

HANDLE WINAPI kCreateFileW(LPCWSTR p, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD c, DWORD f, HANDLE t) {
    std::wstring k;
    HANDLE h = CreateFileW(host(p, k), a, s, sa, c, f, t);
    if (h == INVALID_HANDLE_VALUE) KL_TRACE("files", "CreateFileW(%ls) failed: %lu", p ? p : L"", GetLastError());
    return h;
}
HANDLE WINAPI kCreateFileA(LPCSTR p, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD c, DWORD f, HANDLE t) {
    std::wstring k;
    if (auto* w = host_a(p, k)) return CreateFileW(w, a, s, sa, c, f, t);
    return CreateFileA(p, a, s, sa, c, f, t);
}
HANDLE WINAPI kCreateFile2(LPCWSTR p, DWORD a, DWORD s, DWORD c, LPCREATEFILE2_EXTENDED_PARAMETERS x) {
    std::wstring k;
    return CreateFile2(host(p, k), a, s, c, x);
}
BOOL WINAPI kDeleteFileW(LPCWSTR p) { std::wstring k; return DeleteFileW(host(p, k)); }
BOOL WINAPI kDeleteFileA(LPCSTR p) { std::wstring k; if (auto* w = host_a(p, k)) return DeleteFileW(w); return DeleteFileA(p); }
BOOL WINAPI kRemoveDirectoryW(LPCWSTR p) { std::wstring k; return RemoveDirectoryW(host(p, k)); }
BOOL WINAPI kSetFileAttributesW(LPCWSTR p, DWORD a) { std::wstring k; return SetFileAttributesW(host(p, k), a); }
DWORD WINAPI kGetFileAttributesW(LPCWSTR p) { std::wstring k; return GetFileAttributesW(host(p, k)); }
DWORD WINAPI kGetFileAttributesA(LPCSTR p) { std::wstring k; if (auto* w = host_a(p, k)) return GetFileAttributesW(w); return GetFileAttributesA(p); }
BOOL WINAPI kGetFileAttributesExW(LPCWSTR p, GET_FILEEX_INFO_LEVELS l, LPVOID o) { std::wstring k; return GetFileAttributesExW(host(p, k), l, o); }
BOOL WINAPI kCreateDirectoryW(LPCWSTR p, LPSECURITY_ATTRIBUTES sa) { std::wstring k; return CreateDirectoryW(host(p, k), sa); }
BOOL WINAPI kCreateDirectoryA(LPCSTR p, LPSECURITY_ATTRIBUTES sa) { std::wstring k; if (auto* w = host_a(p, k)) return CreateDirectoryW(w, sa); return CreateDirectoryA(p, sa); }
BOOL WINAPI kGetDiskFreeSpaceExW(LPCWSTR p, PULARGE_INTEGER a, PULARGE_INTEGER b, PULARGE_INTEGER c) {
    std::wstring k;
    if (translate(p, k)) {
        // A volume root: the host volume the folder is on.
        wchar_t vol[MAX_PATH];
        if (GetVolumePathNameW(k.c_str() + 4, vol, MAX_PATH)) return GetDiskFreeSpaceExW(vol, a, b, c);
    }
    return GetDiskFreeSpaceExW(p, a, b, c);
}
BOOL WINAPI kGetDiskFreeSpaceW(LPCWSTR p, LPDWORD a, LPDWORD b, LPDWORD c, LPDWORD d) {
    std::wstring k;
    if (translate(p, k)) {
        wchar_t vol[MAX_PATH];
        if (GetVolumePathNameW(k.c_str() + 4, vol, MAX_PATH)) return GetDiskFreeSpaceW(vol, a, b, c, d);
    }
    return GetDiskFreeSpaceW(p, a, b, c, d);
}
HANDLE WINAPI kFindFirstFileW(LPCWSTR p, LPWIN32_FIND_DATAW d) { std::wstring k; return FindFirstFileW(host(p, k), d); }
HANDLE WINAPI kFindFirstFileExW(LPCWSTR p, FINDEX_INFO_LEVELS l, LPVOID d, FINDEX_SEARCH_OPS o, LPVOID f, DWORD x) {
    std::wstring k;
    return FindFirstFileExW(host(p, k), l, d, o, f, x);
}
BOOL WINAPI kMoveFileExW(LPCWSTR a, LPCWSTR b, DWORD f) { std::wstring k1, k2; return MoveFileExW(host(a, k1), host(b, k2), f); }
BOOL WINAPI kSetCurrentDirectoryA(LPCSTR p) { std::wstring k; if (auto* w = host_a(p, k)) return SetCurrentDirectoryW(w); return SetCurrentDirectoryA(p); }
BOOL WINAPI kSetCurrentDirectoryW(LPCWSTR p) { std::wstring k; return SetCurrentDirectoryW(host(p, k)); }

// CRT: the host ucrt opens files without our IAT, so its path entry points are ours too.
std::wstring wmode(const char* m) { return widen(m ? m : "r"); }
FILE* __cdecl k_wfopen(const wchar_t* p, const wchar_t* m) { std::wstring k; return _wfopen(host(p, k), m); }
errno_t __cdecl k_wfopen_s(FILE** f, const wchar_t* p, const wchar_t* m) { std::wstring k; return _wfopen_s(f, host(p, k), m); }
FILE* __cdecl k_wfsopen(const wchar_t* p, const wchar_t* m, int sh) { std::wstring k; return _wfsopen(host(p, k), m, sh); }
FILE* __cdecl kfopen(const char* p, const char* m) {
    std::wstring k;
    if (auto* w = host_a(p, k)) return _wfopen(w, wmode(m).c_str());
    return fopen(p, m);
}
errno_t __cdecl kfopen_s(FILE** f, const char* p, const char* m) {
    std::wstring k;
    if (auto* w = host_a(p, k)) return _wfopen_s(f, w, wmode(m).c_str());
    return fopen_s(f, p, m);
}
int __cdecl k_open(const char* p, int flags, int mode) {
    std::wstring k;
    if (auto* w = host_a(p, k)) return _wopen(w, flags, mode);
    return _open(p, flags, mode);
}
int __cdecl k_wopen(const wchar_t* p, int flags, int mode) { std::wstring k; return _wopen(host(p, k), flags, mode); }
int __cdecl k_wstat64(const wchar_t* p, struct _stat64* b) { std::wstring k; return _wstat64(host(p, k), b); }
int __cdecl k_stat64i32(const char* p, struct _stat64i32* b) {
    std::wstring k;
    if (auto* w = host_a(p, k)) return _wstat64i32(w, b);
    return _stat64i32(p, b);
}
int __cdecl kremove(const char* p) { std::wstring k; if (auto* w = host_a(p, k)) return _wremove(w); return remove(p); }
int __cdecl k_wremove(const wchar_t* p) { std::wstring k; return _wremove(host(p, k)); }
int __cdecl k_wchmod(const wchar_t* p, int m) { std::wstring k; return _wchmod(host(p, k), m); }

// msvcp140 _Fiopen (fstream opens): forward to the host's after rewriting.
using FiopenW = FILE*(__cdecl*)(const wchar_t*, int, int);
using FiopenA = FILE*(__cdecl*)(const char*, int, int);
FILE* __cdecl kFiopenW(const wchar_t* p, int mode, int prot) {
    static auto real = reinterpret_cast<FiopenW>(hle::host_function("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z"));
    std::wstring k;
    return real ? real(host(p, k), mode, prot) : nullptr;
}
FILE* __cdecl kFiopenA(const char* p, int mode, int prot) {
    static auto realA = reinterpret_cast<FiopenA>(hle::host_function("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z"));
    std::wstring k;
    if (auto* w = host_a(p, k)) return kFiopenW(w, mode, prot);
    return realA ? realA(p, mode, prot) : nullptr;
}

}  // namespace

KL_MODULE("files",
          (std::vector<std::string>{"kernel32.dll", "kernelbase.dll", "api-ms-win-core-file-*",
                                    "api-ms-win-core-processenvironment-*", "ucrtbase.dll", "api-ms-win-crt-stdio-*",
                                    "api-ms-win-crt-filesystem-*", "msvcp140.dll"}),
          (std::vector<hle::Export>{
              KL_EXPORT_AS("CreateFileW", kCreateFileW), KL_EXPORT_AS("CreateFileA", kCreateFileA),
              KL_EXPORT_AS("CreateFile2", kCreateFile2), KL_EXPORT_AS("DeleteFileW", kDeleteFileW),
              KL_EXPORT_AS("DeleteFileA", kDeleteFileA), KL_EXPORT_AS("RemoveDirectoryW", kRemoveDirectoryW),
              KL_EXPORT_AS("SetFileAttributesW", kSetFileAttributesW), KL_EXPORT_AS("GetFileAttributesW", kGetFileAttributesW),
              KL_EXPORT_AS("GetFileAttributesA", kGetFileAttributesA), KL_EXPORT_AS("GetFileAttributesExW", kGetFileAttributesExW),
              KL_EXPORT_AS("CreateDirectoryW", kCreateDirectoryW), KL_EXPORT_AS("CreateDirectoryA", kCreateDirectoryA),
              KL_EXPORT_AS("GetDiskFreeSpaceExW", kGetDiskFreeSpaceExW), KL_EXPORT_AS("GetDiskFreeSpaceW", kGetDiskFreeSpaceW),
              KL_EXPORT_AS("FindFirstFileW", kFindFirstFileW), KL_EXPORT_AS("FindFirstFileExW", kFindFirstFileExW),
              KL_EXPORT_AS("MoveFileExW", kMoveFileExW), KL_EXPORT_AS("SetCurrentDirectoryA", kSetCurrentDirectoryA),
              KL_EXPORT_AS("SetCurrentDirectoryW", kSetCurrentDirectoryW),
              KL_EXPORT_AS("_wfopen", k_wfopen), KL_EXPORT_AS("_wfopen_s", k_wfopen_s), KL_EXPORT_AS("_wfsopen", k_wfsopen),
              KL_EXPORT_AS("fopen", kfopen), KL_EXPORT_AS("fopen_s", kfopen_s), KL_EXPORT_AS("_open", k_open),
              KL_EXPORT_AS("_wopen", k_wopen), KL_EXPORT_AS("_wstat64", k_wstat64), KL_EXPORT_AS("_stat64i32", k_stat64i32),
              KL_EXPORT_AS("remove", kremove), KL_EXPORT_AS("_wremove", k_wremove), KL_EXPORT_AS("_wchmod", k_wchmod),
              KL_EXPORT_AS("?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z", kFiopenW),
              KL_EXPORT_AS("?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z", kFiopenA),
          }));

}  // namespace kl::files
