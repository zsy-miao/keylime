#include "loader.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>

#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

#include "hle/hle.h"
#include "log.h"

namespace fs = std::filesystem;

namespace kl {

static Loader* g_loader = nullptr;
Loader& loader() { return *g_loader; }
void set_loader(Loader* l) { g_loader = l; }

// ------------------------------------------------------------ prepare ----

static uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 0xcbf29ce484222325ull;
    // 8 bytes at a time: the executable is large.
    size_t i = 0;
    for (; i + 8 <= b.size(); i += 8) {
        uint64_t v;
        std::memcpy(&v, b.data() + i, 8);
        h = (h ^ v) * 0x100000001b3ull;
    }
    for (; i < b.size(); ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

template <class T>
static void put(std::vector<uint8_t>& b, size_t at, T v) {
    if (at + sizeof v <= b.size()) std::memcpy(b.data() + at, &v, sizeof v);
}

// Marks the image as a DLL with no entry point, an empty import table and no TLS callbacks.
static std::vector<uint8_t> prepare(const std::vector<uint8_t>& original, const pe::Image& img) {
    std::vector<uint8_t> b = original;
    const size_t fh = img.file_header_off, opt = img.opt_header_off;
    put<uint16_t>(b, fh + offsetof(IMAGE_FILE_HEADER, Characteristics), uint16_t(img.characteristics | IMAGE_FILE_DLL));
    put<uint32_t>(b, opt + offsetof(IMAGE_OPTIONAL_HEADER64, AddressOfEntryPoint), 0);
    put<uint32_t>(b, opt + offsetof(IMAGE_OPTIONAL_HEADER64, CheckSum), 0);
    put<uint16_t>(b, opt + offsetof(IMAGE_OPTIONAL_HEADER64, DllCharacteristics),
                  uint16_t(img.dll_characteristics & ~IMAGE_DLLCHARACTERISTICS_FORCE_INTEGRITY));
    const size_t dirs = opt + offsetof(IMAGE_OPTIONAL_HEADER64, DataDirectory);
    auto dir_at = [&](int i) { return dirs + size_t(i) * sizeof(IMAGE_DATA_DIRECTORY); };
    // Imports: point the directory at the all-zero terminator descriptor.
    uint32_t imp_rva = 0;
    std::memcpy(&imp_rva, b.data() + dir_at(IMAGE_DIRECTORY_ENTRY_IMPORT), 4);
    if (imp_rva) {
        size_t normal = 0;
        for (auto& m : img.imports) normal += m.delayed ? 0 : 1;
        put<uint32_t>(b, dir_at(IMAGE_DIRECTORY_ENTRY_IMPORT), imp_rva + uint32_t(normal * sizeof(IMAGE_IMPORT_DESCRIPTOR)));
        put<uint32_t>(b, dir_at(IMAGE_DIRECTORY_ENTRY_IMPORT) + 4, uint32_t(sizeof(IMAGE_IMPORT_DESCRIPTOR)));
    }
    put<uint64_t>(b, dir_at(IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT), 0);
    // TLS callbacks: detach the list (and its relocation) so the OS calls none.
    if (img.tls && img.tls->callbacks_va) {
        if (auto at = img.rva_to_offset(img.tls->directory_rva)) {
            const uint32_t field_rva = img.tls->directory_rva + offsetof(IMAGE_TLS_DIRECTORY64, AddressOfCallBacks);
            put<uint64_t>(b, *at + offsetof(IMAGE_TLS_DIRECTORY64, AddressOfCallBacks), 0);
            uint32_t rel_rva = 0, rel_size = 0;
            std::memcpy(&rel_rva, b.data() + dir_at(IMAGE_DIRECTORY_ENTRY_BASERELOC), 4);
            std::memcpy(&rel_size, b.data() + dir_at(IMAGE_DIRECTORY_ENTRY_BASERELOC) + 4, 4);
            if (auto start = img.rva_to_offset(rel_rva); start && rel_size) {
                size_t p = *start, end = *start + rel_size;
                while (p + 8 <= end && p + 8 <= b.size()) {
                    uint32_t page, block;
                    std::memcpy(&page, b.data() + p, 4);
                    std::memcpy(&block, b.data() + p + 4, 4);
                    if (block < 8 || p + block > end) break;
                    for (size_t e = p + 8; e + 2 <= p + block; e += 2) {
                        uint16_t v;
                        std::memcpy(&v, b.data() + e, 2);
                        if ((v >> 12) == IMAGE_REL_BASED_DIR64 && page + (v & 0xFFF) == field_rva)
                            put<uint16_t>(b, e, 0);
                    }
                    p += block;
                }
            }
        }
    }
    return b;
}

// ------------------------------------------------------- static TLS ----

// An exe always gets TLS index 0, so its code reads `gs:[0x58]` then `[r]`
// without consulting _tls_index. Mapped as a DLL it gets another index:
// each `mov r64, gs:[0x58]` becomes `call thunk_r; nop4`, the thunk yielding
// &array[index]. (Byte match, as VIctory found exact for this title.)
static void* page_near(uintptr_t base, size_t size) {
    constexpr uintptr_t G = 0x10000;
    uintptr_t above = (base + size + G - 1) & ~(G - 1);
    for (uintptr_t k = 0; k < (1u << 14); ++k) {
        uintptr_t hints[2] = {base > (k + 1) * G ? base - (k + 1) * G : 0, above + k * G};
        for (auto h : hints) {
            if (!h) continue;
            if (void* p = VirtualAlloc(reinterpret_cast<void*>(h), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) return p;
        }
    }
    return nullptr;
}

static size_t redirect_tls(LoadedImage& img) {
    if (!img.info.tls) return 0;
    const uintptr_t base = img.base();
    auto* dir = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(base + img.info.tls->directory_rva);
    auto* index_at = reinterpret_cast<uint32_t*>(dir->AddressOfIndex);
    uint32_t index = *index_at;
    if (index == 0) return 0;
    auto* thunks = static_cast<uint8_t*>(page_near(base, img.info.size_of_image));
    if (!thunks) {
        KL_ERROR("loader", "no page near the image for TLS thunks");
        return 0;
    }
    for (uint8_t reg = 0; reg < 16; ++reg) {
        uint8_t r = reg >> 3, lo = reg & 7;
        std::vector<uint8_t> c = {0x65, uint8_t(0x48 | r << 2), 0x8B, uint8_t(lo << 3 | 4), 0x25, 0x58, 0, 0, 0,
                                  uint8_t(0x48 | r << 2 | r), 0x8D, uint8_t(0x80 | lo << 3 | lo)};
        if (lo == 4) c.push_back(0x24);
        uint32_t disp = index * 8;
        for (int i = 0; i < 4; ++i) c.push_back(uint8_t(disp >> (8 * i)));
        c.push_back(0xC3);
        std::memcpy(thunks + reg * 32, c.data(), c.size());
    }
    DWORD old;
    VirtualProtect(thunks, 4096, PAGE_EXECUTE_READ, &old);
    size_t patched = 0;
    for (auto& s : img.info.sections) {
        if (!(s.characteristics & IMAGE_SCN_MEM_EXECUTE)) continue;
        auto* code = reinterpret_cast<uint8_t*>(base + s.rva);
        size_t len = s.virtual_size;
        VirtualProtect(code, len, PAGE_EXECUTE_READWRITE, &old);
        for (size_t i = 0; i + 9 <= len;) {
            uint8_t* c = code + i;
            uint8_t rex = c[1];
            unsigned reg = ((c[3] >> 3) & 7) | ((rex & 4) ? 8 : 0);
            if (c[0] == 0x65 && (rex == 0x48 || rex == 0x4C) && c[2] == 0x8B && (c[3] & 0xC7) == 0x04 &&
                c[4] == 0x25 && c[5] == 0x58 && !c[6] && !c[7] && !c[8] && reg != 4) {
                int64_t rel = int64_t(reinterpret_cast<uintptr_t>(thunks) + reg * 32) - int64_t(reinterpret_cast<uintptr_t>(c) + 5);
                c[0] = 0xE8;
                int32_t r32 = int32_t(rel);
                std::memcpy(c + 1, &r32, 4);
                c[5] = 0x0F; c[6] = 0x1F; c[7] = 0x40; c[8] = 0x00;
                ++patched;
                i += 9;
            } else {
                ++i;
            }
        }
        VirtualProtect(code, len, old, &old);
        FlushInstructionCache(GetCurrentProcess(), code, len);
    }
    *index_at = 0;
    KL_INFO("loader", "static TLS redirected: index %u, %zu loads patched", index, patched);
    return patched;
}

// ------------------------------------------------------ thread notices ----

namespace {
struct Notice {
    LoadedImage* img;
    bool dll_main;
};
constexpr size_t kMaxNotices = 64;
Notice g_notices[kMaxNotices];
std::atomic<size_t> g_notice_count{0};
std::atomic<bool> g_notices_on{true};

using TlsCallback = void(NTAPI*)(PVOID, DWORD, PVOID);
using DllMainFn = BOOL(WINAPI*)(HINSTANCE, DWORD, LPVOID);

void notify(LoadedImage* img, DWORD reason, bool dll_main) {
    if (img->info.tls)
        for (uint32_t rva : img->info.tls->callback_rvas)
            reinterpret_cast<TlsCallback>(img->base() + rva)(img->module, reason, nullptr);
    if (dll_main && img->info.is_dll() && img->info.entry_rva)
        reinterpret_cast<DllMainFn>(img->base() + img->info.entry_rva)(img->module, reason, nullptr);
}

void NTAPI kl_tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason != DLL_THREAD_ATTACH && reason != DLL_THREAD_DETACH) return;
    if (!g_notices_on.load(std::memory_order_relaxed)) return;
    size_t n = g_notice_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) notify(g_notices[i].img, reason, g_notices[i].dll_main);
}
}  // namespace

#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:kl_tls_callback_ptr")
#pragma const_seg(".CRT$XLB")
extern "C" const PIMAGE_TLS_CALLBACK kl_tls_callback_ptr = kl_tls_callback;
#pragma const_seg()

void attach_notices(LoadedImage* img, bool dll_main) {
    size_t i = g_notice_count.load();
    if (i >= kMaxNotices) return;
    g_notices[i] = {img, dll_main};
    g_notice_count.store(i + 1, std::memory_order_release);
}

// -------------------------------------------------------------- Loader ----

Loader::Loader(fs::path title_dir, fs::path cache_dir) : title_dir_(std::move(title_dir)), cache_dir_(std::move(cache_dir)) {
    InitializeCriticalSection(&lock_);
    fs::create_directories(cache_dir_);
}

static bool read_file(const fs::path& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize(size_t(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size()));
    return bool(f);
}

std::shared_ptr<LoadedImage> Loader::map(const fs::path& file, std::string& error) {
    std::vector<uint8_t> bytes;
    if (!read_file(file, bytes)) return error = "cannot read " + file.string(), nullptr;
    auto img = std::make_shared<LoadedImage>();
    img->name = hle::lower(file.filename().string());
    if (!pe::parse(bytes, img->info, error)) return error = file.string() + ": " + error, nullptr;
    auto prepared = prepare(bytes, img->info);
    char hash[32];
    snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a(bytes));
    img->path = cache_dir_ / ("keylime-" + file.stem().string() + "-" + hash + ".dll");
    std::error_code ec;
    if (!fs::exists(img->path, ec) || fs::file_size(img->path, ec) != prepared.size()) {
        auto tmp = img->path;
        tmp += ".tmp";
        std::ofstream(tmp, std::ios::binary).write(reinterpret_cast<const char*>(prepared.data()), std::streamsize(prepared.size()));
        fs::rename(tmp, img->path, ec);
        if (ec) return error = "writing " + img->path.string() + ": " + ec.message(), nullptr;
    }
    img->module = LoadLibraryExW(img->path.c_str(), nullptr, 0);
    if (!img->module) {
        error = "LoadLibraryExW(" + img->path.string() + ") failed: " + std::to_string(GetLastError());
        return nullptr;
    }
    if (!img->info.is_dll()) redirect_tls(*img);
    KL_INFO("loader", "mapped %s at %p (preferred %llx, %u MB)", img->name.c_str(), (void*)img->module,
            (unsigned long long)img->info.image_base, img->info.size_of_image >> 20);
    return img;
}

bool Loader::wanted(const std::string& dll) const {
    return !hle::host_allows(dll) && !hle::serves(dll) && fs::is_regular_file(title_dir_ / dll);
}

void Loader::map_imports_of(const LoadedImage& img) {
    for (auto& m : img.info.imports) {
        auto name = hle::lower(m.dll);
        if (find(name) || !wanted(name)) continue;
        std::string error;
        auto dll = map(title_dir_ / name, error);
        if (!dll) {
            KL_ERROR("loader", "%s", error.c_str());
            continue;
        }
        images_.push_back(dll);  // reserve the name before recursing (cycles)
        map_imports_of(*dll);
        // dependency order: move after its imports
        auto it = std::find(images_.begin(), images_.end(), dll);
        images_.erase(it);
        images_.push_back(dll);
        unbound_.push_back(dll.get());
    }
}

bool Loader::load_main(const fs::path& exe, std::string& error) {
    main_ = map(exe, error);
    if (!main_) return false;
    map_imports_of(*main_);
    images_.push_back(main_);
    unbound_.push_back(main_.get());
    return true;
}

const LoadedImage* Loader::find(uintptr_t a) const {
    for (auto& i : images_)
        if (i->contains(a)) return i.get();
    return nullptr;
}

const LoadedImage* Loader::find(const std::string& lower_name) const {
    for (auto& i : images_)
        if (i->name == lower_name) return i.get();
    return nullptr;
}

void* Loader::resolve(const std::string& dll, const pe::ImportEntry& e, BindReport& r) {
    const std::string label = e.label();
    if (void* p = hle::lookup(dll, label)) return ++r.hle, p;
    if (void* p = hle::host_resolve(dll, e.name, e.ordinal)) return ++r.host, p;
    if (auto* t = find(hle::lower(dll))) {
        FARPROC p = e.name.empty() ? GetProcAddress(t->module, MAKEINTRESOURCEA(e.ordinal)) : GetProcAddress(t->module, e.name.c_str());
        if (p) return ++r.title, reinterpret_cast<void*>(p);
    }
    r.missing.push_back(dll + "!" + label);
    return hle::make_stub(dll, label);
}

BindReport Loader::bind(LoadedImage& img) {
    BindReport r;
    for (auto& m : img.info.imports) {
        for (auto& e : m.entries) {
            void* target = resolve(m.dll, e, r);
            auto* slot = reinterpret_cast<void**>(img.base() + e.iat_rva);
            DWORD old;
            VirtualProtect(slot, 8, PAGE_READWRITE, &old);
            *slot = target;
            VirtualProtect(slot, 8, old, &old);
        }
    }
    KL_INFO("loader", "bound %s: %zu hle, %zu host, %zu title, %zu missing", img.name.c_str(), r.hle, r.host,
            r.title, r.missing.size());
    for (auto& m : r.missing) KL_INFO("loader", "  missing %s", m.c_str());
    return r;
}

void Loader::bind_all() {
    EnterCriticalSection(&lock_);
    auto todo = std::move(unbound_);
    unbound_.clear();
    LeaveCriticalSection(&lock_);
    for (auto* img : todo) bind(*img);
}

HMODULE Loader::load_late(const std::string& file) {
    auto name = hle::lower(fs::path(file).filename().string());
    EnterCriticalSection(&lock_);
    if (auto* img = find(name)) {
        LeaveCriticalSection(&lock_);
        return img->module;
    }
    if (!wanted(name)) {
        LeaveCriticalSection(&lock_);
        return nullptr;
    }
    std::string error;
    auto img = map(title_dir_ / name, error);
    HMODULE result = nullptr;
    if (img) {
        size_t before = images_.size();
        map_imports_of(*img);
        images_.insert(images_.begin(), img);
        unbound_.push_back(img.get());
        std::vector<LoadedImage*> fresh(unbound_.begin(), unbound_.end());
        LeaveCriticalSection(&lock_);
        bind_all();
        (void)before;
        for (auto* d : fresh) {
            notify(d, DLL_PROCESS_ATTACH, true);
            attach_notices(d, true);
        }
        KL_INFO("loader", "late title DLL %s at %p", name.c_str(), (void*)img->module);
        return img->module;
    }
    LeaveCriticalSection(&lock_);
    KL_ERROR("loader", "%s", error.c_str());
    return result;
}

uint32_t Loader::run_main() {
    struct Ctx {
        Loader* self;
    } ctx{this};
    auto body = [](LPVOID p) -> DWORD {
        auto* self = static_cast<Ctx*>(p)->self;
        for (auto& img : self->images_) {
            if (img == self->main_) continue;
            KL_INFO("loader", "attaching %s", img->name.c_str());
            notify(img.get(), DLL_PROCESS_ATTACH, true);
            attach_notices(img.get(), true);
        }
        notify(self->main_.get(), DLL_PROCESS_ATTACH, false);
        attach_notices(self->main_.get(), false);
        KL_INFO("loader", "entering the title at %s", describe(self->main_->base() + self->main_->info.entry_rva).c_str());
        using Entry = DWORD(WINAPI*)(void*);
        auto entry = reinterpret_cast<Entry>(self->main_->base() + self->main_->info.entry_rva);
        return entry(reinterpret_cast<void*>(__readgsqword(0x60)));
    };
    SIZE_T stack = SIZE_T(main_->info.stack_reserve < (1u << 20) ? (1u << 20) : main_->info.stack_reserve);
    HANDLE t = CreateThread(nullptr, stack, body, &ctx, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!t) return ~0u;
    SetThreadDescription(t, L"title-main");
    WaitForSingleObject(t, INFINITE);
    DWORD code = 0;
    GetExitCodeThread(t, &code);
    return code;
}

// Our own code: function and line from keylime.pdb.
static std::string symbol(uintptr_t a) {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        ok = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    });
    if (!ok) return {};
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    auto* si = reinterpret_cast<SYMBOL_INFO*>(buf);
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen = 255;
    DWORD64 disp = 0;
    if (!SymFromAddr(GetCurrentProcess(), a, &disp, si)) return {};
    std::string out = std::string(" ") + si->Name;
    IMAGEHLP_LINE64 line{sizeof line};
    DWORD d32 = 0;
    if (SymGetLineFromAddr64(GetCurrentProcess(), a, &d32, &line))
        out += " (" + fs::path(line.FileName).filename().string() + ":" + std::to_string(line.LineNumber) + ")";
    return out;
}

std::string describe(uintptr_t a) {
    char buf[160];
    if (g_loader)
        if (auto* img = g_loader->find(a)) {
            snprintf(buf, sizeof buf, "%s+0x%llx", img->name.c_str(), (unsigned long long)(a - img->base()));
            return buf;
        }
    HMODULE m = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(a), &m) && m) {
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(m, path, MAX_PATH);
        snprintf(buf, sizeof buf, "%ls+0x%llx", fs::path(path).filename().c_str(),
                 (unsigned long long)(a - reinterpret_cast<uintptr_t>(m)));
        if (m == GetModuleHandleW(nullptr)) return buf + symbol(a);
        return buf;
    }
    snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)a);
    return buf;
}

}  // namespace kl
