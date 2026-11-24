// keylime: runs the title's x86-64 code natively in our process.
//
// Same approach as VIctory's loader: write a lightly patched copy of each
// image (marked as a DLL, no entry point, no import table, no TLS callbacks)
// and LoadLibrary it, so Windows relocates it, allocates static TLS and
// registers unwind tables while running none of its code. We then bind every
// import ourselves and run DllMain / TLS callbacks / the entry point.
#pragma once
#include <windows.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "pe.h"

namespace kl {

struct LoadedImage {
    std::string name;  // lower-case file name
    std::filesystem::path path;
    HMODULE module = nullptr;
    pe::Image info;
    uintptr_t base() const { return reinterpret_cast<uintptr_t>(module); }
    bool contains(uintptr_t a) const { return a >= base() && a < base() + info.size_of_image; }
};

struct BindReport {
    size_t hle = 0, host = 0, title = 0;
    std::vector<std::string> missing;
};

class Loader {
public:
    Loader(std::filesystem::path title_dir, std::filesystem::path cache_dir);

    // Maps the main executable and every title DLL it (transitively) imports.
    bool load_main(const std::filesystem::path& exe, std::string& error);
    // Binds the imports of every mapped image not yet bound.
    void bind_all();
    // Runs DllMains, TLS callbacks and the entry point on a new thread; waits for it.
    uint32_t run_main();

    // A title DLL the title loads itself (LoadLibrary). nullptr if `file` is no title image.
    HMODULE load_late(const std::string& file);

    const LoadedImage* main() const { return main_.get(); }
    const LoadedImage* find(uintptr_t address) const;
    const LoadedImage* find(const std::string& lower_name) const;
    const std::vector<std::shared_ptr<LoadedImage>>& images() const { return images_; }

private:
    std::shared_ptr<LoadedImage> map(const std::filesystem::path& file, std::string& error);
    void map_imports_of(const LoadedImage& img);
    bool wanted(const std::string& dll) const;
    BindReport bind(LoadedImage& img);
    void* resolve(const std::string& dll, const pe::ImportEntry& e, BindReport& r);

    std::filesystem::path title_dir_, cache_dir_;
    std::shared_ptr<LoadedImage> main_;
    std::vector<std::shared_ptr<LoadedImage>> images_;  // title DLLs in dependency order, then main
    std::vector<LoadedImage*> unbound_;
    CRITICAL_SECTION lock_;
};

Loader& loader();  // the process-wide instance (set by main)
void set_loader(Loader* l);

// Thread notices for attached images (our own TLS callback forwards them).
void attach_notices(LoadedImage* img, bool dll_main);
// "image+0xRVA" for an address, or the raw address.
std::string describe(uintptr_t address);

}  // namespace kl
