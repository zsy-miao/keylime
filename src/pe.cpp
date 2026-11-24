#include "pe.h"

#include <windows.h>

#include <cstring>

namespace kl::pe {

std::optional<uint32_t> Image::rva_to_offset(uint32_t rva) const {
    for (auto& s : sections) {
        uint32_t span = s.virtual_size > s.raw_size ? s.virtual_size : s.raw_size;
        if (rva >= s.rva && rva < s.rva + span) {
            uint32_t d = rva - s.rva;
            if (d >= s.raw_size) return std::nullopt;  // not file-backed
            return s.raw_offset + d;
        }
    }
    return std::nullopt;
}

size_t Image::import_count() const {
    size_t n = 0;
    for (auto& m : imports) n += m.entries.size();
    return n;
}

namespace {

struct Reader {
    const std::vector<uint8_t>& b;
    template <class T>
    bool get(size_t at, T& out) const {
        if (at + sizeof(T) > b.size()) return false;
        std::memcpy(&out, b.data() + at, sizeof(T));
        return true;
    }
    std::string cstr(size_t at) const {
        std::string s;
        while (at < b.size() && b[at]) s.push_back(char(b[at++]));
        return s;
    }
};

bool read_thunks(const Reader& r, const Image& img, uint32_t names_rva, uint32_t iat_rva,
                 ImportModule& mod) {
    uint32_t list = names_rva ? names_rva : iat_rva;
    auto off = img.rva_to_offset(list);
    if (!off) return false;
    for (uint32_t i = 0;; ++i) {
        uint64_t thunk = 0;
        if (!r.get(*off + i * 8, thunk)) return false;
        if (!thunk) break;
        ImportEntry e;
        e.iat_rva = iat_rva + i * 8;
        if (thunk & (1ull << 63)) {
            e.ordinal = uint16_t(thunk & 0xFFFF);
        } else {
            auto name_off = img.rva_to_offset(uint32_t(thunk & 0x7FFFFFFF));
            if (!name_off) return false;
            e.name = r.cstr(*name_off + 2);
        }
        mod.entries.push_back(std::move(e));
    }
    return true;
}

}  // namespace

bool parse(const std::vector<uint8_t>& bytes, Image& img, std::string& error) {
    Reader r{bytes};
    IMAGE_DOS_HEADER dos;
    if (!r.get(0, dos) || dos.e_magic != IMAGE_DOS_SIGNATURE) return error = "no MZ header", false;
    IMAGE_NT_HEADERS64 nt;
    if (!r.get(size_t(dos.e_lfanew), nt) || nt.Signature != IMAGE_NT_SIGNATURE)
        return error = "no PE header", false;
    if (nt.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return error = "not PE32+", false;

    img.file_header_off = uint32_t(dos.e_lfanew + 4);
    img.opt_header_off = img.file_header_off + sizeof(IMAGE_FILE_HEADER);
    img.characteristics = nt.FileHeader.Characteristics;
    img.dll_characteristics = nt.OptionalHeader.DllCharacteristics;
    img.image_base = nt.OptionalHeader.ImageBase;
    img.stack_reserve = nt.OptionalHeader.SizeOfStackReserve;
    img.entry_rva = nt.OptionalHeader.AddressOfEntryPoint;
    img.size_of_image = nt.OptionalHeader.SizeOfImage;

    size_t sec_at = img.opt_header_off + nt.FileHeader.SizeOfOptionalHeader;
    for (int i = 0; i < nt.FileHeader.NumberOfSections; ++i) {
        IMAGE_SECTION_HEADER sh;
        if (!r.get(sec_at + i * sizeof sh, sh)) return error = "truncated section table", false;
        Section s;
        s.name.assign(reinterpret_cast<const char*>(sh.Name), strnlen(reinterpret_cast<const char*>(sh.Name), 8));
        s.rva = sh.VirtualAddress;
        s.virtual_size = sh.Misc.VirtualSize;
        s.raw_offset = sh.PointerToRawData;
        s.raw_size = sh.SizeOfRawData;
        s.characteristics = sh.Characteristics;
        img.sections.push_back(s);
    }

    auto dir = [&](int i) -> IMAGE_DATA_DIRECTORY {
        if (uint32_t(i) >= nt.OptionalHeader.NumberOfRvaAndSizes) return {};
        return nt.OptionalHeader.DataDirectory[i];
    };
    img.has_relocations = dir(IMAGE_DIRECTORY_ENTRY_BASERELOC).Size != 0 &&
                          !(img.characteristics & IMAGE_FILE_RELOCS_STRIPPED);

    if (auto d = dir(IMAGE_DIRECTORY_ENTRY_IMPORT); d.VirtualAddress) {
        auto off = img.rva_to_offset(d.VirtualAddress);
        if (!off) return error = "import directory not file-backed", false;
        for (size_t at = *off;; at += sizeof(IMAGE_IMPORT_DESCRIPTOR)) {
            IMAGE_IMPORT_DESCRIPTOR desc;
            if (!r.get(at, desc)) return error = "truncated imports", false;
            if (!desc.Name && !desc.FirstThunk) break;
            ImportModule mod;
            auto name_off = img.rva_to_offset(desc.Name);
            if (!name_off) return error = "import name not file-backed", false;
            mod.dll = r.cstr(*name_off);
            if (!read_thunks(r, img, desc.OriginalFirstThunk, desc.FirstThunk, mod))
                return error = "bad import thunks in " + mod.dll, false;
            img.imports.push_back(std::move(mod));
        }
    }
    if (auto d = dir(IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT); d.VirtualAddress) {
        auto off = img.rva_to_offset(d.VirtualAddress);
        if (off) {
            for (size_t at = *off;; at += sizeof(IMAGE_DELAYLOAD_DESCRIPTOR)) {
                IMAGE_DELAYLOAD_DESCRIPTOR desc;
                if (!r.get(at, desc) || !desc.DllNameRVA) break;
                ImportModule mod;
                mod.delayed = true;
                auto name_off = img.rva_to_offset(desc.DllNameRVA);
                if (!name_off) break;
                mod.dll = r.cstr(*name_off);
                if (read_thunks(r, img, desc.ImportNameTableRVA, desc.ImportAddressTableRVA, mod))
                    img.imports.push_back(std::move(mod));
            }
        }
    }
    if (auto d = dir(IMAGE_DIRECTORY_ENTRY_TLS); d.VirtualAddress) {
        auto off = img.rva_to_offset(d.VirtualAddress);
        IMAGE_TLS_DIRECTORY64 t;
        if (off && r.get(*off, t)) {
            Tls tls;
            tls.directory_rva = d.VirtualAddress;
            tls.callbacks_va = t.AddressOfCallBacks;
            if (t.AddressOfCallBacks) {
                auto cb_off = img.rva_to_offset(uint32_t(t.AddressOfCallBacks - img.image_base));
                for (uint32_t i = 0; cb_off; ++i) {
                    uint64_t va = 0;
                    if (!r.get(*cb_off + i * 8, va) || !va) break;
                    tls.callback_rvas.push_back(uint32_t(va - img.image_base));
                }
            }
            img.tls = tls;
        }
    }
    return true;
}

}  // namespace kl::pe
