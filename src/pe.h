// keylime: PE32+ parsing (file bytes, not a mapped image).
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kl::pe {

struct Section {
    std::string name;
    uint32_t rva = 0, virtual_size = 0, raw_offset = 0, raw_size = 0, characteristics = 0;
};

struct ImportEntry {
    std::string name;      // empty when by ordinal
    uint16_t ordinal = 0;  // valid when name is empty
    uint32_t iat_rva = 0;  // the slot we write
    std::string label() const { return name.empty() ? "#" + std::to_string(ordinal) : name; }
};

struct ImportModule {
    std::string dll;
    bool delayed = false;
    std::vector<ImportEntry> entries;
};

struct Tls {
    uint32_t directory_rva = 0;
    uint64_t callbacks_va = 0;              // AddressOfCallBacks (preferred-base VA)
    std::vector<uint32_t> callback_rvas;    // from the file, relative to image base
};

struct Image {
    uint16_t characteristics = 0, dll_characteristics = 0;
    uint64_t image_base = 0, stack_reserve = 0;
    uint32_t entry_rva = 0, size_of_image = 0;
    uint32_t file_header_off = 0, opt_header_off = 0;
    bool has_relocations = false;
    std::vector<Section> sections;
    std::vector<ImportModule> imports;  // normal then delayed
    std::optional<Tls> tls;

    bool is_dll() const { return (characteristics & 0x2000) != 0; }
    std::optional<uint32_t> rva_to_offset(uint32_t rva) const;
    size_t import_count() const;
};

// Parses `bytes`; returns false with `error` set on malformed input.
bool parse(const std::vector<uint8_t>& bytes, Image& out, std::string& error);

}  // namespace kl::pe
