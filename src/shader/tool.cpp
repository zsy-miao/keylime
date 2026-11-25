// keylime shader <file|dir> [--dis]: parses console shader containers, disassembles their
// GFX10 code and reports decoder coverage and an opcode histogram.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "shader/container.h"
#include "shader/gfx10.h"

namespace fs = std::filesystem;

namespace kl::shader {

int tool(const fs::path& target, bool dis) {
    std::vector<fs::path> files;
    if (fs::is_directory(target)) {
        for (auto& e : fs::directory_iterator(target))
            if (e.path().extension() == ".bin") files.push_back(e.path());
    } else {
        files.push_back(target);
    }
    std::map<std::string, size_t> ops;
    size_t programs = 0, insts = 0, failed_programs = 0, bad_files = 0, wave64 = 0, with_bitcode = 0;
    std::map<std::string, size_t> stages;
    for (auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)), {});
        Container c;
        std::string err;
        if (!parse(b.data(), b.size(), c, err)) {
            ++bad_files;
            printf("%ls: %s\n", f.filename().c_str(), err.c_str());
            continue;
        }
        with_bitcode += c.has_bitcode;
        for (auto& p : c.programs) {
            ++programs;
            ++stages[stage_name(p.stage)];
            wave64 += p.wave64;
            auto code = p.patched_code();
            if (dis)
                printf("== %ls %s wave%d vgprs %u, %zu root params, %zu resources, %zu bytes\n", f.filename().c_str(),
                       stage_name(p.stage), p.wave64 ? 64 : 32, p.vgprs, p.layout.size(), p.used.size(), code.size());
            uint32_t at = 0;
            bool ok = true;
            while (at < code.size()) {
                gfx10::Inst i;
                if (!gfx10::decode(code.data(), code.size(), at, i)) {
                    uint32_t w = 0;
                    memcpy(&w, code.data() + at, 4);
                    if (dis) printf("  %04x: ??? %08x\n", at, w);
                    ok = false;
                    break;
                }
                ++insts;
                ++ops[i.name];
                if (dis) printf("  %04x: %s\n", at, gfx10::format(i).c_str());
                if (gfx10::ends_program(i)) break;
                at += i.size;
            }
            failed_programs += !ok;
        }
    }
    printf("\n%zu files (%zu unreadable, %zu with DXIL bitcode), %zu programs, %zu wave64\n", files.size(), bad_files,
           with_bitcode, programs, wave64);
    for (auto& [k, v] : stages) printf("  %s: %zu\n", k.c_str(), v);
    printf("%zu instructions decoded, %zu programs with an undecodable instruction\n", insts, failed_programs);
    printf("%zu distinct opcodes:\n", ops.size());
    std::vector<std::pair<size_t, std::string>> sorted;
    for (auto& [k, v] : ops) sorted.push_back({v, k});
    std::sort(sorted.rbegin(), sorted.rend());
    for (auto& [v, k] : sorted) printf("  %7zu %s\n", v, k.c_str());
    return failed_programs || bad_files ? 1 : 0;
}

}  // namespace kl::shader
