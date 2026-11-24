// keylime: process-wide state shared by the HLE modules.
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

#include "loader.h"

namespace kl {

enum class Console { SeriesX, SeriesS };

struct Runtime {
    std::filesystem::path title_dir;   // the package folder (G:)
    std::filesystem::path exe_name;    // game_scarlett_master.exe
    std::filesystem::path data_dir;    // %LOCALAPPDATA%\keylime\data\<titleid>
    std::filesystem::path cache_dir;   // %LOCALAPPDATA%\keylime\cache
    uint32_t title_id = 0;
    Console console = Console::SeriesX;

    // The original file name of a mapped image (exe or title DLL).
    std::filesystem::path file_of(const LoadedImage* img) const {
        return img == loader().main() ? exe_name : std::filesystem::path(img->name);
    }
};

Runtime& runtime();

}  // namespace kl
