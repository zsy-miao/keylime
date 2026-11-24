// keylime: virtual drives (G: package, T: temp) over host folders.
#pragma once
#include <filesystem>
#include <string>

namespace kl::files {

void mount(char letter, const std::filesystem::path& root);
// The host path for a path on a virtual drive; false if `path` is on none.
bool translate(const wchar_t* path, std::wstring& out);

}  // namespace kl::files
