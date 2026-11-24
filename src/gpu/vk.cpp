#include "gpu/vk.h"

#include <windows.h>

namespace kl::vk {

#define KL_VK_DEFINE(name) PFN_##name name = nullptr;
KL_VK_GLOBAL(KL_VK_DEFINE)
KL_VK_INSTANCE(KL_VK_DEFINE)
KL_VK_DEVICE(KL_VK_DEFINE)
#undef KL_VK_DEFINE

static PFN_vkGetInstanceProcAddr g_gipa = nullptr;

bool load_global() {
    HMODULE m = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return false;
    g_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(m, "vkGetInstanceProcAddr"));
    if (!g_gipa) return false;
#define KL_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(g_gipa(nullptr, #name));
    KL_VK_GLOBAL(KL_VK_LOAD)
#undef KL_VK_LOAD
    return vkCreateInstance != nullptr;
}

void load_instance(VkInstance i) {
#define KL_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(g_gipa(i, #name));
    KL_VK_INSTANCE(KL_VK_LOAD)
#undef KL_VK_LOAD
}

void load_device(VkDevice d) {
#define KL_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(d, #name));
    KL_VK_DEVICE(KL_VK_LOAD)
#undef KL_VK_LOAD
}

const char* result_text(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "success";
        case VK_NOT_READY: return "not ready";
        case VK_TIMEOUT: return "timeout";
        case VK_SUBOPTIMAL_KHR: return "suboptimal";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
        case VK_ERROR_INITIALIZATION_FAILED: return "initialization failed";
        case VK_ERROR_DEVICE_LOST: return "DEVICE LOST";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "extension not present";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "feature not present";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "invalid external handle";
        case VK_ERROR_OUT_OF_DATE_KHR: return "out of date";
        case VK_ERROR_SURFACE_LOST_KHR: return "surface lost";
        default: return "error";
    }
}

}  // namespace kl::vk
