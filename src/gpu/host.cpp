#include "gpu/host.h"

#include <windows.h>

#include <cstring>
#include <set>
#include <vector>

#include "log.h"

namespace kl::gpu::host {
namespace {

Host g_host;
bool g_ready = false;

std::set<std::string> device_extensions(VkPhysicalDevice p) {
    uint32_t n = 0;
    vk::vkEnumerateDeviceExtensionProperties(p, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> v(n);
    vk::vkEnumerateDeviceExtensionProperties(p, nullptr, &n, v.data());
    std::set<std::string> s;
    for (auto& e : v) s.insert(e.extensionName);
    return s;
}

int score(VkPhysicalDevice p) {
    VkPhysicalDeviceProperties2 pr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vk::vkGetPhysicalDeviceProperties2(p, &pr);
    int s = 0;
    if (pr.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) s += 100;
    if (pr.properties.vendorID == 0x10DE) s += 10;  // NVIDIA: our tuned path
    if (pr.properties.apiVersion >= VK_API_VERSION_1_3) s += 5;
    return s;
}

void pick_memory(Host& h) {
    auto& m = h.memory;
    for (uint32_t i = 0; i < m.memoryHeapCount; ++i)
        if (m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            if (m.memoryHeaps[i].size > h.caps.vram_bytes) h.caps.vram_bytes = m.memoryHeaps[i].size;
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i) {
        auto f = m.memoryTypes[i].propertyFlags;
        uint64_t heap = m.memoryHeaps[m.memoryTypes[i].heapIndex].size;
        const auto DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                   HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if ((f & DL) && !(f & HV) && h.caps.vram_type == ~0u) h.caps.vram_type = i;
        // Resizable BAR: host-visible VRAM larger than the classic 256 MiB window.
        if ((f & DL) && (f & HV) && (f & HC) && heap > (256ull << 20) && h.caps.rebar_type == ~0u) {
            h.caps.rebar_type = i;
            h.caps.rebar = true;
            h.caps.rebar_bytes = heap;
        }
        if (!(f & DL) && (f & HV) && (f & HC) && !(f & CA) && h.caps.host_type == ~0u) h.caps.host_type = i;
        if (!(f & DL) && (f & HV) && (f & CA) && h.caps.readback_type == ~0u) h.caps.readback_type = i;
    }
    if (h.caps.host_type == ~0u) h.caps.host_type = h.caps.readback_type;
}

}  // namespace

Host& get() { return g_host; }

void vram(uint64_t& budget, uint64_t& usage) {
    Host& h = g_host;
    budget = h.caps.vram_bytes, usage = 0;
    if (!h.caps.memory_budget) return;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    p.pNext = &b;
    vk::vkGetPhysicalDeviceMemoryProperties2(h.physical, &p);
    uint64_t best = 0;
    for (uint32_t i = 0; i < p.memoryProperties.memoryHeapCount; ++i)
        if ((p.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && p.memoryProperties.memoryHeaps[i].size > best) {
            best = p.memoryProperties.memoryHeaps[i].size;
            budget = b.heapBudget[i], usage = b.heapUsage[i];
        }
}
bool ready() { return g_ready; }

bool start() {
    Host& h = g_host;
    if (!vk::load_global()) {
        KL_ERROR("vk", "vulkan-1.dll not found: update the GPU driver");
        return false;
    }
    uint32_t api = VK_API_VERSION_1_0;
    if (vk::vkEnumerateInstanceVersion) vk::vkEnumerateInstanceVersion(&api);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "keylime";
    app.pEngineName = "keylime";
    app.apiVersion = VK_API_VERSION_1_3;
    const char* inst_ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = inst_ext;
    // KEYLIME_VALIDATION=1 turns on the Khronos validation layer (needs the SDK's layer installed).
    char v[4]{};
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (GetEnvironmentVariableA("KEYLIME_VALIDATION", v, 4) && v[0] == '1') {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
    }
    if (!KL_VK_CHECK(vk::vkCreateInstance(&ici, nullptr, &h.instance))) return false;
    vk::load_instance(h.instance);

    uint32_t n = 0;
    vk::vkEnumeratePhysicalDevices(h.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vk::vkEnumeratePhysicalDevices(h.instance, &n, devs.data());
    if (devs.empty()) {
        KL_ERROR("vk", "no Vulkan GPU");
        return false;
    }
    int best = -1;
    for (auto d : devs)
        if (score(d) > best) best = score(d), h.physical = d;

    VkPhysicalDeviceProperties2 pr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceSubgroupProperties subp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    pr.pNext = &hostp;
    hostp.pNext = &subp;
    vk::vkGetPhysicalDeviceProperties2(h.physical, &pr);
    h.props = pr.properties;
    h.name = h.props.deviceName;
    vk::vkGetPhysicalDeviceMemoryProperties(h.physical, &h.memory);
    pick_memory(h);
    auto ext = device_extensions(h.physical);
    auto has = [&](const char* e) { return ext.count(e) != 0; };
    h.caps.mesh_shader = has(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    h.caps.host_memory_import = has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    h.caps.cuda_binary = has("VK_NVX_binary_import");
    h.caps.ray_tracing = has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) && has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    h.caps.descriptor_buffer = has(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    h.caps.host_import_alignment = hostp.minImportedHostPointerAlignment;
    h.caps.memory_budget = has(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    KL_INFO("vk", "GPU %s (driver %u.%u, Vulkan %u.%u.%u), subgroup %u lanes", h.name.c_str(),
            (h.props.driverVersion >> 22) & 0x3ff, (h.props.driverVersion >> 14) & 0xff, VK_API_VERSION_MAJOR(h.props.apiVersion),
            VK_API_VERSION_MINOR(h.props.apiVersion), VK_API_VERSION_PATCH(h.props.apiVersion), subp.subgroupSize);
    KL_INFO("vk", "VRAM %llu MiB, Resizable BAR %s (%llu MiB host-visible VRAM)", h.caps.vram_bytes >> 20,
            h.caps.rebar ? "ON" : "off", h.caps.rebar_bytes >> 20);
    KL_INFO("vk", "mesh shaders %d, host memory import %d (align %llu), CUDA binary import %d, ray tracing %d, descriptor buffer %d",
            h.caps.mesh_shader, h.caps.host_memory_import, (unsigned long long)h.caps.host_import_alignment,
            h.caps.cuda_binary, h.caps.ray_tracing, h.caps.descriptor_buffer);

    // Queues: one graphics, one async compute, one transfer (DMA) family if present.
    uint32_t qn = 0;
    vk::vkGetPhysicalDeviceQueueFamilyProperties(h.physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vk::vkGetPhysicalDeviceQueueFamilyProperties(h.physical, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) {
        auto f = qf[i].queueFlags;
        if ((f & VK_QUEUE_GRAPHICS_BIT) && h.queues.graphics_family == ~0u) h.queues.graphics_family = i;
        else if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT) && h.queues.compute_family == ~0u) h.queues.compute_family = i;
        else if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && h.queues.transfer_family == ~0u)
            h.queues.transfer_family = i;
    }
    if (h.queues.graphics_family == ~0u) {
        KL_ERROR("vk", "no graphics queue");
        return false;
    }
    if (h.queues.compute_family == ~0u) h.queues.compute_family = h.queues.graphics_family;
    if (h.queues.transfer_family == ~0u) h.queues.transfer_family = h.queues.compute_family;
    std::set<uint32_t> fams = {h.queues.graphics_family, h.queues.compute_family, h.queues.transfer_family};
    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qci;
    for (uint32_t f : fams) {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = f;
        q.queueCount = 1;
        q.pQueuePriorities = &prio;
        qci.push_back(q);
    }

    // Features: query what exists, keep what we use.
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceMeshShaderFeaturesEXT fmesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    f2.pNext = &f11;
    f11.pNext = &f12;
    f12.pNext = &f13;
    if (h.caps.mesh_shader) f13.pNext = &fmesh;
    vk::vkGetPhysicalDeviceFeatures2(h.physical, &f2);
    h.caps.buffer_address = f12.bufferDeviceAddress;
    h.caps.int64_atomics = f12.shaderBufferInt64Atomics;
    if (!h.caps.buffer_address || !f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) {
        KL_ERROR("vk", "the GPU lacks buffer device address / dynamic rendering / sync2 / timeline semaphores");
        return false;
    }
    // Things we do not want on (cost or irrelevance).
    f12.bufferDeviceAddressCaptureReplay = VK_FALSE;
    f12.bufferDeviceAddressMultiDevice = VK_FALSE;
    fmesh.multiviewMeshShader = VK_FALSE;
    fmesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
    f2.features.robustBufferAccess = VK_FALSE;  // the title's own bounds are the console's

    std::vector<const char*> dext = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (h.caps.mesh_shader) dext.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    if (h.caps.host_memory_import) dext.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (h.caps.memory_budget) dext.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = uint32_t(qci.size());
    dci.pQueueCreateInfos = qci.data();
    dci.enabledExtensionCount = uint32_t(dext.size());
    dci.ppEnabledExtensionNames = dext.data();
    if (!KL_VK_CHECK(vk::vkCreateDevice(h.physical, &dci, nullptr, &h.device))) return false;
    vk::load_device(h.device);
    vk::vkGetDeviceQueue(h.device, h.queues.graphics_family, 0, &h.queues.graphics);
    vk::vkGetDeviceQueue(h.device, h.queues.compute_family, 0, &h.queues.compute);
    vk::vkGetDeviceQueue(h.device, h.queues.transfer_family, 0, &h.queues.transfer);
    KL_INFO("vk", "device ready: queues graphics %u, compute %u, transfer %u", h.queues.graphics_family,
            h.queues.compute_family, h.queues.transfer_family);
    g_ready = true;
    return true;
}

}  // namespace kl::gpu::host
