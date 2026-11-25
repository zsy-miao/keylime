// keylime GPU: Vulkan entry points, loaded from the driver's vulkan-1.dll (no SDK).
#pragma once
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define KL_VK_GLOBAL(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceExtensionProperties) X(vkEnumerateInstanceVersion)

#define KL_VK_INSTANCE(X)                                                                                  \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties2)                   \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties)                            \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice)       \
    X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceFormatProperties) X(vkCreateWin32SurfaceKHR)               \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR)

#define KL_VK_DEVICE(X)                                                                                    \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkQueueSubmit2) X(vkQueueWaitIdle) X(vkDeviceWaitIdle)        \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkGetBufferDeviceAddress) X(vkCreateImage)    \
    X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView)            \
    X(vkDestroyImageView) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool)             \
    X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)      \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores) X(vkSignalSemaphore)                    \
    X(vkGetSemaphoreCounterValue) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)   \
    X(vkCmdPipelineBarrier2) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImage)                \
    X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer)                      \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR)   \
    X(vkQueuePresentKHR) X(vkGetMemoryHostPointerPropertiesEXT)                              X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)     X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateComputePipelines)              X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets)     X(vkUpdateDescriptorSets) X(vkCreateSampler) X(vkDestroySampler) X(vkCmdBindPipeline)                      X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdDraw) X(vkCmdDrawIndexed)        X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdSetViewport) X(vkCmdSetScissor)     X(vkCmdBindVertexBuffers2) X(vkCmdBindIndexBuffer) X(vkCmdSetPrimitiveTopology) X(vkCmdSetBlendConstants)                X(vkCmdSetStencilReference) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdClearDepthStencilImage)                           X(vkCmdCopyImageToBuffer) X(vkCmdClearAttachments) X(vkResetCommandBuffer) X(vkCmdDrawIndirect)                         X(vkCmdDrawIndexedIndirect) X(vkCmdDispatchIndirect)

namespace kl::vk {
#define KL_VK_DECLARE(name) extern PFN_##name name;
KL_VK_GLOBAL(KL_VK_DECLARE)
KL_VK_INSTANCE(KL_VK_DECLARE)
KL_VK_DEVICE(KL_VK_DECLARE)
#undef KL_VK_DECLARE

bool load_global();
void load_instance(VkInstance);
void load_device(VkDevice);
const char* result_text(VkResult r);
}  // namespace kl::vk

#define KL_VK_CHECK(expr)                                                                       \
    ([&] {                                                                                      \
        VkResult kl_r = (expr);                                                                 \
        if (kl_r != VK_SUCCESS) KL_ERROR("vk", "%s failed: %s", #expr, ::kl::vk::result_text(kl_r)); \
        return kl_r == VK_SUCCESS;                                                              \
    }())
