#include "gpu/present.h"

#include "log.h"
#include "stats.h"

namespace kl::gpu::present {
HRESULT present(VkQueue, std::mutex*, uint32_t planes, const void*, const void*) {
    stats::add(stats::Presents);
    static int n = 0;
    if (n++ < 3) KL_INFO("gpu", "PresentX: %u planes (not shown yet)", planes);
    return S_OK;
}
}  // namespace kl::gpu::present
