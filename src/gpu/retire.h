// keylime GPU: deferred destruction. `f` runs once every queue has finished
// the work submitted so far plus the next submission on it (so host objects
// a recording uses live until the GPU is done with them).
#pragma once
#include <functional>

namespace kl::gpu {
void retire(std::function<void()> f);
}
