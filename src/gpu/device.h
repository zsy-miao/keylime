// keylime GPU: the console D3D12.X device (one per process).
#pragma once
#include "gpu/object.h"

namespace kl::gpu {

// The device as any interface it answers to (for GetDevice of children and DXGI).
HRESULT device_query(const Iid* iid, void** out);
void* device_object();
bool device_is(const Iid* iid);
// Created as ID3D12Device11: lists and queues use the newer layouts (two more methods each).
bool newer_layouts();

// The table slot setter other modules use to add device methods (before creation).
void device_set(size_t slot, void* fn);

namespace dxgi {
void* device();
bool is_device(const Iid* iid);
}  // namespace dxgi

}  // namespace kl::gpu
