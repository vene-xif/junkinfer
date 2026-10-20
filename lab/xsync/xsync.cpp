// Cross-device GPU<->GPU sync latency through one pinned host allocation imported into both devices
// (VK_EXT_external_memory_host). usage: xsync [iters] [payload words] ; build: g++ -O2 -o xsync xsync.cpp -lvulkan
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, (int) r_, __LINE__); exit(1); } } while (0)

struct Dev {
    VkPhysicalDevice pd; VkDevice d; VkQueue q; VkBuffer b; VkDeviceMemory m; VkPipeline pl; VkPipelineLayout lay;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds; VkCommandPool cp; VkCommandBuffer cb; VkFence f; void * map = nullptr;
};

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const uint32_t iters = argc > 1 ? atoi(argv[1]) : 10000, np = argc > 2 ? atoi(argv[2]) : 0;
    const size_t SZ = 4 << 20;
    void * host = aligned_alloc(4096, SZ); memset(host, 0, SZ);
    const bool relay = getenv("XS_RELAY") != nullptr, cycle = getenv("XS_CYCLE") != nullptr, evmode = getenv("XS_EVENT") != nullptr;
    const uint32_t want = getenv("XS_FLAGS") ? strtoul(getenv("XS_FLAGS"), nullptr, 0) : 0x6;
    std::ifstream fs((cycle || evmode) ? "cy.spv" : relay ? "pr.spv" : "pp.spv", std::ios::binary); std::vector<char> spv((std::istreambuf_iterator<char>(fs)), {});
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<Dev> ds;
    for (auto pd : pds) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp); if (pp.vendorID == 0x1002) ds.push_back({pd}); }
    if (ds.size() < 2) { fprintf(stderr, "need 2 AMD devices\n"); return 1; }
    auto getHP = (PFN_vkGetMemoryHostPointerPropertiesEXT) nullptr;
    for (auto & D : ds) {
        const uint32_t qf = getenv("XS_QF") ? atoi(getenv("XS_QF")) : 0;
        float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        const char * ext[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host", "VK_AMD_buffer_marker"};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = ext;
        CK(vkCreateDevice(D.pd, &dci, nullptr, &D.d)); vkGetDeviceQueue(D.d, qf, 0, &D.q);
        getHP = (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(D.d, "vkGetMemoryHostPointerPropertiesEXT");
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(D.pd, &mp);
        uint32_t mt = UINT32_MAX;
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.pNext = relay ? nullptr : &ebi; bci.size = SZ; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CK(vkCreateBuffer(D.d, &bci, nullptr, &D.b));
        VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT}; imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT; imp.pHostPointer = host;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = SZ;
        if (relay) {
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(D.d, D.b, &mr);
            for (uint32_t i = 0; i < mp.memoryTypeCount && mt == UINT32_MAX; ++i) if ((mr.memoryTypeBits >> i & 1) && mp.memoryTypes[i].propertyFlags == want) mt = i;
            if (mt == UINT32_MAX) { fprintf(stderr, "no memory type with flags 0x%x\n", want); return 1; }
        } else {
            getHP = (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(D.d, "vkGetMemoryHostPointerPropertiesEXT");
            VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            CK(getHP(D.d, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp));
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((hpp.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { if (mt == UINT32_MAX || !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) mt = i; }
            mai.pNext = &imp;
        }
        mai.memoryTypeIndex = mt;
        CK(vkAllocateMemory(D.d, &mai, nullptr, &D.m)); CK(vkBindBufferMemory(D.d, D.b, D.m, 0));
        if (relay) CK(vkMapMemory(D.d, D.m, 0, SZ, 0, &D.map));
        printf("device memory type %u flags 0x%x\n", mt, mp.memoryTypes[mt].propertyFlags);
        VkDescriptorSetLayoutBinding bd{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 1; dli.pBindings = &bd; CK(vkCreateDescriptorSetLayout(D.d, &dli, nullptr, &D.dsl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &D.dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        CK(vkCreatePipelineLayout(D.d, &pli, nullptr, &D.lay));
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = spv.size(); smi.pCode = (const uint32_t *) spv.data();
        VkShaderModule sm; CK(vkCreateShaderModule(D.d, &smi, nullptr, &sm));
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}; cpi.layout = D.lay;
        CK(vkCreateComputePipelines(D.d, VK_NULL_HANDLE, 1, &cpi, nullptr, &D.pl));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
        CK(vkCreateDescriptorPool(D.d, &dpi, nullptr, &D.dp));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = D.dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &D.dsl; CK(vkAllocateDescriptorSets(D.d, &dai, &D.ds));
        VkDescriptorBufferInfo dbi{D.b, 0, SZ}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = D.ds; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(D.d, 1, &w, 0, nullptr);
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qf; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; CK(vkCreateCommandPool(D.d, &cpci, nullptr, &D.cp));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = D.cp; cai.commandBufferCount = 1; CK(vkAllocateCommandBuffers(D.d, &cai, &D.cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; CK(vkCreateFence(D.d, &fci, nullptr, &D.f));
    }
    if (evmode) {
        auto wbm = (PFN_vkCmdWriteBufferMarkerAMD) vkGetDeviceProcAddr(ds[0].d, "vkCmdWriteBufferMarkerAMD");
        std::vector<std::vector<VkEvent>> ev(2, std::vector<VkEvent>(iters + 1));
        for (int r = 0; r < 2; ++r) for (uint32_t s = 0; s <= iters; ++s) { VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO}; CK(vkCreateEvent(ds[r].d, &eci, nullptr, &ev[r][s])); }
        for (int run = 0; run < 3; ++run) {
            memset(host, 0, SZ);
            for (int r = 0; r < 2; ++r) for (uint32_t s = 0; s <= iters; ++s) CK(vkResetEvent(ds[r].d, ev[r][s]));
            for (int r = 0; r < 2; ++r) {
                Dev & D = ds[r];
                CK(vkResetCommandBuffer(D.cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(D.cb, &bi));
                vkCmdBindPipeline(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pl);
                vkCmdBindDescriptorSets(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.lay, 0, 1, &D.ds, 0, nullptr);
                for (uint32_t st = 1; st <= iters; ++st) {
                    const uint32_t mode = getenv("XS_MODE") ? atoi(getenv("XS_MODE")) : 0, ngrp = std::max<uint32_t>(1, np / 64);
                    uint32_t pc[4] = {(uint32_t) r, st, np, mode}; vkCmdPushConstants(D.cb, D.lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                    vkCmdDispatch(D.cb, ngrp, 1, 1);
                    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(D.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    wbm(D.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, D.b, (56 + r) * 4, st);
                    VkMemoryBarrier wb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; wb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT; wb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
                    vkCmdWaitEvents(D.cb, 1, &ev[r][st], VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &wb, 0, nullptr, 0, nullptr);
                }
                CK(vkEndCommandBuffer(D.cb));
            }
            volatile uint32_t * mk = (volatile uint32_t *) host + 56;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 2; ++r) { VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &ds[r].cb; CK(vkResetFences(ds[r].d, 1, &ds[r].f)); CK(vkQueueSubmit(ds[r].q, 1, &si, ds[r].f)); }
            bool timeout = false;
            for (uint32_t st = 1; st <= iters && !timeout; ++st) {
                const auto tw = std::chrono::steady_clock::now();
                while (mk[0] < st || mk[1] < st) {
                    if (std::chrono::steady_clock::now() - tw > std::chrono::seconds(2)) { printf("relay timeout at step %u: markers %u %u\n", st, mk[0], mk[1]); timeout = true; break; }
                }
                if (timeout) break;
                if (getenv("XS_SWAP")) { CK(vkSetEvent(ds[1].d, ev[1][st])); CK(vkSetEvent(ds[0].d, ev[0][st])); } else { CK(vkSetEvent(ds[0].d, ev[0][st])); CK(vkSetEvent(ds[1].d, ev[1][st])); }
            }
            if (timeout) for (int r = 0; r < 2; ++r) for (uint32_t s = 0; s <= iters; ++s) vkSetEvent(ds[r].d, ev[r][s]);
            for (int r = 0; r < 2; ++r) CK(vkWaitForFences(ds[r].d, 1, &ds[r].f, VK_TRUE, UINT64_MAX));
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            const uint32_t * f = (const uint32_t *) host;
            printf("event run %d: %u steps, payload %u words: %.1f us per step; last A %u B %u; stale reads A %u B %u (ahead A %u B %u), payload errors %u %u\n",
                   run, iters, np, us / iters, f[0], f[16], f[40], f[41], f[50], f[51], f[44], f[45]);
        }
        return 0;
    }
    if (cycle) {
        const bool busy = getenv("XS_SLEEP") == nullptr;
        for (int run = 0; run < 3; ++run) {
            memset(host, 0, SZ);
            const auto t0 = std::chrono::steady_clock::now();
            double t_sub = 0;
            for (uint32_t st = 1; st <= iters; ++st) {
                const auto ts0 = std::chrono::steady_clock::now();
                const int only = getenv("XS_ONLY") ? atoi(getenv("XS_ONLY")) : -1;
                for (int r = 0; r < 2; ++r) {
                    if (only >= 0 && r != only) continue;
                    Dev & D = ds[r];
                    CK(vkResetCommandBuffer(D.cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; CK(vkBeginCommandBuffer(D.cb, &bi));
                    vkCmdBindPipeline(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pl);
                    vkCmdBindDescriptorSets(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.lay, 0, 1, &D.ds, 0, nullptr);
                    uint32_t pc[4] = {(uint32_t) r, st, np, 0}; vkCmdPushConstants(D.cb, D.lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                    vkCmdDispatch(D.cb, 1, 1, 1); CK(vkEndCommandBuffer(D.cb));
                    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &D.cb; CK(vkResetFences(D.d, 1, &D.f)); CK(vkQueueSubmit(D.q, 1, &si, D.f));
                }
                t_sub += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ts0).count();
                for (int r = 0; r < 2; ++r) {
                    if (only >= 0 && r != only) continue;
                    if (busy) { while (vkGetFenceStatus(ds[r].d, ds[r].f) == VK_NOT_READY) {} }
                    else CK(vkWaitForFences(ds[r].d, 1, &ds[r].f, VK_TRUE, UINT64_MAX));
                }
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            const uint32_t * f = (const uint32_t *) host;
            printf("cycle run %d (%s wait): %u steps, payload %u words: %.1f us per step (record+submit %.1f); last A %u B %u; flag errors A %u B %u, payload errors %u %u\n",
                   run, busy ? "busy" : "sleep", iters, np, us / iters, t_sub / iters, f[0], f[16], f[40], f[41], f[44], f[45]);
        }
        return 0;
    }
    for (int run = 0; run < 2; ++run) {
        memset(host, 0, 4096);
        if (relay) for (auto & D : ds) memset(D.map, 0, SZ);
        for (int r = 0; r < 2; ++r) {
            Dev & D = ds[r];
            CK(vkResetCommandBuffer(D.cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(D.cb, &bi));
            vkCmdBindPipeline(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pl);
            vkCmdBindDescriptorSets(D.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.lay, 0, 1, &D.ds, 0, nullptr);
            uint32_t pc[4] = {(uint32_t) r, iters, np, 2000000u}; vkCmdPushConstants(D.cb, D.lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
            vkCmdDispatch(D.cb, 1, 1, 1); CK(vkEndCommandBuffer(D.cb));
        }
        std::atomic<bool> stop{false};
        std::thread th;
        if (relay) th = std::thread([&] {
            volatile uint32_t * a = (volatile uint32_t *) ds[0].map; volatile uint32_t * b = (volatile uint32_t *) ds[1].map;
            uint32_t last_post = 0, last_echo = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const uint32_t pa = a[0];
                if (pa != last_post) { if (np) memcpy((void *) (b + 64), (const void *) (a + 64), np * 4); std::atomic_thread_fence(std::memory_order_seq_cst); b[0] = pa; last_post = pa; }
                const uint32_t eb = b[16];
                if (eb != last_echo) { a[16] = eb; last_echo = eb; }
            }
        });
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 1; r >= 0; --r) { VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &ds[r].cb; CK(vkResetFences(ds[r].d, 1, &ds[r].f)); CK(vkQueueSubmit(ds[r].q, 1, &si, ds[r].f)); }
        for (int r = 0; r < 2; ++r) CK(vkWaitForFences(ds[r].d, 1, &ds[r].f, VK_TRUE, UINT64_MAX));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (relay) { stop = true; th.join(); }
        const uint32_t * f = relay ? (const uint32_t *) ds[0].map : (const uint32_t *) host;
        if (relay) printf("  B side: echo %u done %u payload errors %u\n", ((uint32_t *) ds[1].map)[16], ((uint32_t *) ds[1].map)[33], ((uint32_t *) ds[1].map)[48]);
        printf("run %d: %u round trips, payload %u words: %.2f us per round trip (A %u, B %u, timeouts A %u B %u, payload errors %u)\n",
               run, iters, np, us / iters, f[0], f[16], f[32], f[33], f[48]);
    }
    return 0;
}
