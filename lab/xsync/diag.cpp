// one device, host-visible uncached buffer: how fast do GPU writes reach the CPU, and CPU writes reach a polling GPU?
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, (int) r_, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const uint32_t dev = argc > 1 ? atoi(argv[1]) : 1, want = argc > 2 ? strtoul(argv[2], nullptr, 0) : 0x6;
    std::ifstream fs("dg.spv", std::ios::binary); std::vector<char> spv((std::istreambuf_iterator<char>(fs)), {});
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = pds[dev];
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.vulkanMemoryModel = VK_TRUE; f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &f12; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice d; CK(vkCreateDevice(pd, &dci, nullptr, &d)); VkQueue q; vkGetDeviceQueue(d, 0, 0, &q);
    const size_t SZ = 1 << 20;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = SZ; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer b; CK(vkCreateBuffer(d, &bci, nullptr, &b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(d, b, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) { printf("type %u flags 0x%x heap %u\n", i, mp.memoryTypes[i].propertyFlags, mp.memoryTypes[i].heapIndex); if (mt == UINT32_MAX && (mr.memoryTypeBits >> i & 1) && mp.memoryTypes[i].propertyFlags == want) mt = i; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = SZ; mai.memoryTypeIndex = mt;
    VkDeviceMemory m; CK(vkAllocateMemory(d, &mai, nullptr, &m)); CK(vkBindBufferMemory(d, b, m, 0));
    volatile uint32_t * map; CK(vkMapMemory(d, m, 0, SZ, 0, (void **) &map));
    VkDescriptorSetLayoutBinding bd{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 1; dli.pBindings = &bd; VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(d, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout lay; CK(vkCreatePipelineLayout(d, &pli, nullptr, &lay));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = spv.size(); smi.pCode = (const uint32_t *) spv.data();
    VkShaderModule sm; CK(vkCreateShaderModule(d, &smi, nullptr, &sm));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}; cpi.layout = lay;
    VkPipeline pl; CK(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpi, nullptr, &pl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(d, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(d, &dai, &ds));
    VkDescriptorBufferInfo dbi{b, 0, SZ}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = ds; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; VkCommandPool cp; CK(vkCreateCommandPool(d, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(d, &cai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; CK(vkCreateFence(d, &fci, nullptr, &f));
    auto launch = [&](uint32_t role, uint32_t iters) {
        CK(vkResetCommandBuffer(cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, lay, 0, 1, &ds, 0, nullptr);
        uint32_t pc[4] = {role, iters, 0, 200000000u}; vkCmdPushConstants(cb, lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdDispatch(cb, 1, 1, 1); CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; CK(vkResetFences(d, 1, &f)); CK(vkQueueSubmit(q, 1, &si, f));
    };
    printf("memory type %u (flags 0x%x)\n", mt, want);
    // write visibility: sample the GPU counter
    for (int k = 0; k < 64; ++k) map[k] = 0;
    const double t0 = now(); launch(0, 20000000u);
    uint32_t last = 0; int changes = 0; double first = 0, tl = 0;
    while (vkGetFenceStatus(d, f) == VK_NOT_READY) { const uint32_t v = map[0]; if (v != last) { if (!changes) first = now() - t0; ++changes; last = v; tl = now() - t0; } }
    printf("write: CPU saw %d distinct values, first after %.0f us, last %u at %.0f us, kernel done at %.0f us\n", changes, first, last, tl, now() - t0);
    // read visibility: GPU polls flag[16]; CPU writes it after 20 ms
    for (int k = 0; k < 64; ++k) map[k] = 0;
    launch(1, 12345u);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const double tw = now(); map[16] = 12345u;
    CK(vkWaitForFences(d, 1, &f, VK_TRUE, UINT64_MAX));
    printf("read: GPU saw the CPU write after %.0f us (fence), %u polls\n", now() - tw, map[1]);
    return 0;
}
