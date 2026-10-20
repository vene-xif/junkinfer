// cost of a dependent dispatch chain: N tiny dispatches (G workgroups) with a compute->compute barrier between each
// usage: chain DEV QF N G
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, (int) r_, __LINE__); exit(1); } } while (0)
int main(int argc, char ** argv) {
    const uint32_t dev = atoi(argv[1]), qf = atoi(argv[2]), N = atoi(argv[3]), G = atoi(argv[4]);
    std::ifstream fs("ch.spv", std::ios::binary); std::vector<char> spv((std::istreambuf_iterator<char>(fs)), {});
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice d; CK(vkCreateDevice(pds[dev], &dci, nullptr, &d)); VkQueue q; vkGetDeviceQueue(d, qf, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pds[dev], &mp);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = 1 << 20; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer b; CK(vkCreateBuffer(d, &bci, nullptr, &b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(d, b, &mr);
    uint32_t mt = 0; for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt; VkDeviceMemory m; CK(vkAllocateMemory(d, &mai, nullptr, &m)); CK(vkBindBufferMemory(d, b, m, 0));
    VkDescriptorSetLayoutBinding bd{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 1; dli.pBindings = &bd; VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(d, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr; VkPipelineLayout lay; CK(vkCreatePipelineLayout(d, &pli, nullptr, &lay));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = spv.size(); smi.pCode = (const uint32_t *) spv.data(); VkShaderModule sm; CK(vkCreateShaderModule(d, &smi, nullptr, &sm));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"}; cpi.layout = lay; VkPipeline pl; CK(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpi, nullptr, &pl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps; VkDescriptorPool dp; CK(vkCreateDescriptorPool(d, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(d, &dai, &ds));
    VkDescriptorBufferInfo dbi{b, 0, VK_WHOLE_SIZE}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = ds; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi; vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qf; VkCommandPool cp; CK(vkCreateCommandPool(d, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(d, &cai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; CK(vkCreateFence(d, &fci, nullptr, &f));
    for (int bar = 0; bar < 2; ++bar) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkResetCommandBuffer(cb, 0)); CK(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, lay, 0, 1, &ds, 0, nullptr);
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t pc[2] = {i, G * 64}; vkCmdPushConstants(cb, lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
            vkCmdDispatch(cb, G, 1, 1);
            if (bar) { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                       vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr); }
        }
        CK(vkEndCommandBuffer(cb));
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; CK(vkResetFences(d, 1, &f));
            auto t0 = std::chrono::steady_clock::now(); CK(vkQueueSubmit(q, 1, &si, f)); CK(vkWaitForFences(d, 1, &f, VK_TRUE, UINT64_MAX));
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("dev %u qf %u: %u dispatches x %u groups, %s: %.2f us per dispatch\n", dev, qf, N, G, bar ? "barrier after each" : "no barriers", best / N);
    }
    return 0;
}
