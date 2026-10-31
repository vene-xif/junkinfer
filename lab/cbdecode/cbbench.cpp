// cbbench: EXL3 codebook decode variants (cb.comp) - bit-exact check vs mul1_ref, gfx803 ISA (VK_AMD_shader_info),
// throughput. build: g++ -O2 -std=c++17 cbbench.cpp -o cbbench -lvulkan ; glslc --target-env=vulkan1.3 -O cb.comp -o cb.spv
// run: ./cbbench cb.spv [device index]   (writes isa_<var>.txt)
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)
static const char * NAMES[] = {"mcg", "mul1_ref", "mul1_dot", "mul1_dotacc", "mul1_ubyte", "mul1_fold", "mul1_nof16", "mul1_mad", "mul1_raw", "mcg_u24", "mul1_raw_u24", "mul1_mad_u24", "mul1_int", "mul1_ctr", "mul1_ctrf"};
int main(int argc, char ** argv) {
    const char * spv = argc > 1 ? argv[1] : "cb.spv";
    const int devi = argc > 2 ? atoi(argv[2]) : 0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr); std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    VkPhysicalDevice pd = pds[devi]; VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
    printf("device %d: %s\n", devi, pp.deviceName);
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("shaderFloat16 %d shaderIntegerDotProduct %d\n", f12.shaderFloat16, f13.shaderIntegerDotProduct);
    float prio = 1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueCount = 1; qi.pQueuePriorities = &prio; qi.queueFamilyIndex = 0;
    const char * ext[] = {"VK_AMD_shader_info"};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &f2; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> eps(ne); vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, eps.data());
    bool has_info = false; for (auto & e : eps) has_info = has_info || strcmp(e.extensionName, "VK_AMD_shader_info") == 0;
    di.enabledExtensionCount = has_info ? 1 : 0; di.ppEnabledExtensionNames = ext;
    // without VK_AMD_shader_info: RADV_DEBUG=shaders prints each pipeline's ISA (stderr); markers below on stdout+stderr
    VkDevice dev; VK(vkCreateDevice(pd, &di, nullptr, &dev)); VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
    auto getinfo = (PFN_vkGetShaderInfoAMD) nullptr;
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const size_t N = 1u << 22;   // floats
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = N * 4; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buf; VK(vkCreateBuffer(dev, &bi, nullptr, &buf)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    int mt = -1; for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = (int) i; break; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = mt;
    VkDeviceMemory mem; VK(vkAllocateMemory(dev, &ai, nullptr, &mem)); VK(vkBindBufferMemory(dev, buf, mem, 0)); float * host; VK(vkMapMemory(dev, mem, 0, N * 4, 0, (void **) &host));
    std::ifstream f(spv, std::ios::binary); std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t *) code.data();
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutBinding lb{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 1; dli.pBindings = &lb; VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp)); VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dai, &ds)); VkDescriptorBufferInfo dbi{buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = ds; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi; vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; VkCommandPool cp; VK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1; VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2; VkQueryPool qp; VK(vkCreateQueryPool(dev, &qpi, nullptr, &qp));
    auto run = [&](VkPipeline p, uint32_t groups, uint32_t iters, double * us) {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkResetCommandBuffer(cb, 0); vkBeginCommandBuffer(cb, &cbi);
        vkCmdResetQueryPool(cb, qp, 0, 2); vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0); vkCmdDispatch(cb, groups, 1, 1); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vkEndCommandBuffer(cb); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); VK(vkQueueWaitIdle(q));
        uint64_t ts[2]; vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (us) *us = (ts[1] - ts[0]) * pp.limits.timestampPeriod / 1000.0;
    };
    auto make = [&](uint32_t var, uint32_t mode, bool info) {
        uint32_t sc[2] = {var, mode}; VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}}; VkSpecializationInfo si{2, me, 8, sc};
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sm; ci.stage.pName = "main"; ci.stage.pSpecializationInfo = &si; ci.layout = pl;
        VkPipeline p; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        if (info && getinfo) {
            size_t sz = 0; getinfo(dev, p, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &sz, nullptr);
            std::vector<char> t(sz); getinfo(dev, p, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &sz, t.data());
            char fn[64]; snprintf(fn, sizeof fn, "isa_%s.txt", NAMES[var]); FILE * fo = fopen(fn, "w"); fwrite(t.data(), 1, sz, fo); fclose(fo);
        }
        return p;
    };
    // reference: mul1_ref over all 16-bit inputs
    std::vector<float> ref(65536);
    { VkPipeline p = make(1, 0, false); run(p, 256, 0, nullptr); memcpy(ref.data(), host, 65536 * 4); }
    std::vector<float> refm(65536);
    { VkPipeline p = make(0, 0, false); run(p, 256, 0, nullptr); memcpy(refm.data(), host, 65536 * 4); }
    const uint32_t groups = 36 * 64, iters = 512;   // 36 CUs x 64 groups x 256 threads x 512 iters x 8 decodes
    const double ndec = (double) groups * 256 * iters * 8;
    for (uint32_t var = 0; var < 15; ++var) {
        VkPipeline pc = make(var, 0, false); run(pc, 256, 0, nullptr);
        int bad = 0; double maxd = 0;
        const std::vector<float> & R = (var == 0 || var == 9) ? refm : ref;
        for (int i = 0; i < 65536; ++i) { if (memcmp(&host[i], &R[i], 4) != 0) { bad++; maxd = std::max(maxd, (double) fabsf(host[i] - R[i])); } }
        fprintf(stderr, "\n##### ISA %s #####\n", NAMES[var]); fflush(stderr);
        VkPipeline pb = make(var, 1, true); double us = 0, best = 1e30;
        for (int r = 0; r < 5; ++r) { run(pb, groups, iters, &us); best = std::min(best, us); }
        printf("%-12s %s mismatches vs its reference: %6d (max |d| %.3g) | %.1f Gdecode/s (%.0f us)\n", NAMES[var], var == 0 ? "(other codebook)" : "                ", bad, maxd, ndec / best / 1e3, best);
    }
    return 0;
}
