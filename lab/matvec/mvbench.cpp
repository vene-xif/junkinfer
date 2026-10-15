// mvbench: IQ4_NL matvec bandwidth on one Vulkan device with a GCN-shaped layout (mv_iq4.comp).
// For each (M, K, REP): correctness vs a CPU reference, then time `n` back-to-back matvecs over
// distinct matrices (total >= 256 MB so L2 does not help), with and without a barrier between them.
// build: g++ -O2 -std=c++17 mvbench.cpp -o mvbench -lvulkan ; glslc --target-env=vulkan1.2 -O mv_iq4.comp -o mv_iq4.spv
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static VkDevice dev; static VkPhysicalDevice pdev; static VkPhysicalDeviceMemoryProperties mp;
struct Buf { VkBuffer b; VkDeviceMemory m; void * p = nullptr; size_t size; };

static Buf make_buf(size_t size, VkMemoryPropertyFlags want) {
    Buf B; B.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK(vkCreateBuffer(dev, &bi, nullptr, &B.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, B.b, &mr);
    int type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { type = (int) i; break; }
    if (type < 0) { fprintf(stderr, "no memory type\n"); exit(1); }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = type;
    VK(vkAllocateMemory(dev, &ai, nullptr, &B.m)); VK(vkBindBufferMemory(dev, B.b, B.m, 0));
    if (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK(vkMapMemory(dev, B.m, 0, size, 0, &B.p));
    return B;
}

static const float kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

int main(int argc, char ** argv) {
    const char * spv_path = argc > 1 ? argv[1] : "mv_iq4.spv";
    const bool native = argc > 2 && strcmp(argv[2], "native") == 0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 1; { VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pdev); if (r != VK_SUCCESS && r != VK_INCOMPLETE) { fprintf(stderr, "enumerate failed\n"); return 1; } }
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pdev, &pp);
    float prio = 1; VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex = 0; q.queueCount = 1; q.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
    VK(vkCreateDevice(pdev, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, 0, 0, &queue);

    std::ifstream f(spv_path, std::ios::binary); std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t *) code.data();
    VkShaderModule sm; VK(vkCreateShaderModule(dev, &smi, nullptr, &sm));

    VkDescriptorSetLayoutBinding binds[4];
    for (int i = 0; i < 4; ++i) binds[i] = {(uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 4; dli.pBindings = binds;
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));

    const size_t TOTAL_Q = size_t(320) << 20; // bytes of nibbles across all matrices
    Buf wq = make_buf(TOTAL_Q, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT); // BAR: fill from host
    Buf ws = make_buf(TOTAL_Q / 16 * 4, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    Buf xb = make_buf(4096 * 4, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf yb = make_buf(size_t(64) << 20, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::mt19937 rng(1);
    { std::vector<uint32_t> tmp(TOTAL_Q / 4); for (auto & v : tmp) v = rng(); memcpy(wq.p, tmp.data(), TOTAL_Q); }
    { std::vector<float> tmp(TOTAL_Q / 16); for (auto & v : tmp) v = (rng() % 1000) / 1000.0f * 0.01f; memcpy(ws.p, tmp.data(), tmp.size() * 4); }
    { float * x = (float *) xb.p; for (int i = 0; i < 4096; ++i) x[i] = ((int) (rng() % 2001) - 1000) / 1000.0f; }
    // native GGUF layout: per block fp16(d) + 16 nibble bytes; scales rounded to fp16 in the fp32 array too,
    // so the CPU check (which reads ws) matches the GPU exactly
    Buf wn = make_buf(native ? TOTAL_Q / 16 * 18 : 16, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (native) {
        std::vector<uint8_t> tmp(TOTAL_Q / 16 * 18);
        const uint8_t * Q = (const uint8_t *) wq.p; float * S = (float *) ws.p;
        for (size_t blk = 0; blk < TOTAL_Q / 16; ++blk) {
            // fp32 -> fp16 (normal range only, round toward zero)
            uint32_t fb; memcpy(&fb, &S[blk], 4);
            const uint32_t sign = (fb >> 16) & 0x8000u; int e = (int) ((fb >> 23) & 0xff) - 127 + 15; uint32_t m = (fb >> 13) & 0x3ffu;
            uint16_t h = (uint16_t) (e <= 0 ? sign : (sign | (uint32_t(e) << 10) | m));
            uint32_t back = e <= 0 ? 0 : ((sign << 16) | (uint32_t(e - 15 + 127) << 23) | (m << 13)); memcpy(&S[blk], &back, 4);
            memcpy(&tmp[blk * 18], &h, 2); memcpy(&tmp[blk * 18 + 2], Q + blk * 16, 16);
        }
        memcpy(wn.p, tmp.data(), tmp.size());
    }

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi[4] = {{native ? wn.b : wq.b, 0, VK_WHOLE_SIZE}, {ws.b, 0, VK_WHOLE_SIZE}, {xb.b, 0, VK_WHOLE_SIZE}, {yb.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[4];
    for (int i = 0; i < 4; ++i) { wds[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
    vkUpdateDescriptorSets(dev, 4, wds, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = 0; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; VK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1;
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VkQueryPool qp; VK(vkCreateQueryPool(dev, &qpi, nullptr, &qp));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK(vkCreateFence(dev, &fci, nullptr, &fence));

    struct Shape { uint32_t M, K; const char * what; };
    const Shape shapes[] = {{8192, 2048, "attn_qkv / q"}, {4096, 2048, "attn_gate | 8 experts gate"}, {2048, 4096, "ssm_out / attn_out"},
                            {512, 2048, "shared/expert gate|up"}, {2048, 512, "shared/expert down"}};
    printf("device %s; GB/s counts nibbles + fp32 scales (0.625 B/weight); IQ4_NL as stored in GGUF is 0.5625 B/weight\n", pp.deviceName);
    for (const Shape & s : shapes) {
        const uint32_t nchunk = s.K / 32;
        const uint32_t L = nchunk >= 64 ? 64 : nchunk;
        const uint32_t CPL = nchunk / L;
        const size_t bytes_q = size_t(s.M) * s.K / 2, bytes_s = size_t(s.M) * nchunk * 4;
        const uint32_t nmat = (uint32_t) std::min<size_t>(TOTAL_Q / bytes_q, 4096);
        for (uint32_t REP : {1u, 2u, 4u, 8u}) {
            const uint32_t P = 64 / L;
            if (s.M % (REP * P)) continue;
            uint32_t spec[4] = {L, CPL, REP, s.K};
            VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
            VkSpecializationInfo si{native ? 4u : 3u, me, native ? 16u : 12u, spec};
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.stage.pSpecializationInfo = &si;
            cpci.layout = pl;
            VkPipeline pipe; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
            const uint32_t groups = s.M / (REP * P);
            for (int barrier = 0; barrier < 2; ++barrier) {
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VK(vkBeginCommandBuffer(cb, &bi));
                vkCmdResetQueryPool(cb, qp, 0, 2);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                for (uint32_t i = 0; i < nmat; ++i) {
                    uint32_t pc[4] = {s.M, s.K, (uint32_t) (i * (bytes_q / 16)), (uint32_t) (i * (bytes_s / 4))};
                    if (native) { pc[2] = (uint32_t) (i * (size_t(s.M) * nchunk * 18 / 16)); }
                    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                    vkCmdDispatch(cb, groups, 1, 1);
                    if (barrier) {
                        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    }
                }
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                VK(vkEndCommandBuffer(cb));
                VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO}; sub.commandBufferCount = 1; sub.pCommandBuffers = &cb;
                VK(vkQueueSubmit(queue, 1, &sub, fence)); VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX)); VK(vkResetFences(dev, 1, &fence));
                uint64_t ts[2]; VK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                const double us = (ts[1] - ts[0]) * pp.limits.timestampPeriod / 1000.0;
                const double per = us / nmat;
                if (barrier && REP == 1) {
                    // correctness: the last matrix's output vs CPU
                    const uint32_t i = nmat - 1; const uint8_t * Q = (const uint8_t *) wq.p + size_t(i) * bytes_q; const float * S = (const float *) ws.p + size_t(i) * bytes_s / 4;
                    const float * x = (const float *) xb.p; const float * y = (const float *) yb.p; double maxrel = 0;
                    for (uint32_t r = 0; r < s.M; r += 97) {
                        double acc = 0;
                        for (uint32_t c = 0; c < nchunk; ++c) {
                            const uint8_t * qb = Q + (size_t(r) * nchunk + c) * 16; double sc = 0;
                            for (int b = 0; b < 16; ++b) sc += kv[qb[b] & 15] * x[c * 32 + b] + kv[qb[b] >> 4] * x[c * 32 + 16 + b];
                            acc += S[size_t(r) * nchunk + c] * sc;
                        }
                        maxrel = std::max(maxrel, std::fabs(acc - y[r]) / (std::fabs(acc) + 1e-3));
                    }
                    if (!getenv("MV_NOCHECK")) printf("  check %ux%u: max rel err %.2e %s\n", s.M, s.K, maxrel, maxrel < 1e-3 ? "OK" : "MISMATCH");
                }
                printf("%-28s M=%5u K=%4u REP=%u %s: %7.2f us/matvec  %6.1f GB/s  (%u matrices)\n", s.what, s.M, s.K, REP,
                       barrier ? "barrier " : "nobarrier", per, (native ? double(s.M) * nchunk * 18 : double(bytes_q + bytes_s)) / (per * 1e3), nmat);
            }
            vkDestroyPipeline(dev, pipe, nullptr);
        }
    }
    return 0;
}
