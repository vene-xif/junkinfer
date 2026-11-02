// mmbench: the JDE prefill routed-expert GEMM (engine src/junkinfer/shaders/pf_mm.comp, MMID + X3F) on synthetic
// EXL3 4-bit experts with a realistic routing: T tokens x 8 experts over NE experts (rank 0's share), tiles as
// jde_prefill builds them (64-row tiles; an expert's remainder of <= 32 rows in a 32-row tile of the "half" kernel).
// Per variant: VGPRs / LDS / waves (VK_AMD_shader_info), max |d| vs the first (reference) variant, time per batch.
// build: g++ -O2 -std=c++17 mmbench.cpp -o mmbench -lvulkan ; shaders: see build.sh
// run:   ./mmbench <variant> [<variant> ...]  variant = spv:maxrows[,spv:maxrows...] - each expert's rows are cut into
//        tiles of the largest kernel's rows; a remainder goes to the smallest kernel that holds it (first variant = reference)
//        env: MB_DEV (device, 0), MB_K (2048), MB_M (512 rows out), MB_NE (200 experts), MB_T ("150,500,1000,2000,4000")
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static int envi(const char * n, int d) { const char * v = getenv(n); return v ? atoi(v) : d; }

struct Buf { VkBuffer b; VkDeviceMemory m; void * map; VkDeviceSize size; };

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mmbench spv:rows[,spv:rows...] [...]\n"); return 1; }
    const int devi = envi("MB_DEV", 0);
    const uint32_t K = envi("MB_K", 2048), M = envi("MB_M", 512), NE = envi("MB_NE", 200), NU = 8;
    std::vector<uint32_t> Ts;
    { std::string s = getenv("MB_T") ? getenv("MB_T") : "150,500,1000,2000,4000"; size_t p = 0;
      while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); Ts.push_back((uint32_t) atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    const uint32_t TMAX = *std::max_element(Ts.begin(), Ts.end());

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr); std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    VkPhysicalDevice pd = pds[devi]; VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
    printf("device %d: %s | K %u, M %u, %u experts, %u per token\n", devi, pp.deviceName, K, M, NE, NU);
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    float prio = 1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueCount = 1; qi.pQueuePriorities = &prio; qi.queueFamilyIndex = 0;
    uint32_t nx = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &nx, nullptr); std::vector<VkExtensionProperties> eps(nx); vkEnumerateDeviceExtensionProperties(pd, nullptr, &nx, eps.data());
    bool has_info = false; for (auto & e : eps) has_info = has_info || strcmp(e.extensionName, "VK_AMD_shader_info") == 0;
    const char * ext[] = {"VK_AMD_shader_info"};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &f2; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = has_info ? 1 : 0; di.ppEnabledExtensionNames = ext;
    VkDevice dev; VK(vkCreateDevice(pd, &di, nullptr, &dev)); VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
    auto getinfo = has_info ? (PFN_vkGetShaderInfoAMD) vkGetDeviceProcAddr(dev, "vkGetShaderInfoAMD") : nullptr;
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    auto mkbuf = [&](VkDeviceSize size, bool host) {
        Buf r{}; r.size = size;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK(vkCreateBuffer(dev, &bi, nullptr, &r.b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
        const VkMemoryPropertyFlags want = host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        int mt = -1;
        // host buffers: prefer device-local + host-visible (BAR) is not needed - plain host memory, read by the GPU only for small tables
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = (int) i; break; }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = mt;
        VK(vkAllocateMemory(dev, &ai, nullptr, &r.m)); VK(vkBindBufferMemory(dev, r.b, r.m, 0));
        if (host) VK(vkMapMemory(dev, r.m, 0, size, 0, &r.map));
        return r;
    };
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; VkCommandPool cp; VK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1; VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cai, &cb));
    auto upload = [&](Buf & dst, const void * src, VkDeviceSize n) {   // staging copy into a device-local buffer
        Buf st = mkbuf(n, true); memcpy(st.map, src, n);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkResetCommandBuffer(cb, 0); vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy c{0, 0, n}; vkCmdCopyBuffer(cb, st.b, dst.b, 1, &c); vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); VK(vkQueueWaitIdle(q));
        vkDestroyBuffer(dev, st.b, nullptr); vkFreeMemory(dev, st.m, nullptr);
    };

    // ---- data: weights (random trellis words = valid codes), activations, svh = 1 (f16 pairs), row maps
    std::mt19937 rng(1234);
    const uint32_t TW = 32;   // words per 16x16 tile at 4 bits
    const size_t wwords = (size_t) NE * (K / 16) * (M / 16) * TW;
    std::vector<uint32_t> W(wwords); for (auto & w : W) w = rng();
    Buf bA = mkbuf(wwords * 4, false); upload(bA, W.data(), wwords * 4); W.clear(); W.shrink_to_fit();
    const size_t bfl = (size_t) TMAX * NU * K;
    Buf bB = mkbuf(bfl * 4, false);
    { std::vector<float> B(bfl); std::normal_distribution<float> nd(0.f, 1.f); for (auto & x : B) x = nd(rng); upload(bB, B.data(), bfl * 4); }
    const size_t dfl = (size_t) TMAX * NU * M;
    Buf bD = mkbuf(dfl * 4, false), bDh = mkbuf(dfl * 4, true);
    const uint32_t sv_estride = K + M;   // per expert: suh (K) then svh (M) - only svh is read
    std::vector<uint32_t> SV(((size_t) NE * sv_estride + 1) / 2 + 16, 0x3c003c00u);   // f16 1.0 pairs
    Buf bSV = mkbuf(SV.size() * 4, false); upload(bSV, SV.data(), SV.size() * 4);
    Buf bWT = mkbuf(1024, false);
    const uint32_t maxt = TMAX * NU / 32 + NE + 8;
    const size_t rmw = 1 + 3 * (size_t) maxt + (size_t) maxt * 64;
    const int KMAX = 4;
    std::vector<Buf> bR; for (int i = 0; i < KMAX; ++i) bR.push_back(mkbuf(rmw * 4, false));

    // ---- pipelines: 7 storage buffers (A, B, D, IDS, RowMap, SV, WT), push constants as jde_pf_gemm_id
    VkDescriptorSetLayoutBinding lb[7]; for (uint32_t i = 0; i < 7; ++i) lb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 7; dli.pBindings = lb; VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    struct PC { uint32_t M, N, K, stride_a, stride_b, stride_d, bsa, bsb, bsd, nei0, nei1, nbi1, ne11, n_experts, row_ids_offset,
                sv_estride, sv_f16, w_on, w_tok, n0, yo, kt0, KTF, accum, sv_base; };
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * 4}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 4; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSet ds[4];
    for (int h = 0; h < KMAX; ++h) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
        VK(vkAllocateDescriptorSets(dev, &dai, &ds[h]));
        const Buf & rm = bR[h];
        VkBuffer bs[7] = {bA.b, bB.b, bD.b, rm.b, rm.b, bSV.b, rm.b};
        VkDescriptorBufferInfo dbi[7]; VkWriteDescriptorSet w[7];
        for (int i = 0; i < 7; ++i) {
            dbi[i] = {bs[i], 0, VK_WHOLE_SIZE};
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = ds[h]; w[i].dstBinding = (uint32_t) i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, 7, w, 0, nullptr);
    }
    const uint32_t cbsel = (uint32_t) envi("MB_CB", 1);   // codebook spec constant: 1 = mul1 (Sage)
    auto make = [&](const char * spv, std::string & stats) {
        std::ifstream f(spv, std::ios::binary); std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
        if (code.empty()) { fprintf(stderr, "cannot read %s\n", spv); exit(1); }
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t *) code.data();
        VkShaderModule sm; VK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
        VkSpecializationMapEntry me{0, 0, 4}; VkSpecializationInfo si{1, &me, 4, &cbsel};
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sm; ci.stage.pName = "main"; ci.stage.pSpecializationInfo = &si; ci.layout = pl;
        VkPipeline p; VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        if (getinfo) {
            VkShaderStatisticsInfoAMD st{}; size_t sz = sizeof(st);
            if (getinfo(dev, p, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_STATISTICS_AMD, &sz, &st) == VK_SUCCESS) {
                char b[160]; snprintf(b, sizeof b, "vgpr %u sgpr %u lds %zu B", st.resourceUsage.numUsedVgprs, st.resourceUsage.numUsedSgprs, st.resourceUsage.ldsUsageSizeInBytes);
                stats = b;
            }
        }
        return p;
    };

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2; VkQueryPool qp; VK(vkCreateQueryPool(dev, &qpi, nullptr, &qp));

    // routing per T: token t picks NU distinct experts (skewed: a few hot experts, as real routing); the same routing
    // for every variant (rows per expert kept), tiles per variant's kernel set
    struct Plan { uint32_t T; uint64_t rows; std::vector<uint32_t> nt; };
    std::vector<std::vector<uint32_t>> rows;
    auto route = [&](uint32_t T) {
        rows.assign(NE, {});
        std::vector<double> wexp(NE); for (uint32_t e = 0; e < NE; ++e) wexp[e] = 1.0 / std::pow(1.0 + e, 0.6);
        std::discrete_distribution<uint32_t> pick(wexp.begin(), wexp.end());
        for (uint32_t t = 0; t < T; ++t) {
            uint32_t ch[8]; uint32_t n = 0;
            while (n < NU) { uint32_t e = pick(rng); bool dup = false; for (uint32_t i = 0; i < n; ++i) dup = dup || ch[i] == e; if (!dup) ch[n++] = e; }
            for (uint32_t s = 0; s < NU; ++s) rows[ch[s]].push_back(s | (t << 16));
        }
    };
    // row map per kernel, layout as pf_rowmap: [0] count, (expert, n) pairs, scale ids, rows (stride = kernel rows)
    auto build = [&](uint32_t T, const std::vector<uint32_t> & lim, Plan & P) {
        const size_t nk = lim.size();
        std::vector<std::vector<uint32_t>> R(nk, std::vector<uint32_t>(rmw, 0));
        std::vector<uint32_t> nt(nk, 0); uint64_t tot = 0;
        const uint32_t off = 1 + 3 * maxt, big = lim.back();
        for (uint32_t e = 0; e < NE; ++e) {
            const auto & r = rows[e]; tot += r.size();
            for (size_t i = 0; i < r.size(); ) {
                const size_t left = r.size() - i;
                size_t k = nk - 1;
                if (left < big) for (size_t j = 0; j < nk; ++j) if (left <= lim[j]) { k = j; break; }
                const size_t c = std::min<size_t>(lim[k], left);
                auto & M_ = R[k]; uint32_t & n = nt[k];
                M_[1 + 2 * n] = e; M_[2 + 2 * n] = (uint32_t) c; M_[1 + 2 * maxt + n] = e;
                for (size_t j = 0; j < c; ++j) M_[off + (size_t) n * lim[k] + j] = r[i + j];
                ++n; i += c;
            }
        }
        for (size_t k = 0; k < nk; ++k) { R[k][0] = nt[k]; upload(bR[k], R[k].data(), rmw * 4); }
        P = {T, tot, nt};
    };
    auto record = [&](const std::vector<VkPipeline> & ks, const Plan & P, bool ts, int reps_in_cb = 1) {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkResetCommandBuffer(cb, 0); vkBeginCommandBuffer(cb, &cbi);
        vkCmdFillBuffer(cb, bD.b, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        if (ts) { vkCmdResetQueryPool(cb, qp, 0, 2); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0); }
        for (int rr = 0; rr < reps_in_cb; ++rr)   // MB_STRESS: the batch repeated back to back (same outputs rewritten)
        for (size_t h = 0; h < ks.size(); ++h) {
            const uint32_t nt = P.nt[h];
            if (nt == 0) continue;
            PC pc{M, P.T * NU, K, K, K, M, K * M, NU * K, NU * M, NU, P.T, NU, NU, NE, 1 + 3 * maxt, sv_estride, 1, 0, NU, 0, 0, 0, K / 16, 0, K};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ks[h]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds[h], 0, nullptr);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
            vkCmdDispatch(cb, M / 128, nt, 1);   // production: (N / 128, tiles); dispatches back to back (no barrier, as in jde_prefill)
        }
        if (ts) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy c{0, 0, dfl * 4}; vkCmdCopyBuffer(cb, bD.b, bDh.b, 1, &c);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); VK(vkQueueWaitIdle(q));
        double us = 0;
        if (ts) { uint64_t t2[2]; vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(t2), t2, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT); us = (t2[1] - t2[0]) * pp.limits.timestampPeriod / 1000.0; }
        return us;
    };

    // variants: "a.spv:64,b.spv:32" (kernels sorted by rows)
    const int nv = argc - 1;
    std::vector<std::vector<VkPipeline>> PK(nv); std::vector<std::vector<uint32_t>> LIM(nv);
    for (int v = 0; v < nv; ++v) {
        std::vector<std::pair<uint32_t, std::string>> ks;
        std::string a = argv[1 + v]; size_t p = 0;
        while (p < a.size()) { size_t q2 = a.find(',', p); if (q2 == std::string::npos) q2 = a.size(); std::string it = a.substr(p, q2 - p);
            const size_t c = it.rfind(':'); ks.push_back({(uint32_t) atoi(it.substr(c + 1).c_str()), it.substr(0, c)}); p = q2 + 1; }
        std::sort(ks.begin(), ks.end());
        if ((int) ks.size() > KMAX) { fprintf(stderr, "at most %d kernels per variant\n", KMAX); return 1; }
        printf("variant %d:", v);
        for (auto & k : ks) { std::string st; PK[v].push_back(make(k.second.c_str(), st)); LIM[v].push_back(k.first); printf(" %s (<= %u rows)%s%s", k.second.c_str(), k.first, st.empty() ? "" : " ", st.c_str()); }
        printf("\n");
    }
    // MB_STRESS=<seconds> (undervolt validation, 2026-11-01): variant 0 at the largest T, a reference output first, then
    // command buffers of MB_STRESS_REP back-to-back batches until the time is up; every output is compared bit for bit
    // with the reference (the kernel is deterministic: any difference is a compute error). Exit code 2 on a mismatch.
    if (const int stress_s = envi("MB_STRESS", 0); stress_s > 0) {
        const uint32_t T = TMAX;
        route(T);
        Plan P; build(T, LIM[0], P);
        record(PK[0], P, false);
        const size_t n = (size_t) T * NU * M;
        std::vector<float> ref((const float *) bDh.map, (const float *) bDh.map + n);
        int nonfinite = 0; for (float x : ref) nonfinite += !std::isfinite(x);
        const int rep = envi("MB_STRESS_REP", 30);
        const double us1 = record(PK[0], P, true, rep);
        printf("stress: T %u, %d batches per submit (%.1f ms), reference nonfinite %d\n", T, rep, us1 / 1e3, nonfinite);
        fflush(stdout);
        if (memcmp(ref.data(), bDh.map, n * 4) != 0) { printf("STRESS MISMATCH on the first repeat\n"); return 2; }
        const auto t0 = std::chrono::steady_clock::now();
        long iters = 0, bad = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < stress_s) {
            record(PK[0], P, false, rep);
            const float * d = (const float *) bDh.map;
            long nb = 0;
            if (memcmp(d, ref.data(), n * 4) != 0) for (size_t i = 0; i < n; ++i) nb += memcmp(&d[i], &ref[i], 4) != 0;
            ++iters; bad += nb;
            if (nb) { printf("STRESS MISMATCH iter %ld: %ld values differ\n", iters, nb); fflush(stdout); }
            if (iters % 20 == 0) { printf("stress %.0f s: %ld submits, %ld bad values\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), iters, bad); fflush(stdout); }
        }
        printf("STRESS DONE: %ld submits (%ld batches), %ld bad values\n", iters, iters * rep, bad);
        return bad ? 2 : 0;
    }
    const int reps = envi("MB_REPS", 7);
    for (uint32_t T : Ts) {
        route(T);
        std::vector<float> ref;
        for (int v = 0; v < nv; ++v) {
            Plan P; build(T, LIM[v], P);
            const double fma = (double) P.rows * K * M;   // useful FMAs (one projection over rank 0's experts)
            if (v == 0) printf("T %5u: %llu rows\n", T, (unsigned long long) P.rows);
            double best = 1e30;
            for (int r = 0; r < reps; ++r) best = std::min(best, record(PK[v], P, true));
            const float * d = (const float *) bDh.map;
            double maxd = 0, maxv = 0;
            if (v == 0) ref.assign(d, d + (size_t) T * NU * M);
            else for (size_t i = 0; i < (size_t) T * NU * M; ++i) { maxd = std::max(maxd, (double) fabsf(d[i] - ref[i])); maxv = std::max(maxv, (double) fabsf(ref[i])); }
            int nonfinite = 0; for (size_t i = 0; i < (size_t) T * NU * M; ++i) nonfinite += !std::isfinite(d[i]);
            std::string tl; for (size_t k = 0; k < P.nt.size(); ++k) tl += (k ? "/" : "") + std::to_string(P.nt[k]);
            printf("  v%d: %8.1f us  %6.1f GFMA/s  tiles %-12s %s%s\n", v, best, fma / best / 1e3, tl.c_str(),
                   v == 0 ? "(reference)" : ("max|d| " + std::to_string(maxd) + " of " + std::to_string(maxv)).c_str(), nonfinite ? "  NONFINITE!" : "");
        }
    }
    return 0;
}
