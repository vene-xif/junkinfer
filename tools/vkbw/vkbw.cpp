// vkbw: host -> VRAM upload paths on one Vulkan device, measured.
//   A  vkCmdCopyBuffer from a host-visible staging buffer (uncached / cached) to device-local, by chunk size
//   B  CPU writes straight into device-local + host-visible memory (ReBAR mapping): memcpy and AVX2
//      non-temporal stores, 1..N threads
//   C  CPU memcpy into the staging buffer (the CPU half of path A)
//   D  copy on queue family 0 (graphics+compute) vs family 1 (async compute)
// build: g++ -O2 -mavx2 -std=c++17 vkbw.cpp -o vkbw -lvulkan -lpthread
#include <vulkan/vulkan.h>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pdev;
static VkPhysicalDeviceMemoryProperties mp;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Buf { VkBuffer b; VkDeviceMemory m; void * p = nullptr; size_t size; int type; };

static int find_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const auto f = mp.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return (int) i;
    }
    return -1;
}

static Buf make_buf(size_t size, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, bool map) {
    Buf B; B.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK(vkCreateBuffer(dev, &bi, nullptr, &B.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, B.b, &mr);
    B.type = find_type(mr.memoryTypeBits, want, avoid);
    if (B.type < 0) { fprintf(stderr, "no memory type for want=%x avoid=%x\n", want, avoid); exit(1); }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = B.type;
    VK(vkAllocateMemory(dev, &ai, nullptr, &B.m));
    VK(vkBindBufferMemory(dev, B.b, B.m, 0));
    if (map) VK(vkMapMemory(dev, B.m, 0, size, 0, &B.p));
    return B;
}

static void nt_copy(void * dst, const void * src, size_t n) {
    // dst and src 32-byte aligned, n multiple of 128
    __m256i * d = (__m256i *) dst; const __m256i * s = (const __m256i *) src;
    for (size_t i = 0; i < n / 32; i += 4) {
        __m256i a = _mm256_load_si256(s + i), b = _mm256_load_si256(s + i + 1), c = _mm256_load_si256(s + i + 2), e = _mm256_load_si256(s + i + 3);
        _mm256_stream_si256(d + i, a); _mm256_stream_si256(d + i + 1, b); _mm256_stream_si256(d + i + 2, c); _mm256_stream_si256(d + i + 3, e);
    }
    _mm_sfence();
}

static double cpu_copy(void * dst, const void * src, size_t n, int threads, bool nt, int reps) {
    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
        double t0 = now();
        std::vector<std::thread> th;
        const size_t per = (n / threads) & ~size_t(127);
        for (int t = 0; t < threads; ++t) {
            th.emplace_back([=] {
                char * d = (char *) dst + t * per; const char * s = (const char *) src + t * per;
                size_t len = (t == threads - 1) ? n - t * per : per;
                len &= ~size_t(127);
                if (nt) nt_copy(d, s, len); else memcpy(d, s, len);
            });
        }
        for (auto & x : th) x.join();
        best = std::min(best, now() - t0);
    }
    return n / best / 1e9;
}

int main(int argc, char ** argv) {
    const size_t SZ = (argc > 1 ? atoll(argv[1]) : 512) << 20;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    pdev = pds[0];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pdev, &pp);
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    printf("device: %s, buffer %zu MiB\n", pp.deviceName, SZ >> 20);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qf.data());
    std::vector<VkDeviceQueueCreateInfo> qci; float prio = 1.0f;
    std::vector<uint32_t> fams;
    for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT)) {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex = i; q.queueCount = 1; q.pQueuePriorities = &prio;
        qci.push_back(q); fams.push_back(i);
        printf("queue family %u: flags 0x%x count %u\n", i, qf[i].queueFlags, qf[i].queueCount);
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = (uint32_t) qci.size(); dci.pQueueCreateInfos = qci.data();
    VK(vkCreateDevice(pdev, &dci, nullptr, &dev));

    const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    Buf vram  = make_buf(SZ, DL, HV, false);            // plain device-local
    Buf bar   = make_buf(SZ, DL | HV | HC, 0, true);    // ReBAR mapping
    Buf stg_u = make_buf(SZ, HV | HC, DL | CA, true);   // write-combined system memory
    Buf stg_c = make_buf(SZ, HV | HC | CA, DL, true);   // cached system memory
    printf("memory types: vram %d, bar %d, staging-uncached %d, staging-cached %d\n", vram.type, bar.type, stg_u.type, stg_c.type);

    // pageable source, like an mmap'd model
    char * src = (char *) aligned_alloc(4096, SZ);
    for (size_t i = 0; i < SZ; i += 8) *(uint64_t *) (src + i) = i * 0x9E3779B97F4A7C15ull;

    const int nthr = std::thread::hardware_concurrency() / 2;
    printf("\n-- C: CPU into staging (GB/s)\n");
    for (int t : {1, 2, 4, nthr}) printf("  uncached memcpy %d thr: %.2f   nt %.2f\n", t, cpu_copy(stg_u.p, src, SZ, t, false, 3), cpu_copy(stg_u.p, src, SZ, t, true, 3));
    for (int t : {1, nthr}) printf("  cached   memcpy %d thr: %.2f\n", t, cpu_copy(stg_c.p, src, SZ, t, false, 3));

    printf("\n-- B: CPU straight into VRAM through the BAR (GB/s)\n");
    for (int t : {1, 2, 3, 4, nthr}) printf("  memcpy %d thr: %.2f   nt-stream %d thr: %.2f\n", t, cpu_copy(bar.p, src, SZ, t, false, 3), t, cpu_copy(bar.p, src, SZ, t, true, 3));

    // A / D: GPU copy engine
    for (uint32_t fam : fams) {
        VkQueue q; vkGetDeviceQueue(dev, fam, 0, &q);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = fam;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool pool; VK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
        VkCommandBuffer cb; VK(vkAllocateCommandBuffers(dev, &cai, &cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK(vkCreateFence(dev, &fci, nullptr, &fence));
        printf("\n-- A: vkCmdCopyBuffer staging -> VRAM, queue family %u (GB/s, total %zu MiB per submit)\n", fam, SZ >> 20);
        for (const Buf * stg : {&stg_u, &stg_c}) {
            for (size_t chunk : {size_t(256) << 10, size_t(2) << 20, size_t(16) << 20, SZ}) {
                double best = 1e9;
                for (int r = 0; r < 3; ++r) {
                    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    VK(vkBeginCommandBuffer(cb, &bi));
                    std::vector<VkBufferCopy> regs;
                    for (size_t o = 0; o < SZ; o += chunk) regs.push_back({o, o, std::min(chunk, SZ - o)});
                    vkCmdCopyBuffer(cb, stg->b, vram.b, (uint32_t) regs.size(), regs.data());
                    VK(vkEndCommandBuffer(cb));
                    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                    double t0 = now();
                    VK(vkQueueSubmit(q, 1, &si, fence));
                    VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
                    best = std::min(best, now() - t0);
                    VK(vkResetFences(dev, 1, &fence));
                    VK(vkResetCommandBuffer(cb, 0));
                }
                printf("  %-9s chunk %7zu KiB: %.2f\n", stg == &stg_u ? "uncached" : "cached", chunk >> 10, SZ / best / 1e9);
            }
        }
        // one submit + fence per 2 MiB chunk: the cost of a synchronous per-expert copy
        {
            const size_t chunk = size_t(2) << 20; const int N = 64; double t0 = now();
            for (int i = 0; i < N; ++i) {
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VK(vkBeginCommandBuffer(cb, &bi));
                VkBufferCopy rg{size_t(i) * chunk, size_t(i) * chunk, chunk};
                vkCmdCopyBuffer(cb, stg_u.b, vram.b, 1, &rg);
                VK(vkEndCommandBuffer(cb));
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                VK(vkQueueSubmit(q, 1, &si, fence)); VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
                VK(vkResetFences(dev, 1, &fence)); VK(vkResetCommandBuffer(cb, 0));
            }
            const double dt = now() - t0;
            printf("  sync per 2 MiB copy: %.2f GB/s, %.0f us per submit+wait\n", N * chunk / dt / 1e9, dt / N * 1e6);
        }
        vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    }
    printf("\ndone\n");
    return 0;
}
