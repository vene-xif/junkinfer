// Host <-> device copy bandwidth per Vulkan device (pinned host-visible memory <-> device-local memory, vkCmdCopyBuffer
// on the device's first queue). usage: xferbench [device index] [MiB]
// build: g++ -O2 -o xferbench xferbench.cpp -lvulkan
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, (int) r_); exit(1); } } while (0)

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want && !(mp.memoryTypes[i].propertyFlags & avoid)) return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}

int main(int argc, char ** argv) {
    const uint32_t dev_idx = argc > 1 ? atoi(argv[1]) : 0;
    const VkDeviceSize size = (VkDeviceSize) (argc > 2 ? atoi(argv[2]) : 256) << 20;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    if (dev_idx >= n) { fprintf(stderr, "device %u of %u\n", dev_idx, n); return 1; }
    VkPhysicalDevice pd = pds[dev_idx];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; CK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
    auto mk = [&](VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, VkBuffer & b, VkDeviceMemory & m) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_mem(pd, mr.memoryTypeBits, want, avoid);
        CK(vkAllocateMemory(dev, &mai, nullptr, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    };
    VkBuffer hb, db; VkDeviceMemory hm, dm;
    mk(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hb, hm);
    mk(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, db, dm);
    void * map; CK(vkMapMemory(dev, hm, 0, size, 0, &map)); memset(map, 1, size);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = 0;
    VkCommandPool pool; CK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; CK(vkCreateFence(dev, &fci, nullptr, &f));
    auto run = [&](bool up) {
        double best = 1e30;
        for (int it = 0; it < 5; ++it) {
            CK(vkResetCommandBuffer(cb, 0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
            VkBufferCopy cp{0, 0, size}; vkCmdCopyBuffer(cb, up ? hb : db, up ? db : hb, 1, &cp);
            CK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            CK(vkResetFences(dev, 1, &f));
            const auto t0 = std::chrono::steady_clock::now();
            CK(vkQueueSubmit(q, 1, &si, f)); CK(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return size / best / 1e9;
    };
    const double up = run(true), down = run(false);
    printf("device %u (%s): host->VRAM %.2f GB/s, VRAM->host %.2f GB/s (%llu MiB, best of 5)\n", dev_idx, pp.deviceName, up, down,
           (unsigned long long) (size >> 20));
    return 0;
}
