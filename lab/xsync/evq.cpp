// one device: does a CP wait (vkCmdWaitEvents) on queue family QF see a host vkSetEvent? usage: evq DEV QF
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, (int) r_, __LINE__); exit(1); } } while (0)
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const uint32_t dev = atoi(argv[1]), qf = atoi(argv[2]);
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice d; CK(vkCreateDevice(pds[dev], &dci, nullptr, &d)); VkQueue q; vkGetDeviceQueue(d, qf, 0, &q);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qf; VkCommandPool cp; CK(vkCreateCommandPool(d, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.commandBufferCount = 1; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(d, &cai, &cb));
    VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO}; VkEvent ev; CK(vkCreateEvent(d, &eci, nullptr, &ev));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; CK(vkCreateFence(d, &fci, nullptr, &f));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdWaitEvents(cb, 1, &ev, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(q, 1, &si, f));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto t0 = std::chrono::steady_clock::now();
    CK(vkSetEvent(d, ev));
    const VkResult r = vkWaitForFences(d, 1, &f, VK_TRUE, 3000000000ull);
    printf("device %u queue family %u: %s after %.0f us\n", dev, qf, r == VK_SUCCESS ? "released" : "STUCK (3 s)",
           std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
