#ifdef __linux__
	#define VK_USE_PLATFORM_XCB_KHR
	#define GLFW_EXPOSE_NATIVE_X11
	#include <X11/Xlib-xcb.h>
#endif

#ifdef __APPLE__
	#include <sys/syslimits.h>
	#define VK_USE_PLATFORM_MACOS_MVK
	#define GLFW_EXPOSE_NATIVE_COCOA
#endif

#ifdef _WIN32
	#define VK_USE_PLATFORM_WIN32_KHR 
	#define GLFW_EXPOSE_NATIVE_WIN32
	#define WIN32_LEAN_AND_MEAN 
	#define NOMINMAX
#endif

#define VOLK_IMPLEMENTATION
#define GLFW_INCLUDE_VULKAN
#include "volk.h"
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan_core.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef ARRAYSIZE
#define ARRAYSIZE(x) sizeof(x)/sizeof(x[0])
#endif
#define _CRT_SECURE_NO_WARNINGS

//#define VERBOSE
#define VK_CHECK(x) if (x) {printf("Vulkan error: %d\n", x); assert(0);}
#define VK_CHECK_PRESENT(x) if (x!=0 && x!=VK_SUBOPTIMAL_KHR && x!= VK_ERROR_OUT_OF_DATE_KHR ) {printf("Vulkan Present error: %d\n", x); assert(0);}
#define VK_CHECK_SWAPCHAIN(x) if (x!=0 &&  x!=1 && x!=2 && x!=VK_SUBOPTIMAL_KHR) {printf("Vulkan Swapchain error: %d\n", x); assert(0); }



const u32 desc_count{1};

struct UniformDesc
{
	VkDeviceMemory memory{};
	VkBuffer buffer{};
	VkDescriptorSet set{};
	u8 *ptr{};
};
struct Pipeline
{
	VkPipeline pipeline;
	VkPipelineLayout layout;
	UniformDesc uniform_desc; // Todo [desc_count]
	VkDescriptorSetLayout descriptor_set_layout;
};

struct Swapchain{
	VkSwapchainKHR swapchain{0};
	
	u32 w{0},h{0};
	u32 size{0};
	
	VkImage *images;
	VkFramebuffer *framebuffers;
	VkImageView *views;

	VkImage depth_image{};
	VkImageView depth_view{};
	VkDeviceMemory depth_mem{};
};

struct Buffer
{
	VkBuffer buffer{};
	VkDeviceMemory memory{};
	void      *data{};
	u32 size{};
};
using Barrier = VkBufferMemoryBarrier;

struct vulkan_app
{
	VkInstance			instance;
	VkPhysicalDevice	gpu;
	VkDevice			device;
	VkSurfaceKHR		surface;
	VkFormat			format;
	VkSurfaceFormatKHR	surface_format;
	VkQueue				queue;

	VkSemaphore			semaphore_1;
	VkSemaphore			semaphore_2;
	Swapchain			sfv;

	VkRenderPass		renderpass;
	VkViewport			viewport;
	VkRect2D			scissors;


	VkCommandPool		pool;
	u32					n_buffers;
	VkCommandBuffer		* buffers;

	VkFence				fence;

	VkDescriptorPool	descriptor_pool;

	VkPhysicalDeviceMemoryProperties mem_props;

	VkQueryPool			query_pool; 

	Pipeline* pipeline_1;
	Pipeline* pipeline_2;
	Pipeline* pipeline_3;

	u32 	n_resources;
	Buffer   *resources;
	u32 	n_resource_barriers;
	Barrier  *resource_barriers;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec2 tex;
};

struct IndexedBuffer
{
	struct VertexBuffer
	{
		Vertex *v{nullptr};
		u32 size{0};
	};
	struct IndexBuffer 
	{
		u32 *i{nullptr};
		u32 size{0};
	};

	VertexBuffer vertex_buffer;
	IndexBuffer  index_buffer;
};

struct Bindings
{
	u32 id[8];
	u32 size{0};
};

struct shader_specialization
{
	u32 algorithm {0};
	u32 workload  {0};
	u32 iterations{1};
	u32 n_points  {1};
	u32 n_caps    {1};
};

struct compute_shader_data
{
	u32 n_work_groups_x;
	u32 n_work_groups_y;
	u32 n_work_groups_z;

	u32 local_size_x;
	u32 local_size_y;
	u32 local_size_z;

	u32     n_buffers;
	Buffer   *buffers;
	Bindings  bindings;
	Barrier  *barriers;

	u32 n_dispatches;
	shader_specialization *specialization;
};

struct mvp_phong 
{
	glm::mat4 pview{1};
	glm::vec4 view_pos{0};
	glm::vec4 light_pos{0};
};

struct mvp_mat
{
	glm::mat4 mvp{1};
};

struct point_count 
{
	u32 n_pts;
};

struct model_mat 
{
	glm::mat4 model{1};
};

using PushBuffer = mvp_phong;
using PushBuffer2 = mvp_mat;
using PushBuffer3 = point_count;
using UniformBuffer = model_mat;

VkPhysicalDeviceMemoryProperties get_memory_properties(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);
#ifdef VERBOSE
	printf("Memory Type count %d.\n", mem_props.memoryTypeCount);
	for (u32 i{ 0 }; i < mem_props.memoryTypeCount; ++i)
	{
		VkMemoryType type = mem_props.memoryTypes[i];
		printf("\nMemory Type %d:\n", i);
		if (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			printf("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ");
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			printf("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ");
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
			printf("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ");
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
			printf("VK_MEMORY_PROPERTY_HOST_CACHED_BIT ");
	}
	printf("\nHeap Type count  %d.\n", mem_props.memoryHeapCount);
	for (u32 i{ 0 }; i < mem_props.memoryHeapCount; ++i)
	{
		VkMemoryHeap type = mem_props.memoryHeaps[i];
		printf("\nHeap Type %d:\n", i);
		if (type.flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			printf("VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ");
		if (type.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
			printf("VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ");
	}
	printf("\n");
#endif
	return mem_props;
}

u32 get_mem_type(const VkPhysicalDeviceMemoryProperties& mem_props)
{
    u32 im{0};
	for(; im<mem_props.memoryTypeCount; ++im)
	{
		VkMemoryType type = mem_props.memoryTypes[im];
		if(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && 
		   type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &&
		   type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ){
			break;
		}
	}
	assert(im!=mem_props.memoryTypeCount);
	return im;
}

// Find a memory in `memoryTypeBitsRequirement` that includes all of `requiredProperties`
u32 find_mem_properties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
                       uint32_t memoryTypeBitsRequirement,
                       VkMemoryPropertyFlags requiredProperties) 
{
    const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
        const uint32_t memoryTypeBits = (1 << memoryIndex);
        const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

        const VkMemoryPropertyFlags properties =
            pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
        const bool hasRequiredProperties =
            (properties & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties)
            return static_cast<int32_t>(memoryIndex);
    }

    // failed to find memory type
    return -1;
}

VkSwapchainKHR create_swapchain_object(VkPhysicalDevice gpu, VkDevice device, VkSurfaceFormatKHR format, VkSurfaceKHR surface, const VkSurfaceCapabilitiesKHR &caps, VkSwapchainKHR old_swapchain = 0)
{

	VkSurfaceCapabilitiesKHR surface_properties;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &surface_properties);

	VkSurfaceTransformFlagBitsKHR pre_transform;
	if (surface_properties.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
		pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	else
		pre_transform = surface_properties.currentTransform;

	VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
		composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
		composite = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
		composite = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
	else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
		composite = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;


	VkSwapchainKHR swapchain{0};	
	u32 queue_index{0};
	VkSwapchainCreateInfoKHR swap_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
	swap_info.surface = surface;
	swap_info.minImageCount = std::max(caps.minImageCount,2u);
	swap_info.imageFormat  = format.format; 
	swap_info.imageColorSpace = format.colorSpace;
	swap_info.imageExtent.width = caps.currentExtent.width;
	swap_info.imageExtent.height= caps.currentExtent.height;
	swap_info.imageArrayLayers = 1;
    swap_info.preTransform = pre_transform;
    swap_info.compositeAlpha = composite;
	swap_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swap_info.queueFamilyIndexCount = 1;
	swap_info.pQueueFamilyIndices = &queue_index;
	swap_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	swap_info.oldSwapchain = old_swapchain;
	VK_CHECK(vkCreateSwapchainKHR(device, &swap_info, 0, &swapchain));
	return swapchain;
}


void clean_swapchain(Swapchain& sfv, VkDevice device)
{
	vkDeviceWaitIdle(device);
	
	if(sfv.size){
		for(u32 i{0}; i< sfv.size; ++i){
			//printf("destroy image %i\n",i);
			//printf("destroy imageview %i\n",i);
			vkDestroyImageView(device, sfv.views[i], 0);
			//printf("destroy framebuffer%i\n",i);
			vkDestroyFramebuffer(device, sfv.framebuffers[i], 0);

		}
		//printf("delete pointer objects\n");
		delete[] sfv.views;
		delete[] sfv.framebuffers;
		delete[] sfv.images;

		//printf("destroy depth data\n");
		vkDestroyImage(device, sfv.depth_image,0);
		vkDestroyImageView(device, sfv.depth_view, 0);
		vkFreeMemory(device, sfv.depth_mem, 0);
	}


	//printf("destroy the swapchain\n");
	if(sfv.swapchain)
		vkDestroySwapchainKHR(device, sfv.swapchain, 0);
	//printf("Done cleaning\n");
}

void create_swapchain(Swapchain& sfv, VkPhysicalDevice gpu, VkDevice device, VkSurfaceFormatKHR format, VkSurfaceKHR surface, VkRenderPass renderpass)
{
	VkSurfaceCapabilitiesKHR caps;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps));

	//printf("create sw obj\n");
	Swapchain old_swapchain = sfv;
	sfv.swapchain = create_swapchain_object(gpu, device, format, surface, caps, old_swapchain.swapchain);

	u32 n_images;
	VK_CHECK(vkGetSwapchainImagesKHR(device, sfv.swapchain, &n_images, 0));
	assert(n_images);

	//printf("get images \n");
	sfv.images = new VkImage[n_images];
	VK_CHECK(vkGetSwapchainImagesKHR(device, sfv.swapchain, &n_images, sfv.images));
	sfv.views = new VkImageView[n_images];
	VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	view_info.format = format.format;
	view_info.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;

	sfv.framebuffers = new VkFramebuffer[n_images];
	VkFramebufferCreateInfo frame_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	frame_info.renderPass 	   = renderpass;
	frame_info.attachmentCount = 2;
	frame_info.width  = caps.currentExtent.width;
	frame_info.height = caps.currentExtent.height;
	frame_info.layers = 1;

	
	VkImageCreateInfo depth_image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	depth_image_info.format = VK_FORMAT_D32_SFLOAT;
	depth_image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depth_image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_image_info.extent  = {caps.currentExtent.width, caps.currentExtent.height, 1};
	depth_image_info.arrayLayers = 1;
	depth_image_info.mipLevels   = 1;
	depth_image_info.imageType = VK_IMAGE_TYPE_2D;
	//printf("create images \n");
	VK_CHECK(vkCreateImage(device, &depth_image_info,0,&sfv.depth_image));

	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);
	VkMemoryRequirements d_mem_reqs{};
	vkGetImageMemoryRequirements(device, sfv.depth_image, &d_mem_reqs);
	VkMemoryAllocateInfo d_mem_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	d_mem_alloc.allocationSize = d_mem_reqs.size;


// Try to find an optimal memory type, or if it does not exist try fallback memory type
// `device` is the VkDevice
// `image` is the VkImage that requires memory to be bound
// `memoryProperties` properties as returned by vkGetPhysicalDeviceMemoryProperties
// `requiredProperties` are the property flags that must be present
// `optimalProperties` are the property flags that are preferred by the application
	int32_t memoryType =
    find_mem_properties(&mem_props, d_mem_reqs.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	assert(memoryType != -1); // not found; try fallback properties

	d_mem_alloc.memoryTypeIndex = memoryType;//get_mem_type(mem_props);
	VK_CHECK(vkAllocateMemory(device,&d_mem_alloc,0, &sfv.depth_mem));
	VK_CHECK(vkBindImageMemory(device, sfv.depth_image, sfv.depth_mem,0));

	VkImageViewCreateInfo depth_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	depth_view_info.image = sfv.depth_image;
	depth_view_info.format= VK_FORMAT_D32_SFLOAT;
	depth_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	depth_view_info.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT , 0,1,0,1};
	//printf("create depth imageview\n");	
	VK_CHECK(vkCreateImageView(device, &depth_view_info, 0, &sfv.depth_view));

	VkImageView attachments[2];
	frame_info.pAttachments = attachments;

	for(u32 i{0}; i< n_images; ++i){
		view_info.image = sfv.images[i];
		//printf("create %i imageview\n",i);	
		VK_CHECK(vkCreateImageView(device, &view_info, 0, sfv.views+i));
		attachments[0] = *(sfv.views+i);
		attachments[1] = sfv.depth_view;
		//printf("create %i framebuffer\n",i);	
		VK_CHECK(vkCreateFramebuffer(device, &frame_info, 0, sfv.framebuffers+i));
	}

	sfv.size = n_images;
	sfv.w = caps.currentExtent.width;
	sfv.h = caps.currentExtent.height;
	//printf("clean old swapchain\n");	
	clean_swapchain(old_swapchain, device);
}

VkRenderPass create_renderpass(VkDevice device, VkFormat format)
{
	VkAttachmentDescription attachments[2]{
		{
		.flags = 0,
		.format = format,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		},
		{
		.flags = 0,
		.format = VK_FORMAT_D32_SFLOAT,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		}
	};
	
	VkRenderPass renderpass{0};
	const u32 n_subpasses{1};
	VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription subpasses[n_subpasses]{
	{
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &color_ref,
		.pDepthStencilAttachment=&depth_ref
	}
	};

	/*
	const u32 n_dependencies{1};
	VkSubpassDependency dependencies[n_dependencies] {
	{
		.srcSubpass = VK_SUBPASS_EXTERNAL,
		.dstSubpass = 0,
		.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		.srcAccessMask = VK_ACCESS_NONE,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, //| VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dependencyFlags = 0
	}
	};
	*/
	
	const u32 n_dependencies{2};
	VkSubpassDependency dependencies[n_dependencies];

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	dependencies[0].dependencyFlags = 0;

	dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].dstSubpass = 0;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].srcAccessMask = 0;
	dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
	dependencies[1].dependencyFlags = 0;
/*
	dependencies[2].srcSubpass = 0;
	dependencies[2].dstSubpass = 0;
	dependencies[2].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[2].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[2].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[2].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	dependencies[2].dependencyFlags = 0;
*/
	VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	pass_info.attachmentCount = ARRAYSIZE(attachments);
	pass_info.pAttachments = attachments;
	pass_info.subpassCount = n_subpasses;
	pass_info.pSubpasses = subpasses;
    pass_info.dependencyCount = n_dependencies;
    pass_info.pDependencies = dependencies;
	VK_CHECK(vkCreateRenderPass(device, &pass_info, 0, &renderpass));
	return renderpass;
}

void create_descriptor_pool(VkDevice device, VkDescriptorPool& pool)
{
	VkDescriptorPoolSize desc_pool_size[1]
	{
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, desc_count}
	};
	VkDescriptorPoolCreateInfo desc_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	desc_pool_info.poolSizeCount = 1;
	desc_pool_info.pPoolSizes = desc_pool_size;
	desc_pool_info.maxSets = desc_count;
	VK_CHECK(vkCreateDescriptorPool(device,&desc_pool_info, 0,&pool));
}

void create_descriptor_set_layouts(VkDevice device, VkDescriptorSetLayout *layouts)
{
	VkDescriptorSetLayoutBinding layout_binding{};
	layout_binding.binding = 0;
	layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layout_binding.descriptorCount = 1;
	layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo lay_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	lay_info.bindingCount = 1;
	lay_info.pBindings = &layout_binding;
	for(u32 i{0}; i<desc_count; i++){
		VK_CHECK(vkCreateDescriptorSetLayout(device, &lay_info, 0, &layouts[i]));
	}
}

void create_descriptor_sets(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout *layout, UniformDesc *uniform_desc)
{
	for(u32 i{0}; i<desc_count; i++)
	{
		VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc.pSetLayouts = layout;
		alloc.descriptorPool = pool;
		alloc.descriptorSetCount = 1;
		VkResult res = vkAllocateDescriptorSets(device, &alloc, &uniform_desc[i].set);
		
		switch(res){
			case VK_ERROR_FRAGMENTED_POOL:
				printf("Fragmentation!\n");
				assert(0);
				break;
			case VK_ERROR_OUT_OF_POOL_MEMORY:
				printf("No pool memory!\n");
				assert(0);
				break;
			case VK_ERROR_OUT_OF_HOST_MEMORY:
				printf("No host memory!\n");
				assert(0);
				break;
			case VK_ERROR_OUT_OF_DEVICE_MEMORY:
				printf("No device memory!\n");
				assert(0);
				break;
			default:
				VK_CHECK(res);
		}

		VkWriteDescriptorSet write_set{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		VkDescriptorBufferInfo buffer_info{uniform_desc[i].buffer, 0, sizeof(UniformBuffer)};
		write_set.dstSet = uniform_desc[i].set;
		write_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		write_set.descriptorCount = 1;
		write_set.pBufferInfo = &buffer_info;
		write_set.dstBinding = 0; // Binding 0
		vkUpdateDescriptorSets(device, 1, &write_set, 0, 0);
	}
}

void create_uniform_buffers(VkDevice device, VkPhysicalDevice gpu, UniformDesc* uniform_desc)
{
	VkMemoryRequirements reqs{};
	VkBufferCreateInfo buf_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	buf_info.size = sizeof(UniformBuffer);
	buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);
	for(u32 i{0}; i<desc_count; i++){
		VK_CHECK(vkCreateBuffer(device, &buf_info, 0, &uniform_desc[i].buffer));
		vkGetBufferMemoryRequirements(device, uniform_desc[i].buffer, &reqs);
		alloc.allocationSize = reqs.size;
		alloc.memoryTypeIndex = get_mem_type(mem_props);
		VK_CHECK(vkAllocateMemory(device, &alloc, 0, &uniform_desc[i].memory));
		VK_CHECK(vkBindBufferMemory(device, uniform_desc[i].buffer, uniform_desc[i].memory, 0));
		VK_CHECK(vkMapMemory(device, uniform_desc[i].memory, 0, sizeof(UniformBuffer), 0, 
							(void**)&uniform_desc[i].ptr));
	}
}


struct Buffers
{
	Buffer* buffers[255];
	u32 size{0};
} buffer_list;

void add_buffer(Buffer *buffer)
{
	assert(buffer_list.size < 255-1);
	buffer_list.buffers[buffer_list.size++] = buffer;
}

void destroy_buffer(VkDevice device, Buffer& buffer)
{
	if (buffer.memory)
	{
		vkFreeMemory(device, buffer.memory, 0);
		buffer.memory = 0;
	}
	if(buffer.buffer)
		vkDestroyBuffer(device, buffer.buffer, 0);
	buffer.buffer = 0;
}

Buffer create_buffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, u32 memory_type_index, void* data = nullptr)
{
	Buffer buffer{};
	VkMemoryRequirements requirements{};
	VkBufferCreateInfo  bufferinfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	VkMemoryAllocateInfo allocinfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	bufferinfo.size  = size;
	bufferinfo.usage = usage;
	VK_CHECK(vkCreateBuffer      (device, &bufferinfo, 0, &buffer.buffer));
	vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
	allocinfo.allocationSize  = requirements.size;
	allocinfo.memoryTypeIndex = memory_type_index;
	VK_CHECK(vkAllocateMemory    (device, &allocinfo, 0, &buffer.memory));
	VK_CHECK(vkBindBufferMemory  (device, buffer.buffer, buffer.memory,0));
	VK_CHECK(vkMapMemory         (device, buffer.memory, 0, size, 0, &buffer.data));
	buffer.size = size;

	if(data) memcpy(buffer.data, data, size);

	return buffer;	
};

VkFence create_fence(VkDevice device, VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT)
{
	VkFence fence{};
	VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	fence_info.flags = flags;
	VK_CHECK(vkCreateFence(device, &fence_info, 0, &fence));
	return fence;
};

Buffer create_device_buffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, 
							VkPhysicalDeviceMemoryProperties& properties, void* data = nullptr, u32 mem_flags = 0)
{
	Buffer buffer{};
	VkMemoryRequirements requirements{};
	VkBufferCreateInfo  bufferinfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	VkMemoryAllocateInfo allocinfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	bufferinfo.size  = size;
	bufferinfo.usage = usage;
	if(data) bufferinfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT; 
	VK_CHECK(vkCreateBuffer      (device, &bufferinfo, 0, &buffer.buffer));
	vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
	u32 memory_type =
    find_mem_properties(&properties, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | mem_flags);
	//printf("Device memory type: %u\n", memory_type);
	allocinfo.allocationSize  = requirements.size;
	allocinfo.memoryTypeIndex = memory_type;
	VK_CHECK(vkAllocateMemory    (device, &allocinfo, 0, &buffer.memory));
	VK_CHECK(vkBindBufferMemory  (device, buffer.buffer, buffer.memory,0));
	//VK_CHECK(vkMapMemory         (device, buffer.memory, 0, size, 0, &buffer.data));
	buffer.size = size;
	
	if(data) 
	{
		memory_type =get_mem_type(properties);
		//printf("Staging memory type: %u\n", memory_type);
		u32 memory_type2 =
		find_mem_properties(&properties, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );
		Buffer staging_buffer = create_buffer(device, size, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memory_type2, data);
		vkUnmapMemory(device, staging_buffer.memory);
		VkCommandPool pool;
		VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		VK_CHECK(vkCreateCommandPool(device, &pool_info, 0, &pool));
		VkCommandBuffer cmd_buffer;
		VkCommandBufferAllocateInfo allo_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		allo_info.commandPool= pool;
		allo_info.commandBufferCount = 1;
		VK_CHECK(vkAllocateCommandBuffers(device, &allo_info, &cmd_buffer));
		VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));
		VkBufferCopy buffer_region{};
		buffer_region.size = size;
		vkCmdCopyBuffer(cmd_buffer, staging_buffer.buffer, buffer.buffer,1, &buffer_region);
		VK_CHECK(vkEndCommandBuffer(cmd_buffer));

		VkFence fence = create_fence(device, 0);
		VkQueue queue{0};
		vkGetDeviceQueue(device,0,0,&queue);

		VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &cmd_buffer;
		VK_CHECK(vkQueueSubmit(queue,1, &submit_info, fence));

		VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, pool, 1, &cmd_buffer);

		destroy_buffer(device, staging_buffer);
		vkDestroyCommandPool(device, pool, 0);
	}

	return buffer;	
};

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device)
{
	VkDescriptorSetLayout layout{};
	VkDescriptorSetLayoutBinding binding[1]{
	{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = 0
	}
	};
	VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	info.bindingCount = ARRAYSIZE(binding);
	info.pBindings = binding;

	VK_CHECK(vkCreateDescriptorSetLayout(device, &info, 0, &layout));
	return layout;
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device, Bindings bindings)
{
	VkDescriptorSetLayoutBinding* binding = new VkDescriptorSetLayoutBinding[bindings.size];

	for(u32 i{0}; i<bindings.size; i++){
		binding[i] = 
		{
		.binding = bindings.id[i],
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags =  VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = 0
		};
	
		// Binding 0 will be accessed by vertex shader stage
		//if(bindings.id[i] == 0)
		//	binding[i].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
	}

	VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	info.bindingCount = bindings.size;
	info.pBindings = binding;

	VkDescriptorSetLayout	layout{};
	VK_CHECK(vkCreateDescriptorSetLayout(device, &info, 0, &layout));
	delete[] binding;
	return layout;
}


Pipeline* create_pipeline_vertex_attributes(vulkan_app& vk,
	const char* vert_name, const char* frag_name,
	VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
{
	auto& gpu = vk.gpu;
	auto& device = vk.device;
	auto& renderpass = vk.renderpass;


	auto vert = read_binary_file(shader_folder / vert_name);
	auto frag = read_binary_file(shader_folder / frag_name);

	VkShaderModule modules[2]{0,0};
	VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	shader_info.codeSize = vert.size;
	shader_info.pCode = reinterpret_cast<const u32*>(vert.data);
	VK_CHECK(vkCreateShaderModule(device, &shader_info, 0, modules));
	shader_info.codeSize = frag.size;
	shader_info.pCode = reinterpret_cast<const u32*>(frag.data);
	VK_CHECK(vkCreateShaderModule(device, &shader_info, 0, modules+1));

	VkPipelineShaderStageCreateInfo stages[2];
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].flags = 0;
	stages[0].pNext= 0;
	stages[0].pSpecializationInfo = 0;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = modules[0];
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = modules[1];
	stages[1].pName = "main";
	stages[1].flags = 0;
	stages[1].pSpecializationInfo = 0;
	stages[1].pNext= 0;

	auto pipeline = new Pipeline;
	create_descriptor_set_layouts(device, &pipeline->descriptor_set_layout);

	VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.pushConstantRangeCount = 1;
	VkPushConstantRange push_range[] = {{VK_SHADER_STAGE_VERTEX_BIT , 0, sizeof(PushBuffer)}};
	layout_info.pPushConstantRanges = push_range;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &pipeline->descriptor_set_layout;

	VK_CHECK(vkCreatePipelineLayout(device, &layout_info, 0, &pipeline->layout));

	create_uniform_buffers(device, gpu, &pipeline->uniform_desc);
	create_descriptor_sets(device, vk.descriptor_pool, &pipeline->descriptor_set_layout, &pipeline->uniform_desc);

	VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rasterization.lineWidth = 1.;
	rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterization.polygonMode = VK_POLYGON_MODE_FILL;
	rasterization.cullMode = VK_CULL_MODE_BACK_BIT;

	VkPipelineColorBlendAttachmentState blend_attachments{};
	blend_attachments.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT 
		                       	     | VK_COLOR_COMPONENT_B_BIT |VK_COLOR_COMPONENT_A_BIT; 

	VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	blend.attachmentCount = 1;
	blend.pAttachments = &blend_attachments;

	VkPipelineMultisampleStateCreateInfo multi_sample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	multi_sample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkVertexInputAttributeDescription attributes[3]
	{
		{0,0,VK_FORMAT_R32G32B32_SFLOAT, 0},
		{1,0,VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
		{2,0,VK_FORMAT_R32G32_SFLOAT,	 offsetof(Vertex, nor)}
	};
	
	VkVertexInputBindingDescription bindings[1]
	{
		{0, sizeof(Vertex)}
	};

	VkPipelineVertexInputStateCreateInfo vertex_info{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vertex_info.vertexBindingDescriptionCount = 1;
	vertex_info.vertexAttributeDescriptionCount = 3;
	vertex_info.pVertexBindingDescriptions = bindings;
	vertex_info.pVertexAttributeDescriptions  = attributes;
	VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	input.topology = topology;

	VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	VkDynamicState dstates[2] = { VK_DYNAMIC_STATE_SCISSOR , VK_DYNAMIC_STATE_VIEWPORT};
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dstates;

	VkPipelineViewportStateCreateInfo viewport_info{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	viewport_info.viewportCount = 1;
	viewport_info.scissorCount  = 1;

	VkPipelineDepthStencilStateCreateInfo depths_info{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depths_info.depthTestEnable  = VK_TRUE;
    depths_info.depthWriteEnable = VK_TRUE;
    depths_info.depthCompareOp   = VK_COMPARE_OP_LESS;
	//depths_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

	VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	pipeline_info.stageCount = 2;
	pipeline_info.pStages = stages;
	pipeline_info.pVertexInputState = &vertex_info;
	pipeline_info.pInputAssemblyState = &input;
	pipeline_info.pRasterizationState = &rasterization;
	pipeline_info.pColorBlendState = &blend;
	pipeline_info.pMultisampleState = &multi_sample;
	pipeline_info.pDynamicState = &dynamic;
	pipeline_info.pViewportState = &viewport_info;
	pipeline_info.pDepthStencilState = &depths_info;
	pipeline_info.layout = pipeline->layout;
	pipeline_info.renderPass = renderpass;
	pipeline_info.subpass = 0;

	VK_CHECK(vkCreateGraphicsPipelines(device, 0, 1, &pipeline_info, 0, &pipeline->pipeline));


	vkDestroyShaderModule(device, modules[0], 0);
	vkDestroyShaderModule(device, modules[1], 0);
	delete[] frag.data;
	delete[] frag.name;
	delete[] vert.data; 
	delete[] vert.name;
	return pipeline;
}


Pipeline* create_pipeline_vertex_pull(vulkan_app& vk,
	const char* vert_name, const char* frag_name,
	VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
{
	auto& gpu = vk.gpu;
	auto& device = vk.device;
	auto& renderpass = vk.renderpass;

	auto vert = read_binary_file(shader_folder / vert_name);
	auto frag = read_binary_file(shader_folder / frag_name);

	VkShaderModule modules[2]{0,0};
	VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	shader_info.codeSize = vert.size;
	shader_info.pCode = reinterpret_cast<const u32*>(vert.data);
	VK_CHECK(vkCreateShaderModule(device, &shader_info, 0, modules));
	shader_info.codeSize = frag.size;
	shader_info.pCode = reinterpret_cast<const u32*>(frag.data);
	VK_CHECK(vkCreateShaderModule(device, &shader_info, 0, modules+1));

	VkPipelineShaderStageCreateInfo stages[2];
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].flags = 0;
	stages[0].pNext= 0;
	stages[0].pSpecializationInfo = 0;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = modules[0];
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = modules[1];
	stages[1].pName = "main";
	stages[1].flags = 0;
	stages[1].pSpecializationInfo = 0;
	stages[1].pNext= 0;

	
	auto pipeline = new Pipeline;

	pipeline->descriptor_set_layout = create_descriptor_set_layout(device);

	VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.pushConstantRangeCount = 1;
	VkPushConstantRange push_range[] = { {VK_SHADER_STAGE_VERTEX_BIT ,0, sizeof(PushBuffer2)} };
	layout_info.pPushConstantRanges = push_range;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &pipeline->descriptor_set_layout;

	VK_CHECK(vkCreatePipelineLayout(device, &layout_info, 0, &pipeline->layout));


	VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rasterization.lineWidth = 1.;

	VkPipelineColorBlendAttachmentState blend_attachments{};
	blend_attachments.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT 
		                       	     | VK_COLOR_COMPONENT_B_BIT |VK_COLOR_COMPONENT_A_BIT; 

	VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	blend.attachmentCount = 1;
	blend.pAttachments = &blend_attachments;

	VkPipelineMultisampleStateCreateInfo multi_sample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	multi_sample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


	VkPipelineVertexInputStateCreateInfo vertex_info{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	input.topology = topology;

	VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	VkDynamicState dstates[2] = { VK_DYNAMIC_STATE_SCISSOR , VK_DYNAMIC_STATE_VIEWPORT};
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dstates;

	VkPipelineViewportStateCreateInfo viewport_info{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	viewport_info.viewportCount = 1;
	viewport_info.scissorCount = 1;

	VkPipelineDepthStencilStateCreateInfo depths_info{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depths_info.depthTestEnable = VK_TRUE;
    depths_info.depthWriteEnable = VK_TRUE;
    depths_info.depthCompareOp = VK_COMPARE_OP_LESS;

	VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	pipeline_info.stageCount = 2;
	pipeline_info.pStages = stages;
	pipeline_info.pVertexInputState = &vertex_info;
	pipeline_info.pInputAssemblyState = &input;
	pipeline_info.pRasterizationState = &rasterization;
	pipeline_info.pColorBlendState = &blend;
	pipeline_info.pMultisampleState = &multi_sample;
	pipeline_info.pDynamicState = &dynamic;
	pipeline_info.pViewportState = &viewport_info;
	pipeline_info.pDepthStencilState = &depths_info;
	pipeline_info.layout = pipeline->layout;
	pipeline_info.renderPass = renderpass;
	pipeline_info.subpass = 0;

	VK_CHECK(vkCreateGraphicsPipelines(device, 0, 1, &pipeline_info, 0, &pipeline->pipeline));

	vkDestroyShaderModule(device, modules[0], 0);
	vkDestroyShaderModule(device, modules[1], 0);
	delete[] frag.data; 
	delete[] frag.name;
	delete[] vert.data; 
	delete[] vert.name;
	return pipeline;
}


template <class push_buffer_type>
void create_pipeline_compute(VkDevice device, Pipeline *pipelines, const char* comp_name, u32 n_pipelines, Bindings bindings, shader_specialization *specializations, push_buffer_type)
{
	VkShaderModule module;
	auto comp = read_binary_file(compute_folder / comp_name);

	VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	shader_info.codeSize = comp.size;
	shader_info.pCode = reinterpret_cast<const u32*>(comp.data);
	module = 0;
	VK_CHECK(vkCreateShaderModule(device, &shader_info, 0, &module));
	delete[] comp.data;
	delete[] comp.name;

	VkSpecializationMapEntry specialization_entries[] 
	{
		{0,offsetof(shader_specialization, algorithm),  sizeof(u32)},
		{1,offsetof(shader_specialization, workload ),  sizeof(u32)},
		{2,offsetof(shader_specialization, iterations), sizeof(u32)},
		{3,offsetof(shader_specialization, n_points),	sizeof(u32)},
		{4,offsetof(shader_specialization, n_caps),		sizeof(u32)}
	};

	VkPipelineShaderStageCreateInfo stages[1];
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].flags = 0;
	stages[0].pNext = 0;
	stages[0].stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stages[0].module = module;
	stages[0].pName = "main";


	VkPushConstantRange		   push_range[] = { {VK_SHADER_STAGE_COMPUTE_BIT ,0 , sizeof(push_buffer_type)} };
	VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = push_range;
	layout_info.setLayoutCount = 1;



	for(u32 u{0}; u<n_pipelines; ++u)
	{
		pipelines[u].descriptor_set_layout = create_descriptor_set_layout(device, bindings);

		layout_info.pSetLayouts = &pipelines[u].descriptor_set_layout;
		VK_CHECK(vkCreatePipelineLayout(device, &layout_info, 0, &pipelines[u].layout));

		VkSpecializationInfo specialization_info = 
		{ sizeof(specialization_entries)/sizeof(specialization_entries[0]) , specialization_entries, sizeof(shader_specialization), specializations+u};

		stages[0].pSpecializationInfo = &specialization_info;

		VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};

		pipeline_info.layout = pipelines[u].layout;
		pipeline_info.stage = stages[0];

		VK_CHECK(vkCreateComputePipelines(device, 0, 1, &pipeline_info, 0, &pipelines[u].pipeline));
	}

	vkDestroyShaderModule(device, module, 0);
}


void print_compute_device_properties(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(gpu, &properties);
	
	VkPhysicalDeviceLimits limits = properties.limits;

    printf("Maximal Shared memory: %u\n", limits.maxComputeSharedMemorySize);
    printf("Maximal Work group count : %u, %u, %u\n", limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1], limits.maxComputeWorkGroupCount[2]);
    printf("Maximal Invocation count : %u\n", limits.maxComputeSharedMemorySize);
    printf("Maximal Work group size: %u,%u,%u\n", limits.maxComputeWorkGroupSize[0], limits.maxComputeWorkGroupSize[1], limits.maxComputeWorkGroupSize[2]);
}

f32 ticks_per_nanosecond(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(gpu, &properties);
	
	VkPhysicalDeviceLimits limits = properties.limits;
	
	return limits.timestampPeriod;
}

VkQueryPool create_query_pool(VkDevice device, u32 n_queries)
{
	VkQueryPoolCreateInfo create_info{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	create_info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	create_info.queryCount = n_queries;

	VkQueryPool pool{};
	VK_CHECK(vkCreateQueryPool(device, &create_info, 0, &pool));

	return pool;
}

VkBufferMemoryBarrier make_barrier(Buffer buffer, VkAccessFlags from = VK_ACCESS_SHADER_WRITE_BIT, 
		VkAccessFlags to = VK_ACCESS_SHADER_READ_BIT ){
	VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
	barrier.srcAccessMask = from;
	barrier.dstAccessMask = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = buffer.buffer;
	barrier.offset = 0;
	barrier.size = buffer.size;
	return barrier;
}

VkCommandPool make_pool(VkDevice device){
	VkCommandPool pool;
	VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT ;
	VK_CHECK(vkCreateCommandPool(device, &pool_info, 0, &pool));
	return pool;
}

void allocate_cmd_buffers(VkDevice device, u32 n_buffers, VkCommandBuffer *buffers, VkCommandPool pool)
{
	VkCommandBufferAllocateInfo allo_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	allo_info.commandPool= pool;
	allo_info.commandBufferCount = n_buffers;
	VK_CHECK(vkAllocateCommandBuffers(device, &allo_info, buffers));
}

VkInstance create_instance()
{
	u32 count;

	u32 api_version;	
	VK_CHECK(vkEnumerateInstanceVersion(&api_version));
	u32 v = VK_VERSION_MINOR(api_version);
	printf("Api version: 1.%u\n", v);

#ifdef VERBOSE
	vkEnumerateInstanceExtensionProperties(0, &count, 0);
	VkExtensionProperties all_ext[255];
	vkEnumerateInstanceExtensionProperties(0, &count, all_ext);
	for (u32 i{0}; i < count; i++)
		printf("%s \n", all_ext[i].extensionName);
#endif

	VkInstance instance{0};
	VkInstanceCreateInfo instance_info{};
	instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	VkApplicationInfo app_info{};
	instance_info.pApplicationInfo = &app_info;
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

#ifdef __APPLE__
	instance_info.flags = 1;
	instance_info.enabledExtensionCount = 3;
	app_info.apiVersion = VK_API_VERSION_1_2;
	const char *extensions[3] = {"VK_KHR_portability_enumeration", "VK_KHR_surface", "VK_EXT_metal_surface"};
#endif


#ifdef __linux__
	instance_info.enabledExtensionCount = 2;
	app_info.apiVersion = api_version;
	const char *extensions[2] = { "VK_KHR_surface", "VK_KHR_xcb_surface"};
#endif
#ifdef _WIN32
	instance_info.enabledExtensionCount = 2;
	app_info.apiVersion = VK_API_VERSION_1_3;
	const char* extensions[2] = { "VK_KHR_surface", "VK_KHR_win32_surface" };
#endif

	instance_info.ppEnabledExtensionNames = extensions;
	VK_CHECK(vkCreateInstance(&instance_info, 0, &instance));
	volkLoadInstance(instance);

	return instance;
}

struct Device
{
	VkPhysicalDevice gpu;
	VkDevice 		 device;
};

bool test_extensions(VkPhysicalDevice gpu, VkDeviceQueueCreateInfo& queue_info, u32 n_dev_ext, const char** extensions_dev)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(gpu, &properties);
	u32 qcount{};
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qcount, 0);
	VkQueueFamilyProperties* qprops = new VkQueueFamilyProperties[qcount];
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qcount, qprops);
	printf("Getting Graphics and Compute queue\n");
	for(u32 j{0}; j<qcount; ++j)\
	{
		printf("Queue %d:",j);
		if(qprops[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			printf("graphics ");
		if(qprops[j].queueFlags & VK_QUEUE_COMPUTE_BIT )
			printf("compute ");
		if(qprops[j].queueFlags & VK_QUEUE_TRANSFER_BIT )
			printf("transfer");
		printf("\n");
		if((qprops[j].queueFlags & VK_QUEUE_GRAPHICS_BIT ) && (qprops[j].queueFlags & VK_QUEUE_COMPUTE_BIT)){
			queue_info.queueFamilyIndex = j;
			break;
		}
	}
	printf("Test extensions for %s\n", properties.deviceName);
	u32 ecount;
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ecount, 0);	
	VkExtensionProperties* eprops = new VkExtensionProperties[ecount];
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ecount, eprops);	
	u32 found_ext{0};
	for(u32 e{0}; e<n_dev_ext; ++e){
		bool has_ext{false};
		for(u32 k{0}; k<ecount; ++k){
			if(strcmp(eprops[k].extensionName, extensions_dev[e]) == 0){
				has_ext = true;
				printf("Found %s\n", eprops[k].extensionName); 
				break;
			}
		}
		if(!has_ext){
			break;
		}else
			found_ext++;
	}

	delete[] qprops;
	delete[] eprops;

	if(found_ext != n_dev_ext)
		return false;

	printf("Found all extensions for this %s.\n", 
			[](VkPhysicalDeviceType type){
			switch(type){
			case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			return "OTHER";
			case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return "integrated gpu";
			case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return "discrete gpu";
			case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return "VIRTUAL";
			case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return "CPU";
			default:
			return "?";
			}}(properties.deviceType));

	return true;
}

Device create_device(VkInstance instance, VkPhysicalDeviceType prefered_gpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
{
	u32 count;
	VkDeviceQueueCreateInfo queue_info{};
	queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	float priorities[1] = {1.f};
	queue_info.queueCount = 1;
	queue_info.pQueuePriorities = priorities;

#ifdef __APPLE__
	const u32 n_dev_ext{3};
	const char *extensions_dev[n_dev_ext] = {"VK_KHR_portability_subset", 
											 "VK_KHR_swapchain",
											 "VK_KHR_push_descriptor"};
#endif
#ifdef __linux__
	const u32 n_dev_ext{2};
	const char *extensions_dev[n_dev_ext] = {"VK_KHR_swapchain", "VK_KHR_push_descriptor"};
#endif
#ifdef _WIN32
	const u32 n_dev_ext{ 2 };
	const char* extensions_dev[n_dev_ext] = { "VK_KHR_swapchain", "VK_KHR_push_descriptor" };
#endif

	vkEnumeratePhysicalDevices(instance, &count, 0);
	VkPhysicalDevice gpus[8];
	vkEnumeratePhysicalDevices(instance, &count, gpus);
	u32 u{0};
	VkPhysicalDeviceProperties properties{};
	VkPhysicalDevice gpu{0};

	u32 u_discrete = ~0;
	u32 u_integrated = ~0;
	u32 n_gpus = 0;
	u32 u_gpus[2];

	for (; u < count; u++) 
	{
		vkGetPhysicalDeviceProperties(gpus[u], &properties);
		printf("%d: %s\n",u, properties.deviceName);
		if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			u_discrete = u;
		}
		else if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
		{
			u_integrated = u;
		}
	}

	if (u_discrete != ~0) {
		u_gpus[n_gpus] = u_discrete;
		if (prefered_gpu == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			gpu = gpus[u_discrete];
			printf("Using the %s GPU\n", "discrete");
		}
		n_gpus++;
	}
	if(u_integrated!= ~0)
	{
		u_gpus[n_gpus] = u_integrated;
		if (prefered_gpu == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || u_discrete==~0)
		{
			gpu = gpus[u_integrated];
			printf("Using the %s GPU\n", "integrated");
		}
		n_gpus++;
	}
	if(!n_gpus)	
		assert(!"No gpu");

	

	for(u32 u{0}; u< n_gpus; u++)
		assert(test_extensions(gpus[u],queue_info, n_dev_ext, extensions_dev));


	VkDeviceCreateInfo device_info{};
	device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos = &queue_info;
	device_info.enabledExtensionCount   = n_dev_ext;
	device_info.ppEnabledExtensionNames = extensions_dev;

	printf("Create Device\n");

	VkPhysicalDevice16BitStorageFeatures features16 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
	features16.storageBuffer16BitAccess = true;

	VkPhysicalDevice8BitStorageFeaturesKHR features8 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR };
	features8.storageBuffer8BitAccess = true;
	
	device_info.pNext = &features16;
	features16.pNext = &features8;

	VkPhysicalDeviceShaderFloat16Int8Features features_8_16 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
	features_8_16.shaderFloat16 = true;
	features_8_16.shaderInt8 = true;
	features8.pNext = &features_8_16;

#ifndef __APPLE__
	VkPhysicalDeviceMaintenance4Features maintenance4_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES};
	maintenance4_features.maintenance4 = true;
	features_8_16.pNext = &maintenance4_features;
#endif


	VkDevice device{};
	VK_CHECK(vkCreateDevice(gpu, &device_info, 0, &device));
	printf("Volk Load\n");
	volkLoadDevice(device);
	return  { gpu, device};
}

void destroy_compute_pipeline(VkDevice device, Pipeline *pipelines, u32 n_pipelines){

	vkDeviceWaitIdle(device);
	for(u32 u{0}; u < n_pipelines; ++u){
		vkDestroyPipelineLayout(device, pipelines[u].layout, 0);
		vkDestroyPipeline(device, pipelines[u].pipeline, 0);
		vkDestroyDescriptorSetLayout(device, pipelines[u].descriptor_set_layout, 0);
	}
	delete[] pipelines;

}

void create_surface(VkSurfaceKHR& surface, VkInstance instance, GLFWwindow* window) 
{
#ifdef __APPLE__
	/* Don't deal with cocoa header
	VkMacOSSurfaceCreateInfoMVK surf_info{ VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK };
    surf_info.pView = (void *)glfwGetCocoaWindow(window); 
    VK_CHECK(vkCreateMacOSSurfaceMVK(instance, &surf_info, 0, &surface));
	*/
	glfwCreateWindowSurface(instance, window, nullptr, &surface);
#endif
#ifdef __linux__
	VkXcbSurfaceCreateInfoKHR surf_info{VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR};
	surf_info.connection = XGetXCBConnection(glfwGetX11Display());
	surf_info.window = glfwGetX11Window(window);
	VK_CHECK(vkCreateXcbSurfaceKHR(instance,&surf_info, 0, &surface));

#endif
#ifdef _WIN32
	VkWin32SurfaceCreateInfoKHR surf_info{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
	surf_info.hinstance = GetModuleHandle(0);
	surf_info.hwnd = glfwGetWin32Window(window);

	VK_CHECK(vkCreateWin32SurfaceKHR(instance, &surf_info, 0, &surface));
#endif
}


void setup_format(VkFormat& format, VkSurfaceFormatKHR& surface_format, VkPhysicalDevice gpu, VkSurfaceKHR surface)
{
	u32 count;
	vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, 0);
	VkSurfaceFormatKHR formats[255];
	vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, formats);

	format = formats[0].format;
    VkColorSpaceKHR  colorspace; 
	colorspace = formats[0].colorSpace;
	surface_format = formats[0];

	/*  If we want a special format:
	for(u32 i{0}; i<count; i++)
	{
		printf("Format[%i] = %i\n", i, formats[i].format);
		if(formats[i].format == VK_FORMAT_B8G8R8A8_UNORM){
			format = VK_FORMAT_B8G8R8A8_UNORM;
			printf("Taking format %i found at %i\n", format, i);
			break;
		}
	}
	if(format == 0){
		format = formats[0].format;
		printf("Taking first format because requested not found\n");
	}
	*/	
		
	vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count,0);
	VkPresentModeKHR modes[8];
	VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, modes));
}

void resize(vulkan_app* vk)
{
	create_swapchain(vk->sfv, vk->gpu, vk->device, vk->surface_format, vk->surface, vk->renderpass);
	vk->viewport = {0,static_cast<float>(vk->sfv.h), static_cast<float>(vk->sfv.w), -static_cast<float>(vk->sfv.h), 0, 1};
	vk->scissors = {{0,0},{vk->sfv.w,vk->sfv.h}};
	printf("Resize successful. New size: (%u, %u).\n", vk->sfv.w, vk->sfv.h);
}

void check_resize(vulkan_app& vk)
{
	VkSurfaceCapabilitiesKHR caps;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.gpu, vk.surface, &caps));
	
	if((caps.currentExtent.width != vk.sfv.w) || (caps.currentExtent.height != vk.sfv.h))
	{
		vkDeviceWaitIdle(vk.device);
		resize(&vk);
		vkDeviceWaitIdle(vk.device);
	}
}

void set_fence(vulkan_app& vk)
{
	auto fence_res = vkWaitForFences(vk.device, 1, &vk.fence, 1, 10'000'000'000);
	if (fence_res != VK_SUCCESS)
	{
		if (fence_res == VK_TIMEOUT) printf("Fence Timeout\n");
		if (fence_res == VK_ERROR_OUT_OF_HOST_MEMORY) printf("Fence Out of Host Memory\n");
		if (fence_res == VK_ERROR_OUT_OF_DEVICE_MEMORY) printf("Fence Out of Device Memory\n");
		if (fence_res == VK_ERROR_DEVICE_LOST) printf("Fence Device Lost\n");
		VK_CHECK(fence_res);
	}
	VK_CHECK(vkResetFences(vk.device, 1, &vk.fence));
}

u32 acquire_swapchain(vulkan_app& vk)
{
	u32 image_index{0};
	VkResult res = vkAcquireNextImageKHR(vk.device, vk.sfv.swapchain, ~uint64_t(0), vk.semaphore_1, 0, &image_index);
	if(res != VK_SUCCESS) 
	{
		printf("Acquire call failed %d\n", res);
		VK_CHECK_SWAPCHAIN(res);
		vkDeviceWaitIdle(vk.device);
	}
	return image_index;
}

void submit_draw(vulkan_app& vk, VkCommandBuffer* buf)
{
	VkPipelineStageFlags mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submit_info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &vk.semaphore_1;
	submit_info.pWaitDstStageMask = &mask;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = buf;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &vk.semaphore_2;
	VK_CHECK(vkQueueSubmit(vk.queue, 1, &submit_info, vk.fence));
}

void present(vulkan_app& vk, u32* image_index)
{
	VkPresentInfoKHR pres_info{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	pres_info.swapchainCount = 1;
	pres_info.pSwapchains = &vk.sfv.swapchain;
	pres_info.pImageIndices = image_index;
	pres_info.waitSemaphoreCount = 1;
	pres_info.pWaitSemaphores = &vk.semaphore_2;

	auto res = vkQueuePresentKHR(vk.queue, &pres_info);

	if (res != VK_SUCCESS)
	{
		printf("Present call failed. %d\n", res);
		VK_CHECK_PRESENT(res);
		vkDeviceWaitIdle(vk.device);
		resize(&vk);
	}
}

void begin_renderpass(vulkan_app& vk, VkCommandBuffer& buf, VkFramebuffer framebuffer)
{
		VkClearColorValue color = { 0.04,0.21,0.1,1. };
		VkClearDepthStencilValue depth = { 1.,0 };
		VkClearValue clear[2];
		clear[0].color = color;
		clear[1].depthStencil = depth;

		VkRenderPassBeginInfo pass_info{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		pass_info.renderPass = vk.renderpass;
		pass_info.framebuffer = framebuffer;
		pass_info.renderArea = { {0,0},{vk.sfv.w,vk.sfv.h} };
		pass_info.clearValueCount = 2;
		pass_info.pClearValues = clear;

		VkSubpassContents contents{ VK_SUBPASS_CONTENTS_INLINE };

		vkCmdBeginRenderPass(buf, &pass_info, contents);

		vkCmdSetViewport(buf, 0, 1, &vk.viewport);
		vkCmdSetScissor(buf, 0, 1, &vk.scissors);
}

template <class push_buffer_type>
void draw_vertex_attributes(VkCommandBuffer& buf, Pipeline& p, Buffer vertices, Buffer indicies, u32 count, push_buffer_type& push_buffer)
{
	vkCmdPushConstants(buf, p.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_buffer_type), &push_buffer);
	vkCmdBindDescriptorSets(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &p.uniform_desc.set, 0, 0);
	vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
	VkDeviceSize buff_offsets{ 0 };
	vkCmdBindVertexBuffers(buf, 0, 1, &vertices.buffer, &buff_offsets);
	vkCmdBindIndexBuffer(buf, indicies.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(buf, count, 1, 0, 0, 0);
}

template <class push_buffer_type>
void draw_vertex_pull(VkCommandBuffer& buf, Pipeline& p, Buffer vertices, u32 count, push_buffer_type& push_buffer)
{
	VkDescriptorBufferInfo info;
	info = { vertices.buffer, 0, vertices.size };
	VkWriteDescriptorSet write_dset{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	write_dset.dstBinding = 0;
	write_dset.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write_dset.descriptorCount = 1;
	write_dset.pBufferInfo = &info;

	vkCmdPushConstants(buf, p.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_buffer_type), &push_buffer);
	vkCmdPushDescriptorSetKHR(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &write_dset);
	vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
	vkCmdDraw(buf, count, 1, 0, 0);
}

void compute_dispatch(VkCommandBuffer& buf, Pipeline& pipeline, const compute_shader_data& compute, bool block_final = false)
{
	for (u32 i{ 0 }; i < compute.n_dispatches; ++i)
	{
		vkCmdDispatch(buf, compute.n_work_groups_x,  compute.n_work_groups_y, compute.n_work_groups_z); 
		if (i + 1 < compute.n_dispatches || block_final)
		{
			vkCmdPipelineBarrier(
				buf,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				compute.n_buffers, compute.barriers,
				0, nullptr
			);
		}
	}
}

template <class push_buffer_type>
void compute_prepare(VkCommandBuffer& buf, Pipeline& pipeline, push_buffer_type& push_buffer, const compute_shader_data& compute)
{

	auto set  = new VkWriteDescriptorSet[compute.bindings.size];
	auto info = new VkDescriptorBufferInfo[compute.bindings.size];

	for (int i{ 0 }; i < compute.bindings.size; ++i)
	{
		info[i] = {compute.buffers[i].buffer, 0, compute.buffers[i].size};
		set[i] = {};
		set[i].sType = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		set[i].dstBinding = compute.bindings.id[i];
		set[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		set[i].descriptorCount = 1;
		set[i].pBufferInfo = info+i;
	}

	vkCmdPushConstants(buf, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_buffer_type), &push_buffer);
	vkCmdPushDescriptorSetKHR(buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, compute.bindings.size, set);

	vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

	delete[] set;
	delete[] info;
}

void test_for_memory(vulkan_app& vk, u64 min_needed = 0)
{

	VkPhysicalDeviceMaintenance4Properties props4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
	VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	props.pNext = &props4;
	vkGetPhysicalDeviceProperties2(vk.gpu, &props);

	printf("Maximal Buffersize: %u\n", props4.maxBufferSize);


	printf("Requested Buffersize: %u\n", min_needed);

	if(props4.maxBufferSize < min_needed)
	{
		printf("Too many points. Reduce points to less than %u\n", props4.maxBufferSize);
	}

}
