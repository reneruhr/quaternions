#include <cassert>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>

#include "common.h"
#include "quaternions.h"
#include "random_int.h"


#include "vulkan/load_file.cpp"
#include "vulkan/statistics.cpp"

std::filesystem::path shader_folder {"shaders"};
std::filesystem::path compute_folder {"compute"};
std::filesystem::path data_folder {"data"};

#include "vulkan/vk_backend.cpp"

struct test_state
{
	bool test      { true  };
	bool init	   { false };
	bool cpu_test  { false };
} test_state;


//VkPhysicalDeviceType prefered_gpu = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
VkPhysicalDeviceType prefered_gpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

vulkan_app init_vulkan()
{
	volkInitialize();

	auto instance	= create_instance();
	auto gpu_device	= create_device(instance, prefered_gpu);
	auto gpu		= gpu_device.gpu;
	auto device		= gpu_device.device;

	auto queue		= VkQueue{};
	vkGetDeviceQueue(device, 0, 0, &queue);

	auto pool		= make_pool(device);

	auto fence		= create_fence(device);
	auto mem_props  = get_memory_properties(gpu);

	print_compute_device_properties(gpu);

	vulkan_app app =
	{
		.instance = instance,
		.gpu = gpu,
		.device = device,
		.queue = queue,
		.pool = pool,
		.fence = fence,
		.mem_props = mem_props,
	};
	
	return app;
}

struct algorithm
{
	const char* label;
	u32 id;
};

struct experiment
{
	const char* name{};

	algorithm algorithm{};

	u32 workload{};
	u32 iterations{};

	u32 count{};
	u32 count_caps{};

	f64 *discrepancies{};
	u64 *times{};

	compute_shader_data compute{};
	compute_shader_data compute_sum_quats{};
	compute_shader_data compute_sum_caps{};
};

enum class workload : u32 { write = 0 , discrepancy = 1};
const u32 n_samples{ 100 };
const u32 n_warmup { 10 };

void prepare_experiment(vulkan_app& vk, Pipeline** pipelines, u32& n_experiments, experiment **experiments)
{

	u32 counts[] = { 1 << 20, 1 << 21 };
	u32 counts_caps[]   = { 1<<9 };  
	// If n_points or n_caps is 64 there is a validation error. I presume this is a bug.

  	u32 n_count 	 =  sizeof(counts) / 4;
	u32 n_count_caps =  sizeof(counts_caps) / 4;

	if(test_state.cpu_test)
	{
		n_count = 1;
		n_count_caps = 1;	
		counts[0] = 1 << 16;
		counts_caps[0] = 1 << 7;
	}

	u32 local_size_x = 128;
	u32 local_size_y =   8;


	u32 workloads[]  = { 1 };
	u32 iterations[] = { local_size_x * local_size_y};

	u32 max_count = counts[sizeof(counts)/4-1];
	u32 max_count_caps = counts_caps[sizeof(counts_caps)/4-1];

	test_for_memory(vk, max_count * sizeof(quat)  + (u64)max_count*max_count_caps*4*sizeof(f32)/iterations[0] + 2*max_count_caps*sizeof(f32));

	u32 n_buffers = 5;
	auto buffers = new Buffer[n_buffers];
	*buffers = create_device_buffer(vk.device, max_count * sizeof(quat), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk.mem_props, nullptr);                                                                                                               // 0
	*(buffers + 1) = create_device_buffer(vk.device, max_count_caps * sizeof(f32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.mem_props, nullptr);                                                                       // 3
	*(buffers + 2) = create_device_buffer(vk.device, (u64)max_count * max_count_caps * 4 * sizeof(f32) / iterations[0], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk.mem_props, nullptr);                                                                            // 4
	*(buffers + 3) = create_device_buffer(vk.device, max_count_caps * sizeof(f32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.mem_props, nullptr);                                                                       // 5
	*(buffers + 4) = create_device_buffer(vk.device, max_count_caps * sizeof(f32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.mem_props, nullptr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);                      //staging

	vk.n_resources = n_buffers;
	vk.resources   =   buffers;

	Bindings bindings 			= { {0, 3, 4   }, 3 };
	Bindings bindings_sum_quats = { {      4   }, 1 };
	Bindings bindings_sum_caps	= { {   3, 4, 5}, 3 };

	auto barriers = new Barrier[n_buffers]; 
	for(u32 i{}; i<n_buffers; i++)
		barriers[i] = make_barrier(buffers[i]);

	vk.n_resource_barriers = n_buffers;
	vk.resource_barriers   =  barriers;

	algorithm algorithms[]
	{
		{.label = "polar pcg"			,.id =  1},
		{.label = "polar pcg3"			,.id =  2},
		//{.label = "quat walk"			,.id =  16},
		{.label = "quat sphere"			,.id =  18},
		{.label = "quat walk"			,.id =  19},
		{.label = "fibonacci "			,.id =  20},
		{.label = "rejection"			,.id =  30},
	};

#ifdef __APPLE__
	const char* comp_shader			  = "quat.comp-v120.spv";
	const char* comp_shader_sum_quats = "sum.comp-v120.spv";
	const char* comp_shader_sum_caps  = "sum_caps.comp-v120.spv";
#else
	const char* comp_shader			  = "vk_quat.comp.spv";
	const char* comp_shader_sum_quats = "vk_sum.comp.spv";
	const char* comp_shader_sum_caps  = "vk_sum_caps.comp.spv";
#endif

	n_experiments = 		  (u64)n_count 
							* n_count_caps
							* sizeof(algorithms) / sizeof(algorithms[0]) 
							* sizeof(workloads) / 4
							* sizeof(iterations) / 4;

	auto specializations	= new shader_specialization[n_experiments];

	*experiments = new experiment[n_experiments];

	u32 u{};
	for(u32 workload: workloads)
	for(auto algo   : algorithms)
	for(u32 iteration: iterations)
	for(u32 count : counts)
	for(u32 count_caps : counts_caps)
	{
		if(workload == static_cast<u32>(workload::discrepancy))
			count /= iteration;

		u32 n_work_groups_x = (count + local_size_x - 1) / local_size_x ;
		u32 n_work_groups_y = (count_caps / 4 + local_size_y-1) / local_size_y; 

		specializations[u] = 
		{ 
			.algorithm  = algo.id,
			.workload   = workload,
			.iterations = iteration,
			.n_points   = count,
			.n_caps     = count_caps / 4,
		};

		(*experiments)[u] =
		{
			.algorithm = algo,

			.workload  = workload,
			.iterations= iteration,

			.count = count,
			.count_caps = count_caps / 4,

			.compute = {
						.n_work_groups_x  = n_work_groups_x,
						.n_work_groups_y  = n_work_groups_y,
						.n_work_groups_z  = 1,
						.local_size_x	  = 128,
						.local_size_y     = 8,
						.local_size_z     = 1,
						.n_buffers        = 3,
						.buffers 		  = buffers,
						.bindings         = bindings,
						.barriers         = barriers,
						.n_dispatches 	  = 1,
						.specialization   = specializations+u
						},

			.compute_sum_quats = {
						.n_work_groups_x  = (count+511) / 512,
						.n_work_groups_y  = count_caps,
						.n_work_groups_z  = 1,
						.local_size_x	  = 512,
						.local_size_y     = 1,
						.local_size_z     = 1,
						.n_buffers        = 1,
						.buffers 		  = buffers+2,
						.bindings         = bindings_sum_quats,
						.barriers         = barriers+2,
						.n_dispatches 	  = 1,
						.specialization   = specializations+u
						},

			.compute_sum_caps = {
						.n_work_groups_x  = 1,
						.n_work_groups_y  = (count_caps+63)/64,
						.n_work_groups_z  = 1,
						.local_size_x	  = 1,
						.local_size_y     = 64,
						.local_size_z     = 1,
						.n_buffers        = 3,
						.buffers 		  = buffers+1,
						.bindings         = bindings_sum_caps,
						.barriers         = barriers+1,
						.n_dispatches 	  = 1,
						.specialization   = specializations+u
						}
		};

		u++;
	}
	
	*pipelines = new Pipeline[3*n_experiments];
	create_pipeline_compute(vk.device, *pipelines,        		     comp_shader		   , n_experiments, bindings, 		   specializations, u32{});
	create_pipeline_compute(vk.device, *pipelines +  n_experiments , comp_shader_sum_quats , n_experiments, bindings_sum_quats, specializations, u32{});
	create_pipeline_compute(vk.device, *pipelines + 2*n_experiments, comp_shader_sum_caps  , n_experiments, bindings_sum_caps,  specializations, u32{});

}

void parse_input(int argc, char** args);
void write_to_file(experiment_records& records, experiment *experiments);
void clean_experiment(vulkan_app& vk, Pipeline *p, experiment *e,  u32 n);

struct timing
{
    const char* label;
    u32 n{};
    u32 n_reps{};
    f64* times{};
    f32* discrepancies{};
};

void write(timing timings, const char* suffix)
{ 
    if (!timings.n) return;
    std::string file_name = std::string("vk_compute_timings_") + suffix + "_" + timings.label + ".bin";
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 m  = (u32)strlen(timings.label);
	fwrite(&m,                    sizeof(u32), 1, file);
	fwrite(timings.label,   	  sizeof(char), strlen(timings.label), file);
	fwrite(&timings.n,      	  sizeof(u32), 1, file);
	fwrite(&timings.n_reps, 	  sizeof(u32), 1, file);
	fwrite(timings.times,   	  sizeof(f64), timings.n_reps, file);
	fwrite(timings.discrepancies, sizeof(f32), timings.n_reps, file);
    fclose(file);
}

int main(int argc, char** args)
{
	// Vulkan

	auto vk = init_vulkan();

	experiment *experiments{};
	u32 n_experiments{};
	Pipeline* pipelines{};
	VkBufferMemoryBarrier *barriers{};
	VkCommandBuffer copy_cmd_buffer{};
	void * copy_data;

	experiment_records records{};

	u64 frame{};

	const f32 ns_per_tick = ticks_per_nanosecond(vk.gpu);
	if (ns_per_tick== 0.f)
		printf("\n\n\nWarning: No timestamp query support.\n\n\n");
	else
		printf("GPU Ticks per nanosecond: %f\n", ns_per_tick);

	if (not test_state.init)
	{
		prepare_experiment(vk, &pipelines, n_experiments, &experiments);

		u32 n_compute_buffers = n_experiments;
		
		vk.n_buffers  = n_compute_buffers;
		vk.buffers	= new VkCommandBuffer[n_compute_buffers];
		allocate_cmd_buffers(vk.device, vk.n_buffers, vk.buffers, vk.pool);

		records.n_experiments = n_experiments;
		records.results = new experiment_result[n_experiments];
		records.info    = new experiment_info[n_experiments];

		vk.query_pool = create_query_pool(vk.device, 2*n_compute_buffers);
		for (u32 u{}; u < n_experiments; u++)
		{

			records.results[u].n_samples = n_samples;
			records.results[u].times = new f64[n_samples];
			records.results[u].times2 = new f64[n_samples];
			
			experiments[u].discrepancies = new f64[n_samples];
		}

		allocate_cmd_buffers(vk.device, 1, &copy_cmd_buffer, vk.pool);
		VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		VK_CHECK(vkBeginCommandBuffer(copy_cmd_buffer, &begin_info));
		VkBufferCopy region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size      = 64*sizeof(vec4_f32)
		};
		vkCmdCopyBuffer(copy_cmd_buffer, vk.resources[3].buffer, vk.resources[4].buffer, 1, &region);
		VK_CHECK(vkEndCommandBuffer(copy_cmd_buffer));
		
		test_state.init = true;
	}

	chrono_timer timer("Experiment");
	u64 query_times[2];
	u32 sample{};
	u32 current_experiment{};

	VkSubmitInfo submit_info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submit_info.waitSemaphoreCount = 0;
	submit_info.commandBufferCount = 1;
	submit_info.signalSemaphoreCount = 0;

	set_fence(vk);
	while (test_state.test)
	{

		submit_info.pCommandBuffers = vk.buffers+current_experiment;


		timer.start();
		{
			u32 u = current_experiment;
			auto& buffer = vk.buffers[u];
			u32 push_buffer{};
			auto experiment = experiments[u];
			auto pipeline = pipelines[u];

			vkResetCommandBuffer(buffer, 0);
			VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			VK_CHECK(vkBeginCommandBuffer(buffer, &begin_info));
			compute_prepare(buffer, pipeline, sample, experiment.compute);
			vkCmdResetQueryPool(buffer, vk.query_pool, 2 * u, 2);
			vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk.query_pool, 2 * u + 0);
			compute_dispatch(buffer, pipeline, experiment.compute, true);
			vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk.query_pool, 2 * u + 1);

			auto pipeline_sum = pipelines[u + n_experiments];
			u32 offset = experiment.count / 2;
			compute_prepare(buffer, pipeline_sum, push_buffer, experiment.compute_sum_quats);
			while (offset >= 512)
			{
				vkCmdPushConstants(buffer, pipeline_sum.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32), &offset);
				compute_dispatch(buffer, pipeline_sum, experiment.compute_sum_quats, true);
				offset /= 2;
			}
			auto pipeline_sum_caps = pipelines[u + 2 * n_experiments];
			offset = experiment.count_caps / 2;
			compute_prepare(buffer, pipeline_sum_caps, push_buffer, experiment.compute_sum_caps);
			while (offset >= 64)
			{
				vkCmdPushConstants(buffer, pipeline_sum_caps.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32), &offset);
				compute_dispatch(buffer, pipeline_sum_caps, experiment.compute_sum_caps, true);
				offset /= 2;
			}

			VK_CHECK(vkEndCommandBuffer(buffer));
		}

		if(sample == 0)
			for(u32 i{}; i<n_warmup; i++)
			{
				VK_CHECK(vkQueueSubmit(vk.queue, 1, &submit_info, vk.fence));
				set_fence(vk);
			}

		VK_CHECK(vkQueueSubmit(vk.queue, 1, &submit_info, vk.fence));
		set_fence(vk);

		submit_info.pCommandBuffers = &copy_cmd_buffer;
		VK_CHECK(vkQueueSubmit(vk.queue, 1, &submit_info, vk.fence));
		set_fence(vk);

		auto& experiment = experiments[current_experiment];

		VK_CHECK(vkMapMemory(vk.device, vk.resources[4].memory, 0, 64*sizeof(vec4_f32), 0, &copy_data));
		f32* d2 = (f32*)copy_data;
		f32* d2_data = new f32[1024];
		memcpy(d2_data, copy_data, 1024);
		experiment.discrepancies[sample] = std::sqrt(  ((f64)d2[0] + d2[1] + d2[2] + d2[3]) / ( 2 * experiment.count_caps ));
		//printf("%f\n", experiment.discrepancies[sample]);
		vkUnmapMemory(vk.device, vk.resources[4].memory);


		f64 cpu_time_ns = (f64)timer.stop() / experiment.compute.n_dispatches;

		VK_CHECK(vkGetQueryPoolResults(vk.device, vk.query_pool, 2*current_experiment, ARRAYSIZE(query_times), sizeof(query_times), query_times, sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
		f64 gpu_time_ns = (query_times[1] - query_times[0]) * (f64)ns_per_tick / experiment.compute.n_dispatches;

		auto& results = records.results[current_experiment];
		results.times [sample] = gpu_time_ns;
		results.times2[sample] = cpu_time_ns;


		sample++;
		if (sample == n_samples) 
		{
			u32 u = current_experiment;
			auto [mean,err, mi, ma] = mean_statistics(experiment.discrepancies, n_samples);

			char* label = new char[256];
			snprintf(label, 256, "Algo: %u. NxI: %u. Caps: %u. Iterations %u\n Discrepancy: %f (%f) [%f,%f]",
				experiments[u].algorithm.id, experiments[u].count * experiments[u].iterations,
				experiments[u].count_caps * 4, experiments[u].iterations,
				mean, err, mi, ma);

			records.info[u].label = label;
			records.info[u].count = (u64)experiments[u].count * experiments[u].count_caps * 4;
			records.info[u].iterations = experiments[u].iterations; 
			records.info[u].size  = sizeof(f32);

			auto t_label = std::to_string(experiments[u].algorithm.id);
			timing timing = { .label = t_label.c_str(), .n = experiments[u].count*experiments[u].iterations, .n_reps = n_samples, .times = results.times};
			timing.discrepancies = new f32[n_samples];
			for (u32 t{}; t < n_samples; t++)
				timing.discrepancies[t] = (f32)experiments[u].discrepancies[t];
			auto t_suffix = std::to_string(timing.n);
			if (prefered_gpu == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
				t_suffix += "_integrated";
			write(timing, t_suffix.c_str());

			sample = 0;
			current_experiment++;
		}

		if (current_experiment == n_experiments) test_state.test = false;
	}


	print_statistics(records);
	write_to_file(records, experiments);



	// Clean
	vkDeviceWaitIdle(vk.device);
	clean_experiment(vk, pipelines, experiments, n_experiments);
	vkDestroyFence(vk.device,vk.fence,0);
	vkDestroyQueryPool(vk.device, vk.query_pool,0);
	vkFreeCommandBuffers(vk.device, vk.pool, vk.n_buffers, vk.buffers);
	vkDestroyCommandPool(vk.device, vk.pool, 0);
	vkDestroyDevice(vk.device, 0);
	vkDestroyInstance(vk.instance, 0);
	delete[] vk.buffers;

	return 0;
}













void parse_input(int argc, char** args)
{
	for (int i{ 1 }; i < argc; ++i)
	{
		if (args[i][0] == 'g' && args[i][1] == '=')
		{
			if (!strcmp("gpu", args[i] + 2)) {
				prefered_gpu = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
				printf("Requested the discrete GPU\n");
			}
			else if (!strcmp("cpu", args[i] + 2))
			{
				prefered_gpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
				printf("Requested the integrated GPU\n");
			}
			else
			{
				printf("Usage: g=gpu or g=cpu\n");
				exit(1);
			}
		}
	}
}

void write_to_file(experiment_records& records, experiment* experiments)
{
/*
	auto file_name = std::string("gpu_timings.bin");
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");

	fwrite(&records.n_experiments, sizeof(u32), 1, file);

	for (u32 u{}; u < records.n_experiments; u++)
	{
		auto results = records.results[u];
		auto experiment = experiments[u];
		u32 m = strlen(experiment.label);

		fwrite(&m, sizeof(u32), 1, file);
		fwrite(experiment.label, sizeof(char), m, file);

		fwrite(&experiment.id,			sizeof(u32), 1, file);
		fwrite(&experiment.n_points, sizeof(u32), 1, file);
		fwrite(&experiment.dispatches, sizeof(u32), 1, file);
		fwrite(&experiment.iterations, sizeof(u32), 1, file);

		fwrite(&results.n_samples, sizeof(u32), 1, file);
		fwrite(results.times, sizeof(u64), results.n_samples, file);
	}
	fclose(file);
*/
}

void clean_experiment(vulkan_app& vk, Pipeline *p, experiment *e, u32 n)
{
	destroy_compute_pipeline(vk.device, p, 3*n);
	for(u32 u{0}; u<vk.n_resources; ++u){
		destroy_buffer(vk.device, vk.resources[u]);
	}
	delete[] vk.resources; 
	delete[] vk.resource_barriers; 
	delete[] e;
}