#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <utility>
#include <numbers>
#include <filesystem>
#include <string>
#include <span>

#include "random_int.h"
#include "quaternions.h"
#include "common.h"
#include "vec_math.h"

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#include "render/gl_backend.cpp"
#include "render/load_textfile.cpp"
#include "discrepancy/statistics.cpp"


// experiment

const u32 n_samples{ 1000};
const u32 n_warmup {  10};

u32 counts[]	    = { 1 << 20, 1 << 22, 1 << 24};
u32 counts_caps[]   = { 1 << 8 };

u32 local_sizes_x[] = { 128 };
u32 local_sizes_y[] = {   8 };

u32 count_caps = counts_caps[sizeof(counts_caps)/4-1];

u32 workloads[] = { 2 };						//  As defined in shader quat.comp
u32 iterations[] = { 1024 };
u32 algorithms[] = { 1, 2, 18, 19, 20, 30 };
u32 count = counts[sizeof(counts)/4-1]/iterations[0];

struct experiment
{
	char* name{};

	u32 count{};
	u32 count_caps{};

	u32 workload{};
	u32 algorithm{};
	u32 iterations{};

	f64 discrepancies[n_samples];
	u64 times[n_samples];
};

u32 n_compute{};
opengl_context* compute; 
experiment* experiments; 

storage_info* storages; 
u32 n_storages;
void load_buffers();

std::filesystem::path compute_folder = "compute";
std::filesystem::path data_folder    = "data";
void load_shaders(opengl_context** compute, experiment** experiments, storage_info* storages, u32 n_storages);

// events 

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

struct key_event { int key; int action; };
const u32 n_queue{ 1024 };
struct events
{
	u32 begin{};
	u32 end{};
	key_event queue[n_queue];
} events{};
void register_event(key_event event);
bool has_events();
key_event poll_event();
void event_loop();

const GLuint WIDTH = 2*800, HEIGHT = 2*600;

struct uniform0
{
	u32 n_points;
	u32 n_caps;
	u32 offset;
	u32 iterations;
	u32 frame;
	u32 padding[3];
};

void load_buffers()
{

	storage_info storages_[] =
	{
		{.binding = 0, .size = count * sizeof(quat),	.data = 0 },         // Quats
		{.binding = 1, .size = count * sizeof(u32),		.data = 0 },         // Seeds

		{.binding = 4, .size = count * count_caps/4 * sizeof(vec4_f32),	.data = 0},    // Sum 
		{.binding = 3, .size =		   count_caps/4	* sizeof(vec4_f32), .data = 0},    // Cap Volume

		{.binding = 6, .size = count * count_caps/4 * sizeof(vec4_f32),	.data = 0},    // Copy to Sum
		{.binding = 5, .size =		   count_caps/4 * sizeof(vec4_f32),	.data = 0},    // Discrepancy 
	};

	n_storages = sizeof(storages_) / sizeof(storage_info);
	storages = new storage_info[n_storages];
	memcpy(storages, storages_, sizeof(storages_));

	for (u32 u{}; u < n_storages; u++)
	{
		storages[u].buffer = buffer_storage(storages[u].size, storages[u].data, GL_DYNAMIC_STORAGE_BIT);
		bind_storage_buffer(storages[u].buffer, storages[u].binding);
	}
}

void load_shaders(opengl_context** compute, experiment** experiments, storage_info* storages, u32 n_storages)
{
	auto  set_placeholders = [](std::string& input, const std::string& target, const std::string& replacement)
		{
			size_t start = 0;
			while ((start = input.find(target, start)) != std::string::npos) {
				input.replace(start, target.length(), replacement);
				start += replacement.length();
			}
		};

	auto path = compute_folder / "quat.comp";
	auto s = read_file(path.string().c_str());

	n_compute =     sizeof(counts) / 4 
				  * sizeof(local_sizes_x) / 4
				  * sizeof(local_sizes_y) / 4
				  * sizeof(counts_caps) / 4
				  * sizeof(algorithms) / 4
				  * sizeof(workloads) / 4
				  * sizeof(iterations) / 4;



	*compute     = new opengl_context[n_compute + 2];
	*experiments = new experiment[n_compute];

	u32 u{};
	for(u32 workload: workloads)
	for(u32 algorithm: algorithms)
	for(u32 iteration: iterations)
	for(u32 count : counts)
	for(u32 count_cap : counts_caps)
	for(u32 local_size_x: local_sizes_x)
	for(u32 local_size_y: local_sizes_y)
	{
		if (u == n_compute) break;
		
		count /= iteration;

		assert(count / (local_size_x * local_size_y) > 0 && "Not enough points for workgroup size.");

		u32 local_size_z = 1;

		u32 n_work_groups_x = (count + local_size_x - 1) / local_size_x ;
		u32 n_work_groups_y = (count_cap / 4 + local_size_y-1) / local_size_y; 

		auto source = std::string(s.data);

		set_placeholders(source, "N_LOCAL_SIZE_X", std::to_string(local_size_x));
		set_placeholders(source, "N_LOCAL_SIZE_Y", std::to_string(local_size_y));
		set_placeholders(source, "N_LOCAL_SIZE_Z", std::to_string(local_size_z));
		set_placeholders(source, "algorithm", std::to_string(algorithm));
		set_placeholders(source, "workload", std::to_string(workload));
		
		if (u == 0)
			printf("Compiled %s:\n%s", path.string().c_str(), source.c_str());
		(*compute)[u] =
		{
			.program		= compile_shaders((GLchar*)source.c_str()),
			.count			= count,
			.n_storages	    = 4,
			.storages       = storages,
			.n_work_groups_x  = n_work_groups_x,
			.n_work_groups_y  = n_work_groups_y,
			.n_work_groups_z  = 1,
			.local_size_x   = local_size_x,
			.local_size_y	= local_size_y,
			.local_size_z	= 1,
		};

		(*experiments)[u] =
		{
			.count = count,
			.count_caps = count_cap / 4,
			.workload = workload,
			.algorithm = algorithm,
			.iterations= iteration,
		};

		u++;
	}

	auto path_sum = compute_folder / "sum.comp";
	auto s_sum = read_file(path_sum.string().c_str());
	auto source_sum = std::string(s_sum.data);

	u32 local_size_x = 1024;
	u32 local_size_y = 1;
	set_placeholders(source_sum, "N_LOCAL_SIZE_X", std::to_string(local_size_x));
	set_placeholders(source_sum, "N_LOCAL_SIZE_Y", std::to_string(local_size_y));
	set_placeholders(source_sum, "N_LOCAL_SIZE_Z", std::to_string(1));

	printf("Compiled %s:\n%s", path_sum.string().c_str(), source_sum.c_str());
	(*compute)[n_compute] =
	{
		.program		= compile_shaders((GLchar*)source_sum.c_str()),
		.n_storages	    = 1,
		.storages       = storages+4,
		.local_size_x   = local_size_x,
		.local_size_y	= local_size_y,
		.local_size_z	= 1,
	};

	auto path_sum_caps = compute_folder / "sum_caps.comp";
	auto s_sum_caps = read_file(path_sum_caps.string().c_str());
	auto source_sum_caps = std::string(s_sum_caps.data);

	local_size_y = 128;
	set_placeholders(source_sum_caps, "N_LOCAL_SIZE_X", std::to_string(1));
	set_placeholders(source_sum_caps, "N_LOCAL_SIZE_Y", std::to_string(local_size_y));
	set_placeholders(source_sum_caps, "N_LOCAL_SIZE_Z", std::to_string(1));

	printf("Compiled %s:\n%s", path_sum_caps.string().c_str(), source_sum_caps.c_str());

	(*compute)[n_compute+1] =
	{
		.program		= compile_shaders((GLchar*)source_sum_caps.c_str()),
		.n_storages	    = 3,
		.storages       = storages+3,
		.local_size_x   = 1,
		.local_size_y	= local_size_y,
		.local_size_z	= 1,
	};
}

void print_gpu_buffer(opengl_context* compute, experiment* experiments, u32 u);



struct timing
{
    const char* label;
    u32 n{};
    u32 n_reps{};
    u64* times{};
    f32* discrepancies{};
};

void write(timing timings, const char* suffix)
{ 
    if (!timings.n) return;
    std::string file_name = std::string("compute_timings_") + suffix + timings.label + ".bin";
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 m  = (u32)strlen(timings.label);
	fwrite(&m,                    sizeof(u32), 1, file);
	fwrite(timings.label,   	  sizeof(char), strlen(timings.label), file);
	fwrite(&timings.n,      	  sizeof(u32), 1, file);
	fwrite(&timings.n_reps, 	  sizeof(u32), 1, file);
	fwrite(timings.times,   	  sizeof(u64), timings.n_reps, file);
	fwrite(timings.discrepancies, sizeof(f32), timings.n_reps, file);
    fclose(file);
}



int main(void)
{
	assert(count * sizeof(vec4_f32) < INT_MAX / (count_caps / 4) && "Requested storage is too large.");

	auto window = create_window(WIDTH, HEIGHT, "Compute example");

	glfwSetKeyCallback(window, key_callback);

	opengl_constants gl_constants;
	query_opengl_constants(gl_constants);
	printf("Shared Memory size: %i\n", gl_constants.shared_memory_size);
	printf("Work group count: %i x %i x %i \n", gl_constants.work_group_count_x, gl_constants.work_group_count_y, gl_constants.work_group_count_z);
	printf("Local sizes: %i x %i x %i  (Total <= %i)\n", gl_constants.local_size_x, gl_constants.local_size_y, gl_constants.local_size_z, gl_constants.local_size);
	printf("Storage block size: %i\n", gl_constants.storage_block_size);

	// Compute

	load_buffers();
	load_shaders(&compute, &experiments, storages, n_storages);

	u32 uniform = create_buffer(sizeof(uniform0), GL_DYNAMIC_DRAW);

	bind_uniform_block(uniform, 10);

	// Profiler

	opengl_profiler profiler{};
	init(&profiler);
	profiler.n_data = n_samples+n_warmup;
	profiler.data = new u64[n_samples+n_warmup];

	u32 query;
	glGenQueries(1, &query);

	u32 u = 0;


	u32 n_frames = n_samples;
	uniform0 uniform0;
	
    while (!glfwWindowShouldClose(window)) 
	{
		// Input 
		event_loop();

		// Compute
		if (u < n_compute)
		{
			u32 sample{};
			while (sample < n_samples + profiler.lag_query + n_warmup)
			{
				uniform0 = { compute[u].count, experiments[u].count_caps, 0, experiments[u].iterations, sample-n_warmup};
				glUseProgram(compute[u].program);
				map_buffer(&uniform0, sizeof(uniform0), uniform);

				profile_compute(compute + u, &profiler);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			
				if (experiments[u].workload > 0)
				{
					glCopyNamedBufferSubData(compute[u].storages[2].buffer, compute[n_compute].storages[0].buffer, 0, 0, compute[u].count * experiments[u].count_caps * sizeof(vec4_f32));
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					u32 offset = experiments[u].count / 2;
					glUseProgram(compute[n_compute].program);
					while (offset > 0)
					{
						u32 local_size_x = compute[n_compute].local_size_x;
						u32 local_size_y = compute[n_compute].local_size_y;
						uniform0.offset = offset;
						map_buffer(&uniform0, sizeof(uniform0), uniform);
						glDispatchCompute((offset+local_size_x-1)/local_size_x, (experiments[u].count_caps+local_size_y-1)/local_size_y, 1);
						glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
						offset /= 2;
					}

					offset = experiments[u].count_caps / 2;
					glUseProgram(compute[n_compute + 1].program);
					while (offset > 0)
					{
						u32 local_size = compute[n_compute + 1].local_size_y;
						uniform0.offset = offset;
						map_buffer(&uniform0, sizeof(uniform0), uniform);
						glDispatchCompute(1, (offset + local_size-1)/local_size, 1);
						glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
						offset /= 2;
					}

					vec4_f32 d2v;
					gpu_to_cpu(compute[n_compute + 1].storages[2].buffer, &d2v, sizeof(vec4_f32));
					//print(d2v);

					f32 d2 = (d2v.x + d2v.y + d2v.z + d2v.w) / (experiments[u].count_caps * 4);
					if(sample < n_samples+n_warmup && sample >= n_warmup)
						experiments[u].discrepancies[sample-n_warmup] = std::sqrt(2.f * d2);
					
					//printf("Id: %u. N=%u. Caps=%u\n", experiments[u].algorithm, experiments[u].count, experiments[u].count_caps*4);
					//printf("D2 = %f\n\n", d2);
				}
				sample++;
			}
			//print_gpu_buffer(compute, experiments, u);


			memcpy(experiments[u].times, profiler.data+n_warmup, sizeof(u64) * n_samples);

			u64 sum{};
			for (u32 i{}; i < n_samples; i++) sum += experiments[u].times[i];
			f64 avg = (f64)(sum) / n_samples;

			f64 sum2{};
			for (u32 i{}; i < n_samples; i++) sum2 += experiments[u].discrepancies[i];
			f64 avg2 = sum2 / n_samples;

			printf("Time/sample: %f ns. D2 = %f.\n", avg/experiments[u].count/experiments[u].iterations, avg2);

			memset(profiler.data, 0, sizeof(u64) * (n_samples+n_warmup));
			assert(profiler.current_query = n_warmup + n_samples + profiler.lag_query);
			profiler.current_query = {};

		}
		else if (u == n_compute)
		{
			experiment_records records{};
			records.n_experiments = n_compute;
			records.info	= new experiment_info[n_compute];
			records.results	= new experiment_result[n_compute];
			for (u32 n{}; n < n_compute; n++)
			{
				records.results[n].n_samples = n_samples;
				records.results[n].times = new f64[n_samples];
				for (u32 t{}; t < n_samples; t++)
					records.results[n].times[t] = experiments[n].times[t];
				

				char* label = new char[256];

				auto [mean,err, mi, ma] = mean_statistics(experiments[n].discrepancies, n_samples);

				snprintf(label, 256, "Algo: %u. NxI: %u. Caps: %u. Iterations %u\n Discrepancy: %f (%f) [%f,%f]",
					experiments[n].algorithm, experiments[n].count * experiments[n].iterations,
					experiments[n].count_caps * 4, experiments[n].iterations,
					mean, err, mi, ma);

				records.info[n].label = label;
				records.info[n].count = experiments[n].count * experiments[n].count_caps * 4; 
				records.info[n].iterations = experiments[n].iterations; 
				records.info[n].size  = sizeof(f32);

				auto t_label = std::to_string(experiments[n].algorithm);
				timing timing = { .label = t_label.c_str(), .n = experiments[n].count*experiments[n].iterations, .n_reps = n_samples, .times = experiments[n].times};
				timing.discrepancies = new f32[n_samples];
				for (u32 t{}; t < n_samples; t++)
					timing.discrepancies[t] = (f32)experiments[n].discrepancies[t];
				auto t_suffix = std::to_string(timing.n);
				write(timing, t_suffix.c_str());
			}
			
			print_statistics(records);
		}
		else
		{
			//glfwSetWindowShouldClose(window, GL_TRUE);
		}

		u++;
		glfwSwapBuffers(window);
    }

    glfwTerminate();



	delete[] profiler.data;

	delete[] storages;

	delete[] compute;
	delete[] experiments;

    return 0;
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	else if (action == GLFW_RELEASE)
		register_event({ key, action });
}

void register_event(key_event event)
{
	events.queue[events.end] = event;
	events.end = (events.end + 1u) % n_queue;
}

bool has_events()
{
	return events.begin != events.end;
}

void event_loop()
{
	glfwPollEvents();

	while (has_events())
	{
		auto event = poll_event();
		if (event.key == GLFW_KEY_R)
		{
		}
		else if (event.key == GLFW_KEY_T)
		{
		}
		else if (event.key == GLFW_KEY_ENTER)
		{
		}
		else if (event.key == GLFW_KEY_RIGHT)
		{
		}
		else if (event.key == GLFW_KEY_LEFT)
		{
		}
		else if (event.key == GLFW_KEY_SPACE)
		{
			load_shaders(&compute, &experiments, storages, n_storages);
		}
		else if (event.key == GLFW_KEY_B)
		{
			load_buffers();
		}
	}
}

key_event poll_event()
{
	assert(has_events() && "polled for events but no events");
	key_event event = events.queue[events.begin];
	events.begin = (events.begin + 1u) % n_queue;
	return event;
}

void print_gpu_buffer(opengl_context* compute, experiment* experiments, u32 u)
{
	compute[u].storages[0].data = new u32[compute[u].count];
	compute[u].storages[1].data = (u32*)  new vec4_f32[compute[u].count];
	compute[u].storages[2].data = (u32*)  new vec4_f32[experiments[u].count_caps * compute[u].count];
	compute[u].storages[3].data = (u32*)  new vec4_f32[experiments[u].count_caps];

	auto buffer = compute[u].storages[2].buffer;
	auto data = compute[u].storages[2].data;
	auto count = compute[u].count;
	auto algorithm= experiments[u].algorithm;
	auto iterations = experiments[u].iterations;
	auto caps_count = experiments[u].count_caps;
	gpu_to_cpu(buffer, data, count*caps_count*sizeof(vec4_f32));
	auto buffer_vol = compute[u].storages[3].buffer;
	auto data_vol = compute[u].storages[3].data;
	gpu_to_cpu(buffer_vol, data_vol, caps_count*sizeof(vec4_f32));
	auto vols = (vec4_f32*)data_vol;

	auto sums = new vec4_f32[caps_count];
	memset(sums, 0, sizeof(vec4_f32)*caps_count);


	printf("Algo %u Count %u, Caps Count %u  Iterations %u\n", algorithm, count, caps_count, iterations);
	for (u32 j{}; j < caps_count; j++)
	{
		for (u32 i{}; i < count; i++)
			sums[j] = sums[j] + ((vec4_f32*)data)[i + count * j];
		printf("Sum: %.2f, Vol: %.2f, Difference: %.2f\t", sums[j].x, vols[j].x, sums[j].x - vols[j].x);
		printf("Normalized Sum: %.2f, Vol: %.2f, Difference: %.2f\n", sums[j].x/count, vols[j].x, sums[j].x/count - vols[j].x);
	}
}
