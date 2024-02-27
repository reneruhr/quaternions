#include "common.h"
#include "vec_math.h"

#include "random_samplers_4d.h"
#include "simd_random_samplers_4d.h"
#include "discrepancy/discrepancy.h"
#include "discrepancy/energies.h"

#include <cstdarg>

#include <string>
#include <span>
#include <algorithm>
#include <execution>
#include <unordered_set>
#include <set>
#include <filesystem>


#define v4_tests          1
#define v4_tests_int      0
#define v4_tests_simd     1
#define v4_tests_qmc      0
#define v4_tests_gpu      0

#define tests_energy      1
#define tests_discrepancy 1
#define tests_sampler     1
#define tests_writes      1
#define write_points      0
#define count_points      0

const char* data_folder  = "data";

const u32 n_sizes  = 18;

struct {
    u32 n_samplers = 16'384;
    u32 n_writes = 16'384;

    u32 n_samplers_rep = 100;
    u32 n_writes_rep   = 100;
	u32 sizes[n_sizes] {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16'384, 32'768, 65'536, 131'072, 262'144, 524'288, 1'048'576, 2'097'152, 4'194'304};
    u32 n_discs_t  = 14;
    u32 n_energy_t = 9;

    u32 n_discs_mc = 2048;

    u32 n_save     = 100'000;
} test;

using sampler_func_4d    = quat   (*)();
using sampler_func_4d_x8 = quatx8 (*)();
using fill_func_4d    = void     (*)(quat*, u32);
using no_sampler      = void (*)();

struct sample_algorithm
{
    const char* label;
    union {
        sampler_func_4d sampler4;
        sampler_func_4d_x8 sampler8;
        no_sampler      noop;
    };
    fill_func_4d fill4;
    enum class tag { quat    } tag{ tag::quat};
    enum class typ { mc, qmc } typ{ typ::mc };
    enum class sse { x1, x8  } sse{ sse::x1 };

    u32* sizes{};
    u32 n_sizes{};
    char* log{};
    u32 log_idx{};
};

#define test_case_quat_no_rng(name){.label = "v4 " #name ,          .sampler4  = sampler_##name,      .fill4        = fill_##name,     .tag = sample_algorithm::tag::quat }
#define test_case_quat(name, rng)  {.label = "v4 " #name " "#rng,   .sampler4  = sampler_##name<rng>, .fill4        = fill_##name<rng>,.tag = sample_algorithm::tag::quat }
#define test_case_quat_simd(name, rng)  {.label = "v4 " #name " "#rng,   .sampler8  = sampler_##name##_soa<rng>, .fill4        = fill_##name<rng>,.tag = sample_algorithm::tag::quat, .sse = sample_algorithm::sse::x8}

#define test_case_quat_qmc(name)   {.label = "v4 " #name ,          .noop      = sampler_##name,      .fill4        = fill_##name,     .tag = sample_algorithm::tag::quat, .typ= sample_algorithm::typ::qmc  }
#define test_case_quat_qmc_size(name)   {.label = "v4 special sizes " #name ,          .noop      = sampler_##name,      .fill4        = fill_##name,     .tag = sample_algorithm::tag::quat, .typ= sample_algorithm::typ::qmc, .sizes = hecke5_sizes, .n_sizes = 7  }
#define test_case_quat_gpu(name, rng)   {.label = "v4 " #name " "#rng, .sampler4 = sampler_##name<rng>,      .fill4        = fill_##name<rng>,     .tag = sample_algorithm::tag::quat, .typ= sample_algorithm::typ::mc  }

sample_algorithm sample_algorithms[] =
{
#if(v4_tests)

    test_case_quat(rejection,           xorshift_sampler),
    test_case_quat(marsaglia,           xorshift_sampler),
    test_case_quat(polar_marsaglia,     xorshift_sampler),

    test_case_quat(quaternion_5,         no_backtracking_walk<6u>),
    test_case_quat(quaternion_5_std,     no_backtracking_walk<6u>),
    test_case_quat(quaternion_17,        no_backtracking_walk<18u>),
    test_case_quat(quaternion_29,        no_backtracking_walk<30u>),

    test_case_quat(quaternion_5,         bernoulli_walk<6u>),
    test_case_quat(quaternion_5_std,     bernoulli_walk<6u>),
    test_case_quat(quaternion_17,        bernoulli_walk<18u>),
    test_case_quat(quaternion_29,        bernoulli_walk<30u>),

    test_case_quat(quaternion_5,         bernoulli_walk_no_modulus<7u>),

#endif

#if(v4_tests_simd)
    test_case_quat_simd(polar_marsaglia_x8, simd_xorshift_sampler),
    test_case_quat_simd(polar_marsaglia_x8, simd_lcg_xsh_sampler),
    test_case_quat_simd(quaternion_5_x8, walk_biased<simd_xorshift_sampler>),
    test_case_quat_simd(quaternion_5_x8, walk_biased<simd_lcg_xsh_sampler>),
    test_case_quat_simd(quaternion_5_x8, no_backtracking_walk_biased<simd_xorshift_sampler>),
    test_case_quat_simd(quaternion_5_x8, no_backtracking_walk_biased<simd_lcg_xsh_sampler>),

    test_case_quat(quaternion_5_x4,     bernoulli_walk<6u>),
    test_case_quat(quaternion_5_x4,     no_backtracking_walk<6u>),
    test_case_quat(quaternion_5_x4,     bernoulli_walk_no_modulus<7u>),
    test_case_quat(quaternion_17_x4,     bernoulli_walk<18u>),
    test_case_quat(quaternion_17_x4,     no_backtracking_walk<18u>),
    test_case_quat(quaternion_29_x4,     bernoulli_walk<30u>),
    test_case_quat(quaternion_29_x4,     no_backtracking_walk<30u>),
#endif

#if(v4_tests_int)
    test_case_quat(quaternion_int_S5_5,  no_backtracking_walk<6u>),
    test_case_quat(quaternion_int_S29_3, no_backtracking_walk<30u>),
    test_case_quat(quaternion_int_S29_5, no_backtracking_walk<30u>),
    test_case_quat(quaternion_int_5,     no_backtracking_walk<6u>),
    test_case_quat(quaternion_int_5_norm,no_backtracking_walk<6u>),
    test_case_quat(quaternion_int_29,    no_backtracking_walk<30u>),
    test_case_quat(quaternion_int_5_29,  no_backtracking_walk<6u>),

    test_case_quat(quaternion_int_S5_5,  bernoulli_walk<6u>),
    test_case_quat(quaternion_int_S29_3, bernoulli_walk<30u>),
    test_case_quat(quaternion_int_S29_5, bernoulli_walk<30u>),
    test_case_quat(quaternion_int_5,     bernoulli_walk<6u>),
    test_case_quat(quaternion_int_5_norm,bernoulli_walk<6u>),
    test_case_quat(quaternion_int_29,    bernoulli_walk<30u>),
    test_case_quat(quaternion_int_5_29,  bernoulli_walk<6u>),

    test_case_quat(quaternion_int_5,     bernoulli_walk_no_modulus<7u>),
    test_case_quat(quaternion_int_5_norm,bernoulli_walk_no_modulus<7u>),
#endif

#if(v4_tests_qmc)
    test_case_quat_qmc(hecke_spheres_5),
    test_case_quat_qmc(spherical_fibonacci),
    test_case_quat_qmc(partial_tree),
    test_case_quat_qmc(partial_tree2),
    test_case_quat_qmc(partial_tree3),
    test_case_quat_qmc(partial_tree4),
#endif

#if(v4_tests_gpu)
    test_case_quat_gpu(polar_1_indexed, pcg_indexed_sampler),
    test_case_quat_gpu(polar_2_indexed, pcg3_indexed_sampler),
    test_case_quat_gpu(walk_6, pcg_indexed_sampler),
#endif

};

u32 n_algorithms = sizeof(sample_algorithms) / sizeof(sample_algorithms[0]);

struct timing
{
    const char* label;
    u32 n{};
    u32 n_reps{};
    u64* times{};
    f32* discrepancies{};
};

struct energy 
{
    const char* label;
	u32 n{};
	u32 *sizes{};
	f32 *energies{};
};

struct points
{
    const char* label;
    u32 n{};
    f32* data{};
};

struct discrepancy 
{
    const char* label;
	u32 n{};
	u32 *sizes{};
	f32 *d2{};
	f32 *dinfty{};
};

struct counts
{
    const char* label;
	u32 n{};
    u32* uniques{};
};

timing     profile_sampler(sample_algorithm& algo);
timing     profile_write(sample_algorithm& algo);

energy      calculate_energies(sample_algorithm& algo);
discrepancy calculate_discrepancies(sample_algorithm& algo);

vec4_f32*   fill_vec4_buffer(sample_algorithm& algo, u32 n);
points      produce_points(sample_algorithm& algo);
counts      count_uniques(sample_algorithm&);

void write(timing, const char* suffix = "");
void write(energy);
void write(discrepancy);
void write(points);
void write(counts);

void init_log(sample_algorithm& algo);
void log(sample_algorithm& algo, const char* fmt, ...);
void log(sample_algorithm& algo, const discrepancy_result& result, bool measured_intersections = true);
void log(sample_algorithm& algo, timing);
void log(sample_algorithm& algo, energy_result results);
void log(sample_algorithm& algo, discrepancy_d2_dinfty results);

int main(int argc, char** args)
{
    std::span algos(sample_algorithms, n_algorithms);

    std::for_each(std::execution::par, std::begin(algos), std::end(algos), [&](sample_algorithm& algo)
	{
		init_log(algo);
        
        energy energies{};
        discrepancy discreps{};
        points points{};
        counts counts{};
        
		if(tests_energy)
			energies = calculate_energies(algo);
		if(tests_discrepancy)
			discreps = calculate_discrepancies(algo);
		if(write_points)
			points = produce_points(algo);
		if(count_points)
			counts = count_uniques(algo);

		write(energies);
		write(discreps);
		write(points);
		write(counts);

        delete[] points.data; 
        delete[] energies.energies; 
        delete[] discreps.d2;
        delete[] discreps.dinfty;
        delete[] counts.uniques;
		printf("Finished discrepancy tests and writes of %s\n", algo.label);
	}
    );
    std::for_each(std::execution::seq, std::begin(algos), std::end(algos), [&](sample_algorithm& algo)
	{
        timing timings{}, timings2{};
        
		if(tests_sampler && algo.typ == sample_algorithm::typ::mc)
			timings  = profile_sampler(algo);
		if(tests_writes)
			timings2 = profile_write(algo);
		write(timings, "sampler_");
		write(timings2, "write_");

        delete[] timings.times; 
        delete[] timings.discrepancies;
        delete[] timings2.times;
        delete[] timings2.discrepancies; 
		printf("Finished Speed tests of %s\n", algo.label);
	}
    );

    for (const auto& algo : algos)
    {
        printf("%s", algo.log);
        delete[] algo.log;
    }
}

void remove_angles(std::string &label) {
    label.erase(std::remove(label.begin(), label.end(), '<'), label.end());
    label.erase(std::remove(label.begin(), label.end(), '>'), label.end());
}

void write(timing timings, const char* suffix)
{ 
    if (!timings.n) return;
    std::string file_name = std::string("timings_") + suffix + timings.label + ".bin";
    remove_angles(file_name);
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

void write(energy energy)
{
    if (!energy.n) return;
    std::string file_name = std::string("energies_") + energy.label + ".bin";
    remove_angles(file_name);
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 m  = (u32)strlen(energy.label);
	fwrite(&m,              sizeof(u32), 1, file);
	fwrite(energy.label,    sizeof(char), strlen(energy.label), file);
	fwrite(&energy.n,       sizeof(u32), 1, file);
	fwrite(energy.sizes,      sizeof(u32), energy.n, file);
	fwrite(energy.energies, sizeof(f32), energy.n, file);
    fclose(file);
}

void write(discrepancy disc)
{
    if (!disc.n) return;
    std::string file_name = std::string("discrepancies_") + disc.label + ".bin";
    remove_angles(file_name);
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 m  = (u32)strlen(disc.label);
	fwrite(&m,              sizeof(u32), 1, file);
	fwrite(disc.label,    sizeof(char), strlen(disc.label), file);
	fwrite(&disc.n,       sizeof(u32), 1, file);
	fwrite(disc.sizes,    sizeof(u32), disc.n, file);
	fwrite(disc.d2,       sizeof(f32), disc.n, file);
	fwrite(disc.dinfty,   sizeof(f32), disc.n, file);
    fclose(file);
}

void write(points p)
{
    if (!p.n) return;
    std::string file_name = std::string("points_") + p.label + ".bin";
    remove_angles(file_name);
    auto path = std::filesystem::path(data_folder) / file_name.c_str();
	FILE* file = fopen(path.string().c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 n_text = (u32)strlen(p.label);
	fwrite(&n_text,    sizeof(u32), 1,       file);
	fwrite(p.label,    sizeof(char),n_text,  file);
	fwrite(&p.n,       sizeof(u32), 1,       file);
	fwrite(p.data,   4*sizeof(f32), p.n,     file);
    fclose(file);
}

void write(counts count)
{
    if (!count.n) return;
    std::string file_name = std::string("counts_") + count.label + ".bin";
    remove_angles(file_name);
	FILE* file = fopen(file_name.c_str(), "wb");
    assert(file && "Failed opening file.");
    u32 m  = (u32)strlen(count.label);
	fwrite(&m,             sizeof(u32), 1, file);
	fwrite(count.label,    sizeof(char), strlen(count.label), file);
	fwrite(&count.n,       sizeof(u32), 1, file);
	fwrite(test.sizes,     sizeof(u32), count.n, file);
	fwrite(count.uniques,  sizeof(u32), count.n, file);
    fclose(file);
}

void log(sample_algorithm& algo, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    algo.log_idx += vsnprintf(algo.log + algo.log_idx, 1024*10, fmt, args);
    va_end(args);
}

void init_log(sample_algorithm& algo)
{
    algo.log = new char[4096]; 
    log(algo, "\nTesting %s\n", algo.label);
}

void log(sample_algorithm& algo, const discrepancy_result& result, bool measured_intersections)
{
	f64 time_s = (f64)result.time/cpu_frequency();
	log(algo, "\tTime taken: %f ms (%llu cycles) \n", time_s*1e3, result.time);
	if(result.area || result.discrepancy){
		log(algo, "\tThroughput: %.0f Million %s/s \n", result.n_data/time_s*1.e-6, measured_intersections ? "intersections" : "buffer writes");
		log(algo, "\tRelative Intersections: %.6f", result.area);
		log(algo, "\tDiscrepancy: %.6f", result.discrepancy);
		log(algo, "\tRelative Error: %.6f\n", result.discrepancy/result.area);
	}
}

void log(sample_algorithm& algo, energy_result results)
{
	log(algo, "\tn sizes: %u\n\tSizes: ", results.n);
	for(u32 u = 0; u< results.n; u++)
		log(algo, "%u ", results.sizes[u]);
	log(algo, "\n\tEnergies: ");
	for(u32 u = 0; u< results.n; u++)
		log(algo, "%f ", results.energies[u]);
	log(algo, "\n");
}

void log(sample_algorithm& algo, discrepancy_d2_dinfty results)
{
	log(algo, "\tn sizes: %u\n\tSizes: ", results.n);
	for(u32 u = 0; u< results.n; u++)
		log(algo, "%u ", results.sizes[u]);
	log(algo, "\n\tD2 Discrepancy: ");
	for(u32 u = 0; u< results.n; u++)
		log(algo, "%f ", results.d2[u]);
	log(algo, "\n\tDinfty Discrepancy: ");
	for(u32 u = 0; u< results.n; u++)
		log(algo, "%f ", results.dinfty[u]);
	log(algo, "\n");
}

void log(sample_algorithm& algo, timing timings)
{
	log(algo, "\n\tSamples: %u. Repetitions: %u\n", timings.n, timings.n_reps);
	log(algo, "\tTimes: ");
	for(u32 u = 0; u< timings.n_reps; u++)
		log(algo, "%u ", timings.times[u]);
	log(algo, "\n\tDiscrepancies: ");
	for(u32 u = 0; u< timings.n_reps; u++)
		log(algo, "%f ", timings.discrepancies[u]);
	log(algo, "\n");
}

timing profile_sampler(sample_algorithm& algo)
{
    u32 n = test.n_samplers; 
    spherical_cap4 *caps4{};
    caps4 = new spherical_cap4[test.n_samplers_rep];
	produce_caps(caps4, test.n_samplers_rep,1);

    discrepancy_result result;
    auto discs = new f32[test.n_samplers_rep];
    auto times = new u64[test.n_samplers_rep];

    for (u32 u{}; u < test.n_samplers_rep; u++)
    {
        if(algo.sse == sample_algorithm::sse::x1)
			result = profile_discrepancy<spherical_cap4>(caps4[u], algo.sampler4, n);
        else 
			result = profile_discrepancy_x8<spherical_cap4>(caps4[u], algo.sampler8, n);

        times[u] = result.time;
        discs[u] = (f32)result.discrepancy;

    }
    timing timing{ .label = algo.label, .n = test.n_samplers, .n_reps = test.n_samplers_rep, .times = times, .discrepancies = discs };
	log(algo, timing);
    delete[] caps4;
    return timing;
}

timing profile_write(sample_algorithm& algo)
{
    spherical_cap4* caps4{};
    caps4 = new spherical_cap4[test.n_samplers_rep];
    produce_caps(caps4, test.n_samplers_rep, 1);

    discrepancy_result result;

    u32 n = test.n_writes;
    auto discs = new f32[test.n_writes_rep];
    auto times= new u64[test.n_writes_rep];

    for (u32 u{}; u < test.n_writes_rep; u++)
    {
        result = profile_fill_buffer<spherical_cap4, quat>(caps4[u], algo.fill4, n);

        times[u] = result.time;
        discs[u] = (f32)result.discrepancy;

    }
    timing timing{ .label = algo.label, .n = test.n_writes, .n_reps = test.n_writes_rep, .times = times, .discrepancies = discs };
	log(algo, timing);
    delete[] caps4;
    return  timing;
}

energy calculate_energies(sample_algorithm& algo)
{
    u32 use_max_test_size = test.n_energy_t;
    u32 n = test.sizes[test.n_energy_t- 1];
    energy_result results;
    if (algo.typ == sample_algorithm::typ::mc)
    {
        auto v = new quat[n];
		algo.fill4(v, n);
        results = energy_test((vec4_f32*)v, test.n_energy_t, test.sizes);
		delete[] v; 
    } 
    else if (algo.tag == sample_algorithm::tag::quat && algo.typ == sample_algorithm::typ::qmc)
    {
        u32 n_tests = algo.n_sizes ? algo.n_sizes : test.n_energy_t;
        u32* sizes  = algo.n_sizes ? algo.sizes   : test.sizes;
        while (sizes[n_tests - 1] > test.sizes[test.n_energy_t]) n_tests--;

        auto energies = new f32[n_tests];
        auto v = new quat[sizes[n_tests-1]];
        for (u32 u{}; u < n_tests; u++)
        {
			algo.fill4(v, sizes[u]);
			auto temp = energy_test((vec4_f32*)v, 1, &sizes[u]);
            energies[u] = temp.energies[0];
            delete[] temp.energies;
		}
        results = { n_tests, energies, sizes };
		delete[] v; 
    }
    log(algo, results);
    return { algo.label, results.n, results.sizes, results.energies };
}

points produce_points(sample_algorithm& algo)
{
    u32 n = test.n_save;
    return { algo.label, n, (f32*)fill_vec4_buffer(algo, n)};
}

discrepancy calculate_discrepancies(sample_algorithm& algo)
{
    u32 n = test.sizes[test.n_discs_t-1];
    discrepancy_d2_dinfty results;
    if (algo.tag == sample_algorithm::tag::quat && algo.typ == sample_algorithm::typ::mc)
    {
        auto v = new quat[n];
		algo.fill4(v, n);
        results = approximate_discrepancy<spherical_cap4>(v, test.n_discs_t, test.sizes, test.n_discs_mc*8);
		delete[] v; 
    }
    else if (algo.tag == sample_algorithm::tag::quat && algo.typ == sample_algorithm::typ::qmc)
    {
        u32 n_tests = algo.n_sizes ? algo.n_sizes : test.n_discs_t;
        u32* sizes  = algo.n_sizes ? algo.sizes   : test.sizes;
        auto d2     = new f32[n_tests];
        auto dinfty = new f32[n_tests];
        auto v = new quat[sizes[n_tests-1]];
        for (u32 u{}; u < n_tests; u++)
        {
			algo.fill4(v, sizes[u]);
			auto temp = approximate_discrepancy<spherical_cap4>(v, 1, &sizes[u], test.n_discs_mc*8);
            d2[u] = temp.d2[0];
            dinfty[u] = temp.dinfty[0];
            delete[] temp.d2; 
            delete[] temp.dinfty;
		}
		delete[] v; 
        results = { n_tests, d2, dinfty, sizes};
    }
    log(algo, results);
    return {algo.label, results.n, results.sizes, results.d2, results.dinfty};
}

counts count_uniques(sample_algorithm& algo)
{
    u32 n = test.sizes[n_sizes - 1];
    auto v = fill_vec4_buffer(algo, n);
	std::span points(v, n);

    auto n_uniques = new u32[n_sizes];
    auto vec4_hash = [](const vec4_f32& v) {
        return std::hash<f32>()(v.x) ^ std::hash<f32>()(v.y) ^ std::hash<f32>()(v.z) ^ std::hash<f32>()(v.w);
        };
	std::unordered_set<vec4_f32, decltype(vec4_hash)> uniques({}, vec4_hash);

    u32 prev{};
    log(algo, "\tFraction of unique points: ");
    for (u32 u{}; u < n_sizes; u++)
    { 
		uniques.insert(std::begin(points)+prev, std::begin(points)+test.sizes[u]);
		log(algo, "%.2f, ", (f32)uniques.size()/test.sizes[u]);
        n_uniques[u] = (u32)uniques.size();
        prev = test.sizes[u];
	}
    log(algo, "\n");
    delete[] v;

    return {
        .label = "Counting Uniques",
        .n = n_sizes,
        .uniques = n_uniques
    };
}

vec4_f32* fill_vec4_buffer(sample_algorithm& algo, u32 n)
{
    auto data = new vec4_f32[n];
    auto v = new quat[n];
    algo.fill4(v, n);
    for (u32 u = 0; u < n; u++)
    {
        data[u].x = v[u].r;
        data[u].y = v[u].x;
        data[u].z = v[u].y;
        data[u].w = v[u].z; 
    }
    delete[] v; 
    return data;
}
