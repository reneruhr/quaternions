#pragma once
#include "common.h"
#include "vec_math.h"
#include "quaternions.h"
#include "random_int.h"
#include "random_sphere.h"


// spherical cap 
struct spherical_cap
{
	f32 h{};
	vec3_f32 n{};
};

struct spherical_cap4
{
	f32 h{};
	vec4_f32 n{};
};

void produce_caps(spherical_cap* caps, u32 m)
{
	xorshift2_sampler sampler;
	for (u32 u = 0; u < m; u++)
	{
	  caps[u].h = std::max(0x1p-31f*sampler(1), 0.0f);	 // uniform in [eps,2]
	  caps[u].n = sphere_rejection(sampler);			 // uniform on S2
	}
}

void produce_caps(spherical_cap4* caps, u32 m_caps, u32 m_heights)
{
	xorshift_sampler sampler;
	auto v = new quat[m_caps*m_heights];
	fill_spherical_fibonacci(v, m_caps);

	for (u32 u = 0; u < m_caps*m_heights; u++)
	{
	  caps[u].h = 0x1p-32f*sampler();
	  caps[u].n = to_vec4(v[u/m_heights]);
	}
	delete[] v;
}

inline const char* label(spherical_cap)
{
	return "spherical_cap test";
}

inline const char* label(spherical_cap4)
{
	return "spherical_cap 4d test";
}

inline bool is_inside(spherical_cap spherical_cap, vec3_f32 v)
{
	f32 h = 1.f-(spherical_cap.n,v);
	return h < spherical_cap.h;
}

inline bool is_inside(spherical_cap spherical_cap, vec4_f32 v)
{
	f32 h = 1.f - (spherical_cap.n.x * v.x + spherical_cap.n.y * v.y + spherical_cap.n.z * v.z);
	return h < spherical_cap.h;
}

inline bool is_inside(spherical_cap4 spherical_cap, vec4_f32 v)
{
	f32 h = 1.f - dot(spherical_cap.n,v);
	return h < spherical_cap.h;
}

inline bool is_inside(spherical_cap4 spherical_cap, quat q)
{
	f32 h = 1.f - (spherical_cap.n.x * q.r + spherical_cap.n.y * q.x + spherical_cap.n.z * q.y + spherical_cap.n.w * q.z);
	return h < spherical_cap.h;
}

template <class set, class vec>
inline u32 count(set cap, vec* v, u32 n)
{
	u32 m{};
	for (u32 u= 0; u < n; u++)
		if(is_inside(cap, v[u])) m++;
	return m;
}

inline f64 calc_area(spherical_cap spherical_cap)
{
	return 2*std::numbers::pi*spherical_cap.h;
}

inline f64 ambient_vol(spherical_cap)
{
	return 4*std::numbers::pi;
}

// S. Li: Concise Formulas for the Area and Volume of a Hyperspherical Cap
inline f64 calc_area(spherical_cap4 spherical_cap)
{
	f64 theta = std::acos(1. - spherical_cap.h);
	f64 I = 2 * theta - std::sin(2 * theta);
	return std::numbers::pi * I;
}

inline f64 ambient_vol(spherical_cap4)
{
	return 2*std::numbers::pi*std::numbers::pi;
}
 
inline void print(spherical_cap spherical_cap)
{
	f64 A = calc_area(spherical_cap);
	printf("spherical_cap n=(%.1f,%.1f,%.1f) h=%.1f of area=%.3f\n",
			spherical_cap.n.x, spherical_cap.n.y, spherical_cap.n.z, 
			spherical_cap.h, A);
}

inline f64 calc_discrepancy(u32 n_inside, u32 n_samples, f64 area){
	return std::abs((f64)n_inside/(f64)n_samples - area);
}

struct discrepancy_result
{
	f64 area{};
	f64 discrepancy{};
	u64 time{};
	u64 n_data{};

	// Discrepancy test results
	template <class TestShape>
	discrepancy_result(u32 sum, u32 n_samples, const TestShape& set, u64 start, u64 end) :
		area((f64)sum/(f64)n_samples),	
		discrepancy(calc_discrepancy(sum, n_samples, calc_area(set)/ambient_vol(set))),
		time(end-start),
		n_data(n_samples)
	{}
	discrepancy_result() = default;
};

inline void print(const discrepancy_result& result, bool measured_intersections = true)
{
	f64 time_s = (f64)result.time/cpu_frequency();
	printf("\tTime taken: %f ms (%llu cycles) \n", time_s*1e3, result.time);
	if(result.area || result.discrepancy){
		printf("\tThroughput: %.0f Million %s/s \n", result.n_data/time_s*1.e-6, measured_intersections ? "intersections" : "buffer writes");
		printf("\tRelative Intersections: %.6f", result.area);
		printf("\tDiscrepancy: %.6f", result.discrepancy);
		printf("\tRelative Error: %.6f\n", result.discrepancy/result.area);
	}
}

template <class Set>
inline discrepancy_result profile_discrepancy(const Set& set, vec3_f32(*func)(), u32 n_samples)
{
	u32 sum{ 0 };

	u64 start_time = cpu_timer();
	for (u32 i{ 0 }; i < n_samples; i++)
	{
		if (is_inside(set, func()))
			sum++;
	}
	u64 end_time = cpu_timer();

	return discrepancy_result(sum, n_samples, set, start_time, end_time);
}

template <class Set>
inline discrepancy_result profile_discrepancy(const Set& set, quat(*func)(), u32 n_samples)
{
	u32 sum{ 0 };

	u64 start_time = cpu_timer();
	for (u32 i{ 0 }; i < n_samples; i++)
	{
		if (is_inside(set, func()))
			sum++;
	}
	u64 end_time = cpu_timer();

	return discrepancy_result(sum, n_samples, set, start_time, end_time);
}

template <class Set>
inline discrepancy_result profile_discrepancy_x8(const Set& set, quatx8(*func)(), u32 n_samples)
{
	u32 count = n_samples / 8;
	
	u64 start_time = cpu_timer();

	__m256i sum = _mm256_setzero_si256();

	auto set_simd = make_simd_x8(set);

	const __m256i one = _mm256_set1_epi32(1);

	while(count--)
	{
		alignas(32) auto v = func();
		__mmask8 dirac = is_inside_simd(set_simd, reinterpret_cast<__m256*>(&v));
		sum = _mm256_mask_add_epi32(sum, dirac, sum, one);
	}

	sum = _mm256_hadd_epi32(sum, sum);
	sum = _mm256_hadd_epi32(sum, sum);

	__m128i sum_a = _mm256_extracti128_si256(sum, 0);
	__m128i sum_b = _mm256_extracti128_si256(sum, 1);

	u32 sum_s = _mm_cvtsi128_si32(_mm_add_epi32(sum_a, sum_b));

	u64 end_time = cpu_timer();

	return discrepancy_result(sum_s, n_samples, set, start_time, end_time);
}


template <class Set, class vec>
inline discrepancy_result profile_fill_buffer(const Set& set, void(*func)(vec*, u32), u32 n_samples)
{
	u32 sum{ 0 };
	
	auto* v = new vec[n_samples];

	u64 start_time = cpu_timer();
	func(v, n_samples);
	u64 end_time = cpu_timer();

	for (u32 i{ 0 }; i < n_samples; i++)
	{
		if (is_inside(set, v[i]))
			sum++;
	}
	delete[] v;
	return discrepancy_result(sum, n_samples, set, start_time, end_time);
}


struct discrepancy_d2_dinfty
{
	u32 n{};
	f32 *d2{};
	f32 *dinfty{};
	u32* sizes{};
};

template <class set, class vec>
discrepancy_d2_dinfty approximate_discrepancy(vec *v, u32 n_test_sizes, u32* sizes, u32 n_qmc)
{
	discrepancy_d2_dinfty discrepancies 
	{
		.n = n_test_sizes,
		.d2 = new f32[n_test_sizes],
		.dinfty = new f32[n_test_sizes],
		.sizes = sizes
	};

    u32 m = n_qmc;
    auto caps = new set[m];
    produce_caps(caps, m/8, 8);
    auto discs = new f64[m];
    for(u32 i=0; i<n_test_sizes; i++)
    {    
        f64 dinfty = 0;
        f64 d2 = 0;
        for (u32 u=0; u<m; u++)
        {
            discs[u] = (f64)count(caps[u], v, sizes[i])/(f64)sizes[i] - calc_area(caps[u]) / ambient_vol(caps[u]);
            dinfty   = std::max(std::abs(discs[u]), dinfty);
            discs[u] *= discs[u];
            d2 		 += discs[u];
        }
        d2 /= m;
        discrepancies.d2[i] 	= (f32)std::sqrt(d2 * 2.); // 2 coming from measure over [-1,1] in D2
        discrepancies.dinfty[i] = (f32)dinfty;
	}

	delete[] caps;
	delete[] discs;
	return discrepancies;
}

void print(discrepancy_d2_dinfty results)
{
	printf("\tn sizes: %u\n\tSizes: ", results.n);
	for(u32 u = 0; u< results.n; u++)
		printf("%u ", results.sizes[u]);
	printf("\n\tD2 Discrepancy: ");
	for(u32 u = 0; u< results.n; u++)
		printf("%f ", results.d2[u]);
	printf("\n\tDinfty Discrepancy: ");
	for(u32 u = 0; u< results.n; u++)
		printf("%f ", results.dinfty[u]);
	printf("\n");
}

#ifndef __APPLE__
#include <immintrin.h>

struct spherical_cap4_x8
{
	__m256 h;
	__m256 n[4];

	spherical_cap4_x8(spherical_cap4 spherical_cap)
	{
		h    = _mm256_set1_ps(spherical_cap.h  );
		n[0] = _mm256_set1_ps(spherical_cap.n.x);
		n[1] = _mm256_set1_ps(spherical_cap.n.y);
		n[2] = _mm256_set1_ps(spherical_cap.n.z);
		n[3] = _mm256_set1_ps(spherical_cap.n.w);
	}
};

inline spherical_cap4_x8 make_simd_x8(spherical_cap4 spherical_cap) 
{
	return spherical_cap4_x8(spherical_cap);
}

inline __mmask8 is_inside_simd(spherical_cap4_x8 spherical_cap, __m256 v[4])
{
	__m256 nv = _mm256_fmadd_ps(spherical_cap.n[0], v[0], _mm256_fmadd_ps(spherical_cap.n[1], v[1], _mm256_fmadd_ps(spherical_cap.n[2], v[2],  _mm256_mul_ps(spherical_cap.n[3], v[3]))));
	__m256 h = _mm256_sub_ps(_mm256_set1_ps(1.f), nv);
	return _mm256_cmp_ps_mask(h, spherical_cap.h, _CMP_LT_OQ);
}
#endif