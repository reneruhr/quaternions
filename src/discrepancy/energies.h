#pragma once
#include "common.h"
#include "vec_math.h"

template <class vec>
inline f64 sum_of_distances(vec *v, u32 size, u32 offset = 0)
{
	// ||x-y||^2 = 2-2(x,y)
	// sum_{ij}  = 2*sum_{j<i}
	
	
	f64 c{};
	u64 size2 = (u64)(size-offset)*u64(size-offset);

	// Divivde calculation into buckets.
	const u32 n_buckets{ 10 };
	f64 sums[n_buckets];
	u64 bucket_size  = (size2 + n_buckets - 1) / n_buckets;
	memset(sums, 0, sizeof(sums));
	u32 u{};  //(global)index of bucket 
	u64 k{};  //(local) index in bucket

	for (u32 i = offset; i < size; i++)
	{
		//Optional renormalization:
		//v[i] = fast_inv_sqrt(dot(v[i], v[i])) * v[i];

		for (u32 j = offset; j < i; j++)
		{ 
			f64& sum = sums[u];
			if (k++ == bucket_size) 
			{
				u++; k = 0;
			}
			f64 y = std::sqrt(std::abs(1.f - dot(v[i], v[j])));
			{  // Kahan Summation
				y -= c;
				volatile f64 t = sum + y;
				volatile f64 z = t - sum;
				c = z - y;
				sum = t;
			}
		}
	}

	f64 sum{};
	for (u32 j{}; j < n_buckets; j++) sum += sums[j];
	sum *= std::sqrt(2.);
	sum *= 2.;
	sum /= size2;
	return sum;
}


inline f32 calc_energy(f64 distance_sum, vec3_f32)
{
	return (f32)(std::sqrt(std::abs(4./3. - distance_sum))/2);
}

inline f32 calc_energy(f64 distance_sum, vec4_f32)
{
	return (f32)std::sqrt(std::abs(64./15./3.14159265 - distance_sum))/2;
}


struct energy_result
{
	u32 n{};
	f32 *energies{};
	u32 *sizes{};
};

template <class vec, class vec2 = vec>
energy_result energy_test(vec *v, u32 n_sizes, u32 *sizes)
{
	f32 *energies = new f32[n_sizes];

	for(u32 u=0; u<n_sizes; u++)
	{
		energies[u] = calc_energy(sum_of_distances<vec>(v, sizes[u], 0), vec2{});
	}
	
	return {n_sizes, energies, sizes};
}

void print(energy_result results)
{
	printf("\tn sizes: %u\n\tSizes: ", results.n);
	for(u32 u = 0; u< results.n; u++)
		printf("%u ", results.sizes[u]);
	printf("\n\tEnergies: ");
	for(u32 u = 0; u< results.n; u++)
		printf("%f ", results.energies[u]);
	printf("\n");
}