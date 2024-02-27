#pragma once
#include "random_int.h"

#include "simd_random_int.h"
#include "simd_random_quaternions.h"


template <class RandomWalk> 
inline quat sampler_quaternion_5_x4()
{
	static RandomWalk walk;
	static mm_quat q {.q = {0,0,0,1} };
	q.mm = mm_cross4_ps(mm_T5r.s[walk()].mm, q.mm);	
	q.mm = normalize(q.mm);
	return q.q;
}

template <class RandomWalk> 
inline void fill_quaternion_5_x4(quat* v, u32 n)
{
	RandomWalk walk{};
	mm_quat q {.q = {0,0,0,1} };

	u32 u = 0;
	while(u<n)
	{
		q.mm = mm_cross4_ps(mm_T5r.s[walk()].mm, q.mm);	
		if((u&1023)==0) q.mm = normalize(q.mm);
		_mm_storeu_ps((f32*)(v+u),q.mm);
		u++;
	}
}

template <class RandomWalk> 
inline quat sampler_quaternion_17_x4()
{
	static RandomWalk walk;
	static mm_quat q {.q = {0,0,0,1} };
	q.mm = mm_cross4_ps(mm_T17r.s[walk()].mm, q.mm);	
	q.mm = normalize(q.mm);
	return q.q;
}

template <class RandomWalk> 
inline void fill_quaternion_17_x4(quat* v, u32 n)
{
	RandomWalk walk{};
	mm_quat q {.q = {0,0,0,1} };

	u32 u = 0;
	while(u<n)
	{
		q.mm = mm_cross4_ps(mm_T17r.s[walk()].mm, q.mm);	
		if((u&1023)==0) q.mm = normalize(q.mm);
		_mm_storeu_ps((f32*)(v+u),q.mm);
		u++;
	}
}

template <class RandomWalk> 
inline quat sampler_quaternion_29_x4()
{
	static RandomWalk walk;
	static mm_quat q {.q = {0,0,0,1} };
	q.mm = mm_cross4_ps(mm_T29r.s[walk()].mm, q.mm);	
	q.mm = normalize(q.mm);
	return q.q;
}

template <class RandomWalk> 
inline void fill_quaternion_29_x4(quat* v, u32 n)
{
	RandomWalk walk{};
	mm_quat q {.q = {0,0,0,1} };

	u32 u = 0;
	while(u<n)
	{
		q.mm = mm_cross4_ps(mm_T29r.s[walk()].mm, q.mm);	
		if((u&1023)==0) q.mm = normalize(q.mm);
		_mm_storeu_ps((f32*)(v+u),q.mm);
		u++;
	}
}

template <class Sampler>
inline quat sampler_polar_marsaglia_x8()
{
	static Sampler sampler{};

	quatx8  v = marsaglia_polar_simd(sampler);
	quat v_out[8];
	soa_to_aof(&v, v_out);
	return { v_out[0].r, v_out[0].x, v_out[0].y, v_out[0].z};
}

template <class Sampler>
inline quatx8 sampler_polar_marsaglia_x8_soa()
{
	static Sampler sampler{};
	return marsaglia_polar_simd(sampler);
}

template <class Sampler>
inline void fill_polar_marsaglia_x8(quat* v, u32 n)
{
	Sampler sampler{};

	u32 nn = n / 8 * 8;
	u32 u = 0;
	while (u < nn)
	{
		auto w = marsaglia_polar_simd(sampler);
		soa_to_aof(&w, v + u);
		u += 8;
	}
	if (u < n)
	{
		quat v_out[8];
		u32 j = 0;
		auto w = marsaglia_polar_simd(sampler);
		soa_to_aof(&w, v_out);
		while (u < n) v[u++] = v_out[j++];
	}
}


template <class Walk>
inline quat sampler_quaternion_5_x8()
{
	static quatx8 q {};
	static Walk walk{};

	const __m256i choice = walk();
	q = construct_T5(choice) * q;
	q = normalize(q);
	quat v_out[8];
	soa_to_aof(&q, v_out);
	return { v_out[0].r, v_out[0].x, v_out[0].y, v_out[0].z};
}

template <class Walk>
inline quatx8 sampler_quaternion_5_x8_soa()
{
	static quatx8 q {};
	static Walk walk{};

	const __m256i choice = walk();
	q = construct_T5(choice) * q;
	q = normalize(q);
	return q;
}

template <class Walk>
inline void fill_quaternion_5_x8(quat* v, u32 n)
{
	quatx8 q {};

	Walk walk{};

	u32 nn = n / 8 * 8;
	u32 u = 0;

	while (u < nn)
	{
		const __m256i choice = walk();
		q = construct_T5(choice) * q;
		if ((u & 4095) == 0) q = normalize(q);
		soa_to_aof(&q, v+u);
		u += 8;
	}
	if (u < n)
	{
		u32 j = 0;
		quat v_out[8];
		const __m256i choice = walk();
		q = construct_T5(choice) * q;
		soa_to_aof(&q, v_out);
		while (u < n) v[u++] = v_out[j++]; 
	}
}
