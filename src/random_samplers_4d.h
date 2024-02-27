#pragma once
#include "random_int.h"
#include "random_sphere.h"
#include "quaternions.h"

/*
There are the following kind of samplers: 
"sampler_xxx_xxx",
"fill_xxx_xxx"
The standard sampler has a global variable storing the seed.
The fill version resets the seed for every function call.
*/

template <class Sampler>
inline quat sampler_rejection()
{
	static Sampler sampler{};
	return sphere4_rejection(sampler);
}

template <class Sampler>
inline quat sampler_marsaglia()
{
	static Sampler sampler{};
	return sphere4_marsaglia(sampler);
}

template <class Sampler>
inline quat sampler_polar_marsaglia()
{
	static Sampler sampler{};
	return sphere4_marsaglia_polar(sampler);
}

template <class Sampler>
inline void fill_marsaglia(quat* v, u32 n)
{
	Sampler sampler{};
	u32 u = 0;
	while(n--)
		v[u++]= sphere4_marsaglia(sampler);
}

template <class Sampler>
inline void fill_rejection(quat* v, u32 n)
{
	Sampler sampler{};
	u32 u = 0;
	while(n--)
		v[u++]= sphere4_rejection(sampler);
}

template <class Sampler>
inline void fill_polar_marsaglia(quat* v, u32 n)
{
	Sampler sampler{};
	u32 u = 0;
	while(n--)
		v[u++]= sphere4_marsaglia_polar(sampler);
}

template <class RandomWalk> 
inline quat sampler_quaternion_5()
{
	static RandomWalk walk{};
	static quat q{};
	const quat* table = T5.s;
	q = normalize_fast(table[walk()] * q);
	return q;
}

template <class RandomWalk> 
inline quat sampler_quaternion_5_enc()
{
	static RandomWalk walk{};
	static u32 w = encode(quat{});
	const quat* table = T5.s;
	
	auto q = decode(w);
	q = table[walk()] * q;
	w = encode(q);
	return q;
}

template <class RandomWalk> 
inline void fill_quaternion_5_enc(quat*v, u32 n)
{
	RandomWalk walk{};
	u32 w = encode(quat{});
	const quat* table = T5.s;
	
	u32 u{};
	while (n--)
	{
		auto q = decode(w);
		q = table[walk()] * q;
		w = encode(q);
		v[u++] = q;
	}
}

template <class RandomWalk> 
inline quat sampler_quaternion_5_std()
{
	static RandomWalk walk{};
	static quat q{};
	const quat* table = T5.s;
	q = normalize(table[walk()] * q);
	return q;
}

template <class RandomWalk> 
inline quat sampler_quaternion_17()
{
	static RandomWalk walk{};
	static quat q{};
	const quat* table = T17.s;
	q = normalize_fast(table[walk()] * q);
	return q;
}

template <class RandomWalk> 
inline quat sampler_quaternion_29()
{
	static RandomWalk walk{};
	static quat q{};
	const quat* table = T29.s;
	q = normalize_fast(table[walk()] * q);
	return q;
}

template <class RandomWalk> 
inline quat sampler_quaternion_int_5()
{
	static RandomWalk walk{};
	static quatz q{};
	const quatz* table = T5i.s;
	static u32 tt = 26;
	static u64 norm = 1;
	if (tt == 0)
	{
		q = {};
		tt = 26;
		norm = 1;
	}
	tt--;
	q = table[walk()] * q;
	norm *= 5u;
	return to_quatf(q, (f32)(1./std::sqrt(static_cast<f64>(norm))));
}


template <class RandomWalk> 
inline quat sampler_quaternion_int_5_norm()
{
	static RandomWalk walk{};
	static quatz q{};
	const quatz* table = T5i.s;
	static u32 tt = 26;
	static u64 norm = 1;
	if (tt == 0)
	{
		q = {};
		tt = 26;
		norm = 1;
	}
	tt--;
	q = table[walk()] * q;
		
	f32 norm_f{ 1.f };
	if (tt % 2 == 0) 
	{
		norm *= 5u;
		norm_f = 1./ static_cast<f32>(norm);
	}
	else 
	{
		norm_f = s5 / static_cast<f32>(norm);
	}
	return to_quatf(q, norm_f);
}


template <class JitteredRandomWalk> 
inline quat sampler_quaternion_int_5_jittered()
{
	static JitteredRandomWalk walk{};
	static quatz q{};
	const quatz* table = T5i.s;
	static u32 tt = 26;
	static u64 norm = 1;
	if (tt == 0)
	{
		q = {};
		tt = 26;
		norm = 1;
	}
	tt--;
	q = table[walk()] * q;
	norm *= 5u;
	walk.jitter(q, norm);
	
	return to_quatf(q, fast_inv_sqrt(static_cast<f32>(norm)));
}

const f32 s5_5_norm = static_cast<f32>(1./std::sqrt(5*5*5*5*5.));
const f32 s29_3_norm = static_cast<f32>(1./std::sqrt(29. * 29. * 29.));
const f32 s29_5_norm = static_cast<f32>(1./std::sqrt(29u * 29u * 29u * 29u * 29.));
template <class RandomWalk> 
inline quat sampler_quaternion_int_S5_5()
{
	static RandomWalk walk{};
	quatz q{};
	const quatz* table = T5i.s;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	return make_quat(q, 3125u);
}

template <class RandomWalk> 
inline quat sampler_quaternion_int_S29_3()
{
	static RandomWalk walk{};
	quatz q{};
	const quatz* table = T29i.s;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	return to_quatf(q, s29_3_norm);
}

template <class RandomWalk> 
inline quat sampler_quaternion_int_S29_5()
{
	static RandomWalk walk{};
	quatz q{};
	const quatz* table = T29i.s;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	q = table[walk()] * q;
	return to_quatf(q, s29_5_norm);
}

template <class RandomWalk> 
inline quat sampler_quaternion_int_5_29()
{
	static RandomWalk walk{};
	static quatz q{};
	const quatz* table = T5i.s;
	const quatz* table2 = T29i.s;
	static u32 tt = 24;
	static u32 u = 0;
	static u64 norm = 1;
	if (tt == 0)
	{
		q = {};
		tt = 24;
		norm = 1;
	}
	if (u == 30)
	{
		u = 0;
		q = table[walk()] * q;
		norm *= 5u;
		tt--;
		return to_quatf(q, 1./std::sqrt(static_cast<f64>(norm)));
	}
	else
	{
		return to_quatf(table2[u++]*q, 1./std::sqrt(static_cast<f64>(norm*29)));
	}
}

template <class RandomWalk> 
inline quat sampler_quaternion_int_29()
{
	static RandomWalk walk{};
	static quatz q{};
	const quatz* table = T29i.s;
	static u32 tt = 12;
	static u64 norm = 1;
	if (tt == 0)
	{
		q = {};
		tt = 12;
		norm = 1;
	}
	tt--;
	q = table[walk()] * q;
	norm *= 29u;
	return to_quatf(q, 1./std::sqrt(static_cast<f64>(norm)));
}

template <class RandomWalk>
inline void fill_quaternion_5(quat* v, u32 n)
{
	RandomWalk walk{};
	quat q{};
	const quat* table = T5.s;

	u32 u = 0;
	while (n--)
	{
		if ((n & ((1<<10)-1)) == 0) q = normalize(q);
		q = table[walk()] * q;
		v[u++] = q;
	}
}

template <class RandomWalk>
inline void fill_quaternion_5_std(quat* v, u32 n)
{
	RandomWalk walk{};
	quat q{};
	const quat* table = T5.s;

	u32 u = 0;
	while (n--)
	{
		if ((n & ((1<<10)-1)) == 0) q = normalize(q);
		q = table[walk()] * q;
		v[u++] = q;
	}
}

template <class RandomWalk>
inline void fill_quaternion_17(quat* v, u32 n)
{
	RandomWalk walk{};
	quat q{};
	const quat* table = T17.s;

	u32 u = 0;
	while (n--)
	{
		if ((n & ((1<<10)-1)) == 0) q = normalize(q);
		q = table[walk()] * q;
		v[u++] = q;
	}
}

template <class RandomWalk>
inline void fill_quaternion_29(quat* v, u32 n)
{
	RandomWalk walk{};
	quat q{};
	const quat* table = T29.s;

	u32 u = 0;
	while (n--)
	{
		if ((n & ((1<<10)-1)) == 0) q = normalize(q);
		q = table[walk()] * q;
		v[u++] = q;
	}
}


template <class RandomWalk> 
inline void fill_quaternion_int_5(quat* v, u32 n)
{
	RandomWalk walk{};
	quatz q{};
	const quatz* table = T5i.s;
	u32 tt = 26;
	u64 norm = 1;
	u32 u = 0;
	while (u < n)
	{
		if (tt == 0)
		{
			q = {};
			tt = 26;
			norm = 1;
		}
		tt--;
		q = table[walk()] * q;
		norm *= 5u;
		v[u++] = to_quatf(q, (f32)(1./std::sqrt(static_cast<f64>(norm))));
	}
}

template <class RandomWalk> 
inline void fill_quaternion_int_5_norm(quat* v, u32 n)
{
	RandomWalk walk{};
	quatz q{};
	const quatz* table = T5i.s;
	u32 tt = 26;
	u64 norm = 1;
	u32 u = 0;
	while (u < n)
	{
		if (tt == 0)
		{
			q = {};
			tt = 26;
			norm = 1;
		}
		tt--;
		q = table[walk()] * q;

		f32 norm_f{ 1.f };
		if (tt % 2 == 0) 
		{
			norm *= 5u;
			norm_f = 1./ static_cast<f32>(norm);
		}
		else 
		{
			norm_f = s5 / static_cast<f32>(norm);
		}
		v[u++] = to_quatf(q, norm_f);
	}
}

template <class JitteredRandomWalk> 
inline void fill_quaternion_int_5_jittered(quat* v, u32 n)
{
	JitteredRandomWalk walk{};
	quatz q{};
	const quatz* table = T5i.s;
	u32 tt = 26;
	u64 norm = 1;
	u32 u = 0;
	while (u < n)
	{
		if (tt == 0)
		{
			q = {};
			tt = 26;
			norm = 1;
		}
		tt--;
		q = table[walk()] * q;

		norm *= 5u;
		walk.jitter(q, norm);

		v[u++] = to_quatf(q, fast_inv_sqrt(static_cast<f32>(norm)));
	}
}

template <class RandomWalk> 
inline void fill_quaternion_int_29(quat* v, u32 n)
{
	RandomWalk walk{};
	quatz q{};
	const quatz* table = T29i.s;
	u32 tt = 12;
	u64 norm = 1;
	u32 u = 0;
	while (u < n)
	{
		if (tt == 0)
		{
			q = {};
			tt = 12;
			norm = 1;
		}
		tt--;
		q = table[walk()] * q;
		norm *= 29u;
		v[u++] = to_quatf(q, (f32)(1./std::sqrt(static_cast<f64>(norm))));
	}
}

template <class RandomWalk> 
inline void fill_quaternion_int_S5_5(quat* v, u32 n)
{
	RandomWalk walk{};
	const quatz* table = T5i.s;
	u32 u = 0;
	while (u < n)
	{
		quatz q{};
		u64 norm = 1;
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		v[u++] = to_quatf(q, s5_5_norm);
	}
}

template <class RandomWalk> 
inline void fill_quaternion_int_S29_3(quat* v, u32 n)
{
	RandomWalk walk{};
	const quatz* table = T29i.s;
	u32 u = 0;
	while (u < n)
	{
		quatz q{};
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		v[u++] = to_quatf(q, s29_3_norm);
	}
}



template <class RandomWalk> 
inline void fill_quaternion_int_S29_5(quat* v, u32 n)
{
	RandomWalk walk{};
	const quatz* table = T29i.s;
	u32 u = 0;
	while (u < n)
	{
		quatz q{};
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		q = table[walk()] * q;
		v[u++] = to_quatf(q, s29_5_norm);
	}
}

template <class RandomWalk> 
inline void fill_quaternion_int_5_29(quat* v, u32 n)
{
	RandomWalk walk{};
	quatz q{};
	const quatz* table = T5i.s;
	const quatz* table2 = T29i.s;
	u32 tt = 24;
	u64 norm = 1;
	u32 j = 0;
	u32 u = 0;
	while (u < n)
	{
		if (tt == 0)
		{
			q = {};
			tt = 24;
			norm = 1;
		}
		if (j == 30)
		{
			j = 0;
			q = table[walk()] * q;
			norm *= 5u;
			tt--;
			v[u++] = to_quatf(q, (f32)(1. / std::sqrt(static_cast<f64>(norm))));
		}
		else
		{
			v[u++] = to_quatf(table2[j++] * q, (f32)(1. / std::sqrt(static_cast<f64>(norm * 29))));
		}
	}
}





//GPU implementations

template <class Sampler>
inline quat sampler_polar_1_indexed()
{
	static Sampler sampler{};
	return polar_1(sampler);
}

template <class Sampler>
inline void fill_polar_1_indexed(quat* v, u32 n)
{
	Sampler sampler{};
	u32 u = 0;
	while(n--)
	{
		sampler.state = u;
		v[u++]= polar_1(sampler);
	}
}

template <class Sampler>
inline quat sampler_polar_2_indexed()
{
	static Sampler sampler{};
	return polar_2(sampler);
}

template <class Sampler>
inline void fill_polar_2_indexed(quat* v, u32 n)
{
	Sampler sampler{};
	u32 u = 0;
	while(u<n)
	{
		sampler.state = vec3_u32(u)+n*vec3_u32(0, 2, 3);
		v[u++]= polar_2(sampler);
	}
}

template <class Sampler> 
inline quat sampler_walk_6()
{
	static quat q{1,0,0,0};
	static Sampler sampler{};
	q = walk_6(q, sampler());
	return q;
}

template <class Sampler> 
inline void fill_walk_6(quat *v, u32 n)
{
	quat q{1,0,0,0};
	Sampler sampler{};
	u32 u{};
	while(u<n)
	{
		q = walk_6(q, sampler(u));
		*(v+u) = q;
		u++;
	}
}


void sampler_spherical_fibonacci(){}

void fill_spherical_fibonacci(quat* v, u32 n)
{
    spherical_fibonacci(v, n);
}

void sampler_hecke_spheres_5() {}
void fill_hecke_spheres_5(quat* v, u32 n)
{
    make_tree(T5.s, 5, v, n);
}

void sampler_partial_tree(){}
void sampler_partial_tree2(){}
void sampler_partial_tree3(){}
void sampler_partial_tree4(){}

void fill_partial_tree(quat*v, u32 n)
{
    auto tree_root = new quat[1024];

    u32 cur{};
    for(u32 u{}; u<1024; u++)
    {
        cur = u/125;
		tree_root[u] =  u < 750 ? quat{} : T5.s[0];
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/25 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/5 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
    }
    u32 u{};
    while(u<n)
    {
        vec4_u32 seed = vec4_u32(u) + n * vec4_u32{0, 2, 3, 5};
        seed = pcg4d(seed);
		*(v+u) = quat_mul(tree_root[seed.y  & 1023], tree_root[seed.x & 1023]);
        u++;
    }
    delete[] tree_root;
}


void fill_partial_tree2(quat*v, u32 n)
{
    auto tree_root = new quat[1024];

    u32 cur{};
    for(u32 u{}; u<1024; u++)
    {
        cur = u/125;
		tree_root[u] =  u < 750 ? quat{} : T5.s[0];
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/25 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/5 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
    }
    u32 u{};
    vec4_u32 seed{0};
    while(u<n)
    {
        seed = vec4_u32(u) + n * vec4_u32{0, 2, 3, 5};
        seed = pcg4d(seed);
		*(v+u) = quat_mul(tree_root[seed.z & 1023], quat_mul(tree_root[seed.y & 1023], tree_root[seed.x & 1023]));
        u++;
    }
    delete[] tree_root;
}

void fill_partial_tree3(quat*v, u32 n)
{
    auto tree_root = new quat[1024];

    u32 cur{};
    for(u32 u{}; u<1024; u++)
    {
        cur = u/125;
		tree_root[u] =  u < 750 ? quat{} : T5.s[0];
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/25 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/5 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
    }
    u32 u{};
    vec4_u32 seed{0};
    while(u<n)
    {
        seed = vec4_u32(u) + n * vec4_u32{0, 2, 3, 5};
        seed = pcg4d(seed);
		*(v+u) = quat_mul(tree_root[seed.w & 1023], (quat_mul(tree_root[seed.z & 1023], quat_mul(tree_root[seed.y & 1023], tree_root[seed.x & 1023]))));
        u++;
    }
    delete[] tree_root;
}

void fill_partial_tree4(quat*v, u32 n)
{
    auto tree_root = new quat[1024];

    u32 cur{};
    for(u32 u{}; u<1024; u++)
    {
        cur = u/125;
		tree_root[u] =  u < 750 ? quat{} : T5.s[0];
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/25 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/5 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
    }
    u32 u{};
    vec4_u32 seed{0};
    while(u<n)
    {
        seed = vec4_u32(u) + n * vec4_u32{0, 2, 3, 5};
        seed = pcg4d(seed);
		*(v+u) = quat_mul(tree_root[seed.x>>16 & 1023], 
                 quat_mul(tree_root[seed.w & 1023], 
                 quat_mul(tree_root[seed.z & 1023], 
                 quat_mul(tree_root[seed.y & 1023], tree_root[seed.x & 1023]))));
        u++;
    }
    delete[] tree_root;
}

void fill_partial_tree5(quat*v, u32 n)
{
    u32 m = 2048;
    auto tree_root = new quat[m];

    u32 cur{};
    for(u32 u{}; u<m; u++)
    {
		tree_root[u] =  u < 750 ? quat{} : 
                        u < 1500 ? T5.s[0] : 
                        u < 2250 ? T5.s[1] : 
                        u < 3000 ? T5.s[2] : T5.s[3];
        cur = u/125;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/25 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u/5 % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		cur = cur + 4 + u % 5;
		tree_root[u] = quat_mul( T5.s[ cur % 6] , tree_root[u]);
		tree_root[u] = quat_mul( T29.s[0] , tree_root[u]);
    }
    u32 u{};
    while(u<n)
    {
        vec4_u32 seed = vec4_u32(u) + n * vec4_u32{0, 2, 3, 5};
        seed = pcg4d(seed);
		*(v+u) = quat_mul(tree_root[seed.y  & m-1], tree_root[seed.x & m-1]);
        u++;
    }
    delete[] tree_root;
}
