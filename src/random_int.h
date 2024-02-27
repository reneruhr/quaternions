#pragma once

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif

#include <cmath>
#include <cstdint>
#include "quaternions.h"

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

inline u32 lcg(u32 seed)
{
	return seed * 747796405u + 2891336453u;
}

inline u32 lcg_xsh(u32& seed)
{
	seed = seed * 747796405u + 2891336453u;
	return (seed >> 16) ^ seed;
}

consteval u32 lcg_a_n(u32 a, u32 n)
{
	u32 A = 1;
	while(n--)
		A*=a;
	return A;
}

consteval u32 lcg_c_n(u32 a, u32 c, u32 n)
{
	u32 C = 0;
	while(n--)
		C=a*C+c;
	return C;
}

inline u32 lcg8(u32 seed)
{
	return seed * lcg_a_n(747796405u, 8) + lcg_c_n(747796405u, 2891336453u, 8);
}

inline u32 lcg8_xsh(u32& seed)
{
	seed *= lcg_a_n(747796405u, 8);
	seed += lcg_c_n(747796405u, 2891336453u, 8);
	return (seed >>16) ^ seed;
}

// PCG rng O'Neil (www.pcg-random.org)
constexpr u64 pcg32_init[2]   {0x853c49e6748fea9b, 0xda3e39cb94b95bdb};
constexpr u64 pcg32_multiplier{0x5851f42d4c957f2d};

struct pcg_state
{
	u64 seed{0};
	u64 c{0}; 
};

inline u32 pcg32(pcg_state& state)
{
	u32 a = (u32)(((state.seed >> 18) ^ state.seed) >> 27);
	u32 b = (u32)(state.seed >> 59);
	state.seed = state.seed * pcg32_multiplier + state.c;
	return (a >> b) | (a >> ((-b) & 31));
}

inline pcg_state init_pcg32(u64 seed=pcg32_init[0], u64 stream_id=pcg32_init[1])
{
	pcg_state state;
  	state.c = {(stream_id << 1) | 1};
	pcg32(state);
	state.seed+=seed;
	pcg32(state);
	return state;
}

// Marsaglia Xorshift RNG's 2003    Journal of Statistical Software
struct xorshift_state
{
	u32 seed{2463534242};
};

inline u32 xorshift(xorshift_state& state)
{
	u32& x = state.seed;
	x^= x <<13;
	x^= x >>17;
	x^= x << 5;
	return x;
}

inline u32 xorshift(u32& x)
{
	x^= x <<13;
	x^= x >>17;
	x^= x <<5;
	return x;
}

inline u32 xorshift2(u32& x)
{
	x^= x <<5;
	x^= x >>9;
	x^= x <<7;
	return x;
}

inline f32 xorshift_f32(xorshift_state& state)
{
	u32& x = state.seed;
	x^= x <<13;
	x^= x >>17;
	x^= x << 5;
	return x*0x1p-32f;
}

struct xorwow_state
{
	u32 s[5] = {123456789, 362436069,521288629,88675123,5783321};
	u32 c{2463534242};
};

inline u32 xorwow(xorwow_state& state)
{
	auto&s = state.s;
	u32 x = s[0]^(s[0] >>2);
	s[0] = s[1];
	s[1] = s[2];
	s[2] = s[3];
	s[3] = s[4];
	s[4] = (s[4]^(s[4]<<4)) ^ (x^(x<<1));
	state.c+=362437;
	return s[4]+state.c;
}

struct xorshift64_state
{
	u64 seed{88172645463325252};
};

inline u64 xorshift(xorshift64_state& state)
{
	u64& x = state.seed;
	x^= x <<13;
	x^= x >> 7;
	x^= x <<17;
	return x;
}

inline f32 rand_f32(u32 u)
{
	return u*0x1p-32f;
}

inline f64 rand_f64(u64 u)
{
	return u*0x1p-64f;
}

//GPU implementations

// https://www.pcg-random.org/
inline u32 pcg(u32 v)
{
	u32 state = v * 747796405u + 2891336453u;
	u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

// http://www.jcgt.org/published/0009/03/02/
inline vec3_u32 pcg3d(vec3_u32 v) 
{
    v = 1013904223u + 1664525u * v;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v ^= v >> 16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}

// http://www.jcgt.org/published/0009/03/02/
inline vec4_u32 pcg4d(vec4_u32 v)
{
    v = 1013904223u + 1664525u * v;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    v ^= v >> 16u;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    return v;
}

struct pcg_indexed_sampler
{
	u32 state{};
	u32 operator()(u32 a){ return state = pcg(a); }
	u32 operator()(){ return state = pcg(state); }
};

struct pcg3_indexed_sampler
{
	vec3_u32 state{0};
	vec3_u32 operator()(vec3_u32 v){ return state = pcg3d(v);	}
	vec3_u32 operator()(){ return state = pcg3d(state);			}
};

struct pcg4_indexed_sampler
{
	vec4_u32 state{0};
	vec4_u32 operator()(vec4_u32 v){ return state = pcg4d(v);	}
	vec4_u32 operator()(){ return state = pcg4d(state);			}
};

struct xorshift_sampler
{
	xorshift_state state;
	u32 operator()(){ return xorshift(state); }
};

struct xorshift2_sampler
{
	u32 seed[2] {2463534242,123456789};
	u32 operator()(){ return xorshift (seed[0]); }
	u32 operator()(int){ return xorshift2(seed[1]); }
};

struct xorwow_sampler
{
	xorwow_state state;
	u32 operator()(){ return xorwow(state); }
};

struct xorshift_sampler_f32
{
	xorshift_state state;
	f32 operator()(){ return rand_f32(xorshift(state)); }
};

inline u32 xorshift_bounded(xorshift_state& state, u32 bound)
{
	u32 threshold = -bound %bound;
	for(;;){
		u32 u = xorshift(state);
		if(u >= threshold)
			return u%bound;
	}
};

void fill_xorwow(u32* data, u32 n)
{
	xorwow_sampler sampler{};
	for (u32 u{}; u < n; u++)
		data[u] = sampler();

}


template <u32 n>
struct bernoulli_walk
{
	u32 seed{2463534242};
	u32 operator()() {	return xorshift(seed) % n;	}
};

template <u32 mask> // mask = (1 << num_bits)-1;
struct bernoulli_walk_no_modulus
{
	u32 seed{2463534242};
	u32 operator()(){	return xorshift(seed) & mask; }
};

// Backtrack if choice sequence a,b satisfies b=a+(n/2) mod n
// Enforce non-backtracking by storing last choice, go to its inverse
// located n/2 slots further up
// and sampling among the n-1 following entries
template <u32 n>
struct no_backtracking_walk 
{
	u32 seed{2463534242};
	u32 last{0};

	u32 operator()()
	{
		last = (last+(n/2+1)+(xorshift(seed) % (n-1) )) %n;	
		return last;
	}
};


// scalar implementation of simd lcg using strides
struct no_backtracking_walk_biased_scalar
{
	u32 seed[8];
	u32 last[8];
	
	no_backtracking_walk_biased_scalar()
	{
		u32 s= 1;
		for(u32 i=0; i<8; i++)
		{
			s		= lcg(s);
			seed[i] = s;
			u32 l	= ((s >> 16) ^ s) & 7;
			last[i] = l> 5 ? l- 2 : l;
		}
	}

	u32 operator()()
	{
		static u32 c = 0;
		u32 x = lcg8_xsh(seed[c]) & 7;
		x = x > 5 ? x-2 : x;
		const u8 forbidden = last[c] & 1 ? last[c] - 1 : last[c] + 1;
		last[c] = forbidden == x ? last[c] : x;
		return last[c++]; 
	}
};