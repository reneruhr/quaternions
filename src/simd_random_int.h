#pragma once
#include "common.h"
#include "random_int.h"

inline __m128i xorshift(__m128i& x)
{
	
	__m128i t1 = _mm_slli_epi32(x, 13);
			 x = _mm_xor_si128(x, t1);

	__m128i t2 = _mm_srli_epi32(x, 17);
	         x = _mm_xor_si128(x, t2);

	__m128i t3 = _mm_slli_epi32(x, 5);
		     x = _mm_xor_si128(x, t3);

	return x;
}

inline __m128i xorshift2(__m128i& x)
{
	
	__m128i t1 = _mm_slli_epi32(x, 5);
	x		   = _mm_xor_si128(x, t1);

	__m128i t2 = _mm_srli_epi32(x, 9);
	x		   = _mm_xor_si128(x, t2);

	__m128i t3 = _mm_slli_epi32(x, 7);
	x		   = _mm_xor_si128(x, t3);

	return x;
}

inline __m256i xorshift(__m256i& x)
{
	
	__m256i t1 = _mm256_slli_epi32(x, 13);
			 x = _mm256_xor_si256(x, t1);

	__m256i t2 = _mm256_srli_epi32(x, 17);
	         x = _mm256_xor_si256(x, t2);

	__m256i t3 = _mm256_slli_epi32(x, 5);
		     x = _mm256_xor_si256(x, t3);

	return x;
}


void set_first_eight(u32 *s)
{
	u32 seed = 1;
	for(u32 i=0; i<8; i++)
	{
		seed = lcg(seed);
		s[i] = seed;
	}
}

#if defined(_MSC_VER) // MSVC Compiler
    #define ALIGNAS_256 __declspec(align(32))
#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang
    #define ALIGNAS_256 __attribute__((aligned(32)))
#else
    #error "Unsupported compiler"
#endif

__m256i make_seed_lcg()
{
	ALIGNAS_256 u32 s_[8];
	set_first_eight(s_);
	__m256i seed8 = _mm256_load_si256((__m256i*)s_);
	return seed8;
}

__m256i make_seed_xorwow()
{
	xorwow_sampler sampler;
	__m256i seed8 = _mm256_set_epi32(sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler(), sampler());
	return seed8;
}

inline __m256i lcg(__m256i x)
{
	return _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(747796405u), x), _mm256_set1_epi32(2891336453u));
}

constexpr u32 A = lcg_a_n(747796405u, 8) ;
constexpr u32 C = lcg_c_n(747796405u, 2891336453u, 8);

inline __m256i lcg8(__m256i x)
{
	return _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(A), x), _mm256_set1_epi32(C));
}

inline __m256i lcg_xsh(__m256i& x)
{
	x = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(A), x), _mm256_set1_epi32(C));
	const __m256i y  = _mm256_srli_epi32(x, 16);
	return  _mm256_xor_si256(x, y);
}

struct simd_xorshift_sampler
{
	__m256i state;
	simd_xorshift_sampler() { state = make_seed_xorwow(); }
	__m256i operator()(){ return xorshift(state); }
};

struct simd_lcg_xsh_sampler
{
	__m256i state;
	simd_lcg_xsh_sampler() { state = make_seed_lcg(); }
	__m256i operator()(){ return lcg_xsh(state); }
};


template <class Sampler>
struct walk_biased
{
	Sampler sampler{};
	__m256i operator()()
	{
		const __m256i mask = _mm256_set1_epi32(7);
		return _mm256_and_si256(sampler(), mask);
	}
};

template <class Sampler>
struct no_backtracking_walk_biased
{
	Sampler sampler{};
	__m256i last;

//Assumes that the backtracking choice is when the lsb is flipped.
//If we were to backtrack, we instead walk in the same direction as the previous
//Adapted for 6 choices causing it to be biased since choice 5 and 6 are twice as likely.
	
	no_backtracking_walk_biased()
	{
		const __m256i seed = sampler();

		const __m256i one =_mm256_set1_epi32(1);
		const __m256i two =_mm256_add_epi32(one,one);
		const __m256i five = _mm256_set1_epi32(5);
		const __m256i seven= _mm256_add_epi32(two,five);

		const __m256i y  = _mm256_srli_epi32(seed, 16);
		__m256i x = _mm256_xor_si256(seed, y);
		x = _mm256_and_si256(x, seven);
		const __m256i too_big = _mm256_cmpgt_epi32(x, five);
		x = _mm256_blendv_epi8(x, _mm256_sub_epi32(x, two), too_big);
		last = x;
	}

	__m256i operator()()
	{
		const __m256i one =_mm256_set1_epi32(1);
		const __m256i two =_mm256_add_epi32(one,one);
		const __m256i five = _mm256_set1_epi32(5);
		const __m256i seven= _mm256_add_epi32(two,five);

		const __m256i sign_bit = _mm256_and_si256(last, one);
		// blendv wants hsb not lsb! so need another transform
		const __m256i sign_bit_h = _mm256_cmpeq_epi32(sign_bit, one);
		const __m256i forbidden = _mm256_blendv_epi8(_mm256_add_epi32(last,one), _mm256_sub_epi32(last, one), sign_bit_h); // flip lsb which encodes sign

		__m256i x = _mm256_and_si256(sampler(), seven);
		const __m256i too_big = _mm256_cmpgt_epi32(x, five);
		x = _mm256_blendv_epi8(x, _mm256_sub_epi32(x, two), too_big);
		const __m256i is_backtracking = _mm256_cmpeq_epi32(x, forbidden);
		last = _mm256_blendv_epi8(x, last, is_backtracking);
		return last;
	}
};

