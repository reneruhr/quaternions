#pragma once
#include "simd_quaternions.h"

/*
    This file includes code for quaternion simd by throughput (x8 = 256 bits) and by compression (x4 = 128 bits)
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////  x8  ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const __m256 s5_x8 =_mm256_set1_ps(1.f/std::sqrt(5.f));
const __m256 s5_2_x8 =_mm256_add_ps(s5_x8,s5_x8);
const __m256i onei_x8 = _mm256_set1_epi32(1);

// s_i = b2b1b0   b0 = sign.  b2b1 in {0,1,2,3} where 2 and 3 both get mapped to choice 2
inline quatx8 construct_T5(__m256i s)
{
	const __m256 a = _mm256_xor_ps(s5_2_x8, _mm256_castsi256_ps(_mm256_slli_epi32(s,31))) ;  // Use 0th bit for sign

	s = _mm256_srli_epi32(s, 1);
	const __m256 x = _mm256_and_ps(a, _mm256_castsi256_ps(
							    _mm256_cmpeq_epi32(s,_mm256_setzero_si256())));        // == 0?
	const __m256 y = _mm256_and_ps(a, _mm256_castsi256_ps(
							    _mm256_cmpeq_epi32(s,onei_x8)));   // == 1? 
	s = _mm256_srli_epi32(s, 1);
	const __m256 z = _mm256_and_ps(a, _mm256_castsi256_ps(
							    _mm256_cmpeq_epi32(s,onei_x8)));   // >= 2?

	return { s5_x8, x, y ,z };
}


template <class Sampler>
inline quatx8 marsaglia_polar_simd(Sampler& sampler)
{	
	__m256 phi =  _mm256_mul_ps(_mm256_set1_ps(tau_uint_max_inv), _mm256_cvtepu32_ps(sampler())) ;
	__m256 r   =  _mm256_mul_ps(_mm256_set1_ps(0x1p-32f), _mm256_cvtepu32_ps(sampler()));
	__m256 phi2 =  _mm256_mul_ps(_mm256_set1_ps(tau_uint_max_inv), _mm256_cvtepu32_ps(sampler())) ;
	__m256 t = _mm256_sqrt_ps(_mm256_sub_ps(one_x8,r));

	r = _mm256_sqrt_ps(r);
	__m256 cos;
	__m256 sin = _mm256_sincos_ps(&cos, phi);
	__m256 cos2;
	__m256 sin2 = _mm256_sincos_ps(&cos2, phi2);
	__m256 x = _mm256_mul_ps(sin, r);
	__m256 y = _mm256_mul_ps(cos, r);
	__m256 x2 = _mm256_mul_ps(sin2, t);
	__m256 y2 = _mm256_mul_ps(cos2, t);

  	return {x,y,x2,y2};
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////  x4  ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct S_5_simd_r
{
    const f32 s5 = 1.f/std::sqrt(5.f);

	mm_quat s[8] { 			
 	{.q=			s5*quat{ 2,0,0,1}},
	{.q=			s5*quat{0, 2,0,1}},
	{.q=			s5*quat{0,0, 2,1}},
	{.q=			s5*quat{-2,0,0,1}},
	{.q=			s5*quat{0,-2,0,1}},
	{.q=			s5*quat{0,0,-2,1}},
 	{.q=			s5*quat{ 2,0,0,1}},
	{.q=			s5*quat{0, 2,0,1}},
	}; 
};

struct S_29_simd_r
{
    const f32 s29 = 1.f/std::sqrt(29.f);

	mm_quat s[30] {
		{.q=    s29*quat{3,0,-2,-4}},
		{.q=    s29*quat{3,0,2,-4}},
		{.q=	s29*quat{3,0,-4,-2}},
		{.q=	s29*quat{3,0,4,-2}},
		{.q=	s29*quat{3,-2,0,-4}},
		{.q=	s29*quat{3,-2,0,4}},
		{.q=	s29*quat{3,-2,-4,0}},
		{.q=	s29*quat{3,2,-4,0}},
		{.q=	s29*quat{3,-4,0,-2}},
		{.q=	s29*quat{3,-4,0,2}},
		{.q=	s29*quat{3,4,-2,0}},
		{.q=	s29*quat{3,-4,-2,0}},
		{.q=	s29*quat{5,0,0,-2}},
		{.q=	s29*quat{5,0,-2,0}},
		{.q=	s29*quat{5,-2,0,0}},
		{.q=	s29*quat{3,0,2,4}},
		{.q=	s29*quat{3,0,-2,4}},
		{.q=	s29*quat{3,0,4,2}},
		{.q=	s29*quat{3,0,-4,2}},
		{.q=	s29*quat{3,2,0,4}},
		{.q=	s29*quat{3,2,0,-4}},
		{.q=	s29*quat{3,2,4,0}},
		{.q=	s29*quat{3,-2,4,0}},
		{.q=	s29*quat{3,4,0,2}},
		{.q=	s29*quat{3,4,0,-2}},
		{.q=	s29*quat{3,4,2,0}},
		{.q=	s29*quat{3,-4,2,0}},
		{.q=	s29*quat{5,0,0,2}},
		{.q=	s29*quat{5,0,2,0}},
		{.q=	s29*quat{5,2,0,0}},
	};
};

struct S_17_simd_r
{
    const f32 s17 = 1.f/std::sqrt(17.f);

	mm_quat s[18] {
		{.q = s17*quat{1,0,0,4}},
		{.q = s17*quat{1,0,4,0}},
		{.q = s17*quat{1,4,0,0}},
		{.q = s17*quat{3,0,2,2}},
		{.q = s17*quat{3,0,-2,2}},
		{.q = s17*quat{3,2,0,2}},
		{.q = s17*quat{3,-2,0,2}},
		{.q = s17*quat{3,2,2,0}},
		{.q = s17*quat{3,-2,2,0}},
		{.q = s17*quat{1,0,0,-4}},
		{.q = s17*quat{1,0,-4,0}},
		{.q = s17*quat{1,-4,0,0}},
		{.q = s17*quat{3,0,-2,-2}},
		{.q = s17*quat{3,0,2,-2}},
		{.q = s17*quat{3,-2,0,-2}},
		{.q = s17*quat{3,2,0,-2}},
		{.q = s17*quat{3,-2,-2,0}},
		{.q = s17*quat{3,2,-2,0}}
	};
};

const S_5_simd_r mm_T5r;
const S_29_simd_r mm_T29r;
const S_17_simd_r mm_T17r;
	