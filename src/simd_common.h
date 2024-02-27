#pragma once
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#include "common.h"
#include "vec_math.h"
#include "quaternions.h"

struct vec2x4
{
	__m128 x;
	__m128 y;
};

struct vec2x8
{
	__m256 x;
	__m256 y;
};

struct vec2x16
{
	__m512 x;
	__m512 y;
};

struct vec3x8
{
	__m256 x;
	__m256 y;
	__m256 z;
};


struct vec4x8
{
	__m256 x,y,z,w;
};

struct mat3x8
{
	vec3x8 u;
	vec3x8 v;
	vec3x8 w;
};

struct quatx8
{
	__m256 r,x,y,z;

	quatx8(quat q) : 
		r(_mm256_set1_ps(q.r)), x(_mm256_set1_ps(q.x)), y(_mm256_set1_ps(q.y)), z(_mm256_set1_ps(q.z)) {}
	quatx8() : 
		r(_mm256_set1_ps(1.)), x(_mm256_set1_ps(0)), y(_mm256_set1_ps(0)), z(_mm256_set1_ps(0)) {}
	quatx8(__m256 r, __m256 x, __m256 y, __m256 z) : 
		r(r), x(x), y(y), z(z) {}
};

inline bool is_not_zero(vec2x4 p)
{
	return !_mm_test_all_zeros(_mm_castps_si128(p.x), _mm_set1_epi32(~0)) || !_mm_test_all_zeros(_mm_castps_si128(p.y), _mm_set1_epi32(~0));
}

inline vec2x4 reflect_x4(vec2x4 p)
{
	return { _mm_sub_ps(_mm_setzero_ps(), p.x), _mm_sub_ps(_mm_setzero_ps(), p.y) };
}

// can be replaced using _mm512_kor/and if need this
inline bool is_not_zero(vec2x16 p)
{
	return !_mm512_test_epi32_mask(_mm512_castps_si512(p.x), _mm512_setzero_si512()) || !_mm512_test_epi32_mask(_mm512_castps_si512(p.y), _mm512_setzero_si512());
}

const __m256 one_x8  = _mm256_set1_ps(1.f);


// Converts a struct of arrays to an array of structs
inline void soa_to_aof(mat3x8* uvw, vec4_f32* v_out)
{
	vec3x8& u = uvw->u;
	vec3x8& v = uvw->v;
	vec3x8& w = uvw->w;

	const __m256 zero = _mm256_castsi256_ps(_mm256_setzero_si256());	

	const __m256 u_xy_lo = _mm256_unpacklo_ps(u.x, u.y); // x76543210, y76543210 ->    y5x5y4x4 y1x1y0x0
	const __m256 u_xy_hi = _mm256_unpackhi_ps(u.x, u.y); // x76543210, y76543210 ->    y7x7y6x6 y3x3y2x2
	const __m256 u_zw_lo = _mm256_unpacklo_ps(u.z, zero);//                            00z500z4 00z100z0 
	const __m256 u_zw_hi = _mm256_unpackhi_ps(u.z, zero);//                            00z700z6 00z300z2 

	const __m256 u_xyzw_04 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b01'00'01'00); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  004z4y4x4  000z0y0x0
	const __m256 u_xyzw_15 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b11'10'11'10); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  005z5y5x5  001z1y0x1
	const __m256 u_xyzw_26 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b01'00'01'00); 
	const __m256 u_xyzw_37 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b11'10'11'10); 


	const __m256 v_xy_lo = _mm256_unpacklo_ps(v.x, v.y);
	const __m256 v_xy_hi = _mm256_unpackhi_ps(v.x, v.y);
	const __m256 v_zw_lo = _mm256_unpacklo_ps(v.z, zero);
	const __m256 v_zw_hi = _mm256_unpackhi_ps(v.z, zero);

	const __m256 v_xyzw_04 = _mm256_shuffle_ps(v_xy_lo, v_zw_lo, 0b01'00'01'00); 
	const __m256 v_xyzw_15 = _mm256_shuffle_ps(v_xy_lo, v_zw_lo, 0b11'10'11'10); 
	const __m256 v_xyzw_26 = _mm256_shuffle_ps(v_xy_hi, v_zw_hi, 0b01'00'01'00); 
	const __m256 v_xyzw_37 = _mm256_shuffle_ps(v_xy_hi, v_zw_hi, 0b11'10'11'10); 


	const __m256 w_xy_lo = _mm256_unpacklo_ps(w.x, w.y);
	const __m256 w_xy_hi = _mm256_unpackhi_ps(w.x, w.y);
	const __m256 w_zw_lo = _mm256_unpacklo_ps(w.z, zero);
	const __m256 w_zw_hi = _mm256_unpackhi_ps(w.z, zero);

	const __m256 w_xyzw_04 = _mm256_shuffle_ps(w_xy_lo, w_zw_lo, 0b01'00'01'00); 
	const __m256 w_xyzw_15 = _mm256_shuffle_ps(w_xy_lo, w_zw_lo, 0b11'10'11'10); 
	const __m256 w_xyzw_26 = _mm256_shuffle_ps(w_xy_hi, w_zw_hi, 0b01'00'01'00); 
	const __m256 w_xyzw_37 = _mm256_shuffle_ps(w_xy_hi, w_zw_hi, 0b11'10'11'10); 

	const __m256 u0v0 = _mm256_permute2f128_ps(u_xyzw_04, v_xyzw_04, 0x20);
	const __m256 w0u1 = _mm256_permute2f128_ps(w_xyzw_04, u_xyzw_15, 0x20);
	const __m256 v1w1 = _mm256_permute2f128_ps(v_xyzw_15, w_xyzw_15, 0x20);

	const __m256 u2v2 = _mm256_permute2f128_ps(u_xyzw_26, v_xyzw_26, 0x20);
	const __m256 w2u3 = _mm256_permute2f128_ps(w_xyzw_26, u_xyzw_37, 0x20);
	const __m256 v3w3 = _mm256_permute2f128_ps(v_xyzw_37, w_xyzw_37, 0x20);

	const __m256 u4v4 = _mm256_permute2f128_ps(u_xyzw_04, v_xyzw_04, 0x31);
	const __m256 w4u5 = _mm256_permute2f128_ps(w_xyzw_04, u_xyzw_15, 0x31);
	const __m256 v5w5 = _mm256_permute2f128_ps(v_xyzw_15, w_xyzw_15, 0x31);

	const __m256 u6v6 = _mm256_permute2f128_ps(u_xyzw_26, v_xyzw_26, 0x31);
	const __m256 w6u7 = _mm256_permute2f128_ps(w_xyzw_26, u_xyzw_37, 0x31);
	const __m256 v7w7 = _mm256_permute2f128_ps(v_xyzw_37, w_xyzw_37, 0x31);

	f32* v_out_p = (f32*)v_out;

	_mm256_store_ps(v_out_p,      u0v0);
	_mm256_store_ps(v_out_p+2*4,  w0u1);
	_mm256_store_ps(v_out_p+4*4,  v1w1);

	_mm256_store_ps(v_out_p+6*4,  u2v2);
	_mm256_store_ps(v_out_p+8*4,  w2u3);
	_mm256_store_ps(v_out_p+10*4, v3w3);

	_mm256_store_ps(v_out_p+12*4, u4v4);
	_mm256_store_ps(v_out_p+14*4, w4u5);
	_mm256_store_ps(v_out_p+16*4, v5w5);

	_mm256_store_ps(v_out_p+18*4, u6v6);
	_mm256_store_ps(v_out_p+20*4, w6u7);
	_mm256_store_ps(v_out_p+22*4, v7w7);
}

inline void soa_to_aof(vec3x8* u, vec4_f32* v_out)
{
	const __m256 zero = _mm256_castsi256_ps(_mm256_setzero_si256());	

	const __m256 u_xy_lo = _mm256_unpacklo_ps(u->x, u->y); // x76543210, y76543210 ->    y5x5y4x4 y1x1y0x0
	const __m256 u_xy_hi = _mm256_unpackhi_ps(u->x, u->y); // x76543210, y76543210 ->    y7x7y6x6 y3x3y2x2
	const __m256 u_zw_lo = _mm256_unpacklo_ps(u->z, zero);//                            00z500z4 00z100z0 
	const __m256 u_zw_hi = _mm256_unpackhi_ps(u->z, zero);//                            00z700z6 00z300z2 

	const __m256 u_xyzw_04 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b01'00'01'00); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  004z4y4x4  000z0y0x0
	const __m256 u_xyzw_15 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b11'10'11'10); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  005z5y5x5  001z1y0x1
	const __m256 u_xyzw_26 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b01'00'01'00); 
	const __m256 u_xyzw_37 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b11'10'11'10); 

	const __m256 w0u1 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x20);
	const __m256 w2u3 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x20);
	const __m256 w4u5 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x31);
	const __m256 w6u7 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x31);

	f32* v_out_p = (f32*)v_out;

	_mm256_store_ps(v_out_p,  w0u1);
	_mm256_store_ps(v_out_p+8,  w2u3);
	_mm256_store_ps(v_out_p+16, w4u5);
	_mm256_store_ps(v_out_p+24, w6u7);
}

inline void soa_to_aof(quatx8* u, vec4_f32* v_out)
{
	const __m256 u_xy_lo = _mm256_unpacklo_ps(u->x, u->y); // x76543210, y76543210 ->    y5x5y4x4 y1x1y0x0
	const __m256 u_xy_hi = _mm256_unpackhi_ps(u->x, u->y); // x76543210, y76543210 ->    y7x7y6x6 y3x3y2x2
	const __m256 u_zw_lo = _mm256_unpacklo_ps(u->z, u->r);
	const __m256 u_zw_hi = _mm256_unpackhi_ps(u->z, u->r);

	const __m256 u_xyzw_04 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b01'00'01'00); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  004z4y4x4  000z0y0x0
	const __m256 u_xyzw_15 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b11'10'11'10); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  005z5y5x5  001z1y0x1
	const __m256 u_xyzw_26 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b01'00'01'00); 
	const __m256 u_xyzw_37 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b11'10'11'10); 

	const __m256 w0u1 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x20);
	const __m256 w2u3 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x20);
	const __m256 w4u5 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x31);
	const __m256 w6u7 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x31);

	f32* v_out_p = (f32*)v_out;

	_mm256_store_ps(v_out_p,  w0u1);
	_mm256_store_ps(v_out_p+8,  w2u3);
	_mm256_store_ps(v_out_p+16, w4u5);
	_mm256_store_ps(v_out_p+24, w6u7);
}

inline void soa_to_aof(quatx8* u, quat* v_out)
{
	const __m256 u_xy_lo = _mm256_unpacklo_ps(u->r, u->x); // x76543210, y76543210 ->    y5x5y4x4 y1x1y0x0
	const __m256 u_xy_hi = _mm256_unpackhi_ps(u->r, u->x); // x76543210, y76543210 ->    y7x7y6x6 y3x3y2x2
	const __m256 u_zw_lo = _mm256_unpacklo_ps(u->y, u->z);
	const __m256 u_zw_hi = _mm256_unpackhi_ps(u->y, u->z);

	const __m256 u_xyzw_04 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b01'00'01'00); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  004z4y4x4  000z0y0x0
	const __m256 u_xyzw_15 = _mm256_shuffle_ps(u_xy_lo, u_zw_lo, 0b11'10'11'10); // y5x5y4x4 y1x1y0x0, 00z500z4 00z100z0  ->  005z5y5x5  001z1y0x1
	const __m256 u_xyzw_26 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b01'00'01'00); 
	const __m256 u_xyzw_37 = _mm256_shuffle_ps(u_xy_hi, u_zw_hi, 0b11'10'11'10); 

	const __m256 w0u1 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x20);
	const __m256 w2u3 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x20);
	const __m256 w4u5 = _mm256_permute2f128_ps(u_xyzw_04, u_xyzw_15, 0x31);
	const __m256 w6u7 = _mm256_permute2f128_ps(u_xyzw_26, u_xyzw_37, 0x31);

	f32* v_out_p = (f32*)v_out;

	_mm256_store_ps(v_out_p,    w0u1);
	_mm256_store_ps(v_out_p+8,  w2u3);
	_mm256_store_ps(v_out_p+16, w4u5);
	_mm256_store_ps(v_out_p+24, w6u7);
}
