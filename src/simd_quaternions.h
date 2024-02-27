#pragma once

#include "common.h"
#include "vec_math.h"
#include "simd_common.h"
#include "quaternions.h"

#include <xmmintrin.h>
#include <immintrin.h> // AVX2
#include <pmmintrin.h> // hadd
#include <smmintrin.h> // blend, dp

/*
    This file includes code for quaternion simd by throughput (x8 = 256 bits) and by compression (x4 = 128 bits)
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////  x8  ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


inline quatx8 operator*(quatx8 a, quatx8 b)
{
	const __m256 r = _mm256_fmsub_ps(a.r, b.r, 
			   _mm256_fmadd_ps(a.x, b.x, 
			   _mm256_fmadd_ps(a.y, b.y, _mm256_mul_ps(a.z, b.z))));
		
	const __m256 x = _mm256_fmadd_ps(a.r, b.x, 
					 _mm256_fmadd_ps(b.r, a.x, 
					 _mm256_fmsub_ps(a.y, b.z, _mm256_mul_ps(a.z, b.y))));

	const __m256 y = _mm256_fmadd_ps(a.r, b.y, 
					 _mm256_fmadd_ps(b.r, a.y, 
					 _mm256_fmsub_ps(a.z, b.x, _mm256_mul_ps(a.x, b.z))));

	const __m256 z = _mm256_fmadd_ps(a.r, b.z, 
					 _mm256_fmadd_ps(b.r, a.z, 
					 _mm256_fmsub_ps(a.x, b.y, _mm256_mul_ps(a.y, b.x))));
	
	return { r,x,y,z };
}

void print(quatx8 q)
{
	f32 *r = new f32[8];
	f32 *x = new f32[8];
	f32 *y = new f32[8];
	f32 *z = new f32[8];

	_mm256_store_ps(r,  q.r);
	_mm256_store_ps(x,  q.x);
	_mm256_store_ps(y,  q.y);
	_mm256_store_ps(z,  q.z);

	vec4_f32 *v = new vec4_f32[8];

	for(u32 i=0; i<8; i++)
	{
		v[i] = {r[i], x[i], y[i], z[i]};	
		print(v[i]);
	}

	delete[] r;
	delete[] x;
	delete[] y;
	delete[] z;
	delete[] v;
}

inline quatx8 normalize(quatx8 a)
{
	const __m256 d = _mm256_rsqrt_ps(_mm256_fmadd_ps(a.r, a.r, _mm256_fmadd_ps(a.x, a.x, _mm256_fmadd_ps(a.y, a.y, _mm256_mul_ps(a.z, a.z)))));

	return { _mm256_mul_ps(a.r,d), _mm256_mul_ps(a.x,d), _mm256_mul_ps(a.y,d), _mm256_mul_ps(a.z,d) };

}

const __m256 two_x8  = _mm256_set1_ps(2.f);
const __m256 ntwo_x8 = _mm256_set1_ps(-2.f);

inline mat3x8 rotation(quatx8 q)
{
	const __m256 x2 = _mm256_mul_ps(q.x,q.x);
	const __m256 y2 = _mm256_mul_ps(q.y,q.y);
	const __m256 z2 = _mm256_mul_ps(q.z,q.z);

	const __m256 xy = _mm256_mul_ps(q.x,q.y);
	const __m256 zy = _mm256_mul_ps(q.z,q.y);
	const __m256 xz = _mm256_mul_ps(q.x,q.z);

	const __m256 rx = _mm256_mul_ps(q.r,q.x);
	const __m256 ry = _mm256_mul_ps(q.r,q.y);
	const __m256 rz = _mm256_mul_ps(q.r,q.z);

	const vec3x8 u{ 
			  .x = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(y2,z2), one_x8),
			  .y = _mm256_mul_ps(two_x8, _mm256_sub_ps(xy,rz)),
			  .z = _mm256_mul_ps(two_x8, _mm256_add_ps(xz,ry))};

	const vec3x8 v{ 
		      .x = _mm256_mul_ps(two_x8, _mm256_add_ps(xy,rz)),
			  .y = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(x2,z2), one_x8),
			  .z = _mm256_mul_ps(two_x8, _mm256_sub_ps(zy,rx))};

	const vec3x8 w{ 
		      .x = _mm256_mul_ps(two_x8, _mm256_sub_ps(xz,ry)),
			  .y = _mm256_mul_ps(two_x8, _mm256_add_ps(zy,rx)),
	          .z = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(x2,y2), one_x8)};

	return { u, v, w };
}

inline vec3x8 vec3(quatx8 q)
{
	const __m256 y2 = _mm256_mul_ps(q.y,q.y);
	const __m256 z2 = _mm256_mul_ps(q.z,q.z);

	const __m256 xy = _mm256_mul_ps(q.x,q.y);
	const __m256 xz = _mm256_mul_ps(q.x,q.z);

	const __m256 ry = _mm256_mul_ps(q.r,q.y);
	const __m256 rz = _mm256_mul_ps(q.r,q.z);

	const vec3x8 u{ 
			  .x = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(y2,z2), one_x8),
			  .y = _mm256_mul_ps(two_x8, _mm256_sub_ps(xy,rz)),
			  .z = _mm256_mul_ps(two_x8, _mm256_add_ps(xz,ry))};
	return u;
}

inline void quat8_to_vec4(quatx8 q, vec4_f32* v_out)
{
	const __m256 x2 = _mm256_mul_ps(q.x,q.x);
	const __m256 y2 = _mm256_mul_ps(q.y,q.y);
	const __m256 z2 = _mm256_mul_ps(q.z,q.z);

	const __m256 xy = _mm256_mul_ps(q.x,q.y);
	const __m256 zy = _mm256_mul_ps(q.z,q.y);
	const __m256 xz = _mm256_mul_ps(q.x,q.z);

	const __m256 rx = _mm256_mul_ps(q.r,q.x);
	const __m256 ry = _mm256_mul_ps(q.r,q.y);
	const __m256 rz = _mm256_mul_ps(q.r,q.z);

	const vec3x8 u{ 
			  .x = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(y2,z2), one_x8),
			  .y = _mm256_mul_ps(two_x8, _mm256_sub_ps(xy,rz)),
			  .z = _mm256_mul_ps(two_x8, _mm256_add_ps(xz,ry))};

	const vec3x8 v{ 
		      .x = _mm256_mul_ps(two_x8, _mm256_add_ps(xy,rz)),
			  .y = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(x2,z2), one_x8),
			  .z = _mm256_mul_ps(two_x8, _mm256_sub_ps(zy,rx))};

	const vec3x8 w{ 
		      .x = _mm256_mul_ps(two_x8, _mm256_sub_ps(xz,ry)),
			  .y = _mm256_mul_ps(two_x8, _mm256_add_ps(zy,rx)),
	          .z = _mm256_fmadd_ps(ntwo_x8, _mm256_add_ps(x2,y2), one_x8)};

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////  x4  ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/questions/18542892/how-to-multiply-two-quaternions-with-minimal-instructions
inline __m128 mm_cross4_ps(__m128 xyzw, __m128 abcd)
{
	__m128 wzyx = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(0, 1, 2, 3));
	__m128 baba = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(0, 1, 0, 1));
	__m128 dcdc = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(2, 3, 2, 3));

	__m128 ZnXWY = _mm_hsub_ps(_mm_mul_ps(xyzw, baba), _mm_mul_ps(wzyx, dcdc));
	__m128 XZYnW = _mm_hadd_ps(_mm_mul_ps(xyzw, dcdc), _mm_mul_ps(wzyx, baba));

	__m128 XZWY = _mm_addsub_ps(_mm_shuffle_ps(XZYnW, ZnXWY, _MM_SHUFFLE(3, 2, 1, 0)),
		_mm_shuffle_ps(ZnXWY, XZYnW, _MM_SHUFFLE(2, 3, 0, 1)));

	return _mm_shuffle_ps(XZWY, XZWY, _MM_SHUFFLE(2, 1, 3, 0));
}

inline __m128 normalize(__m128 q)
{
    __m128 norm2= _mm_dp_ps(q, q, 0xFF);
    __m128 rsqrt = _mm_rsqrt_ps(norm2);

    return _mm_mul_ps(q, rsqrt);
}

union mm_quat
{
	__m128 mm;
	fquat32 q;
};

inline void print(mm_quat q)
{
	printf("(%f, %f, %f, %f)\n", q.q.r, q.q.x, q.q.y, q.q.z);
}



/*
===========================================================================

Doom 3 BFG Edition GPL Source Code
Copyright (C) 1993-2012 id Software LLC, a ZeniMax Media company. 

	This file is part of the Doom 3 BFG Edition GPL Source Code ("Doom 3 BFG Edition Source Code").  

	Doom 3 BFG Edition Source Code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

  Doom 3 BFG Edition Source Code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Doom 3 BFG Edition Source Code.  If not, see <http://www.gnu.org/licenses/>.

	  In addition, the Doom 3 BFG Edition Source Code is also subject to certain additional terms. You should have received a copy of these additional terms immediately following the terms and conditions of the GNU General Public License which accompanied the Doom 3 BFG Edition Source Code.  If not, please request a copy in writing from id Software at the address below.

	  If you have questions concerning this license or the applicable additional terms, you may contact in writing id Software LLC, c/o ZeniMax Media Inc., Suite 120, Rockville, Maryland 20850 USA.

	  ===========================================================================
	  */

#include <xmmintrin.h>
#include <emmintrin.h>


#if !defined( R_SHUFFLE_D )
#define R_SHUFFLE_D( x, y, z, w )	(( (w) & 3 ) << 6 | ( (z) & 3 ) << 4 | ( (y) & 3 ) << 2 | ( (x) & 3 ))
#endif

// make the intrinsics "type unsafe"
union __m128c {
				__m128c() {}
				__m128c( __m128 f ) { m128 = f; }
				__m128c( __m128i i ) { m128i = i; }
	operator	__m128() { return m128; }
	operator	__m128i() { return m128i; }
	__m128		m128;
	__m128i		m128i;
};

#define _mm_madd_ps( a, b, c )				_mm_add_ps( _mm_mul_ps( (a), (b) ), (c) )
#define _mm_nmsub_ps( a, b, c )				_mm_sub_ps( (c), _mm_mul_ps( (a), (b) ) )
#define _mm_splat_ps( x, i )				__m128c( _mm_shuffle_epi32( __m128c( x ), _MM_SHUFFLE( i, i, i, i ) ) )
#define _mm_perm_ps( x, perm )				__m128c( _mm_shuffle_epi32( __m128c( x ), perm ) )
#define _mm_sel_ps( a, b, c )  				_mm_or_ps( _mm_andnot_ps( __m128c( c ), a ), _mm_and_ps( __m128c( c ), b ) )
#define _mm_sel_si128( a, b, c )			_mm_or_si128( _mm_andnot_si128( __m128c( c ), a ), _mm_and_si128( __m128c( c ), b ) )
#define _mm_sld_ps( x, y, imm )				__m128c( _mm_or_si128( _mm_srli_si128( __m128c( x ), imm ), _mm_slli_si128( __m128c( y ), 16 - imm ) ) )
#define _mm_sld_si128( x, y, imm )			_mm_or_si128( _mm_srli_si128( x, imm ), _mm_slli_si128( y, 16 - imm ) )


void quat_to_mat(float *mat, __m128 q0) 
{
	const __m128 vector_float_first_sign_bit		= __m128c( _mm_set_epi32( 0x00000000, 0x00000000, 0x00000000, 0x80000000 ) );
	const __m128 vector_float_last_three_sign_bits	= __m128c( _mm_set_epi32( 0x80000000, 0x80000000, 0x80000000, 0x00000000 ) );
	const __m128 vector_float_first_pos_half		= {   0.5f,   0.0f,   0.0f,   0.0f };	// +.5 0 0 0
	const __m128 vector_float_first_neg_half		= {  -0.5f,   0.0f,   0.0f,   0.0f };	// -.5 0 0 0
	const __m128 vector_float_quat2mat_mad1			= {  -1.0f,  -1.0f,  +1.0f,  -1.0f };	//  - - + -
	const __m128 vector_float_quat2mat_mad2			= {  -1.0f,  +1.0f,  -1.0f,  -1.0f };	//  - + - -
	const __m128 vector_float_quat2mat_mad3			= {  +1.0f,  -1.0f,  -1.0f,  +1.0f };	//  + - - +

	{
		//__m128 q0 = _mm_load_ps( &quat[0] );
		__m128 t0 = _mm_set_ss( 0.0f );

		__m128 d0 = _mm_add_ps( q0, q0 );

		__m128 sa0 = _mm_perm_ps( q0, _MM_SHUFFLE( 1, 0, 0, 1 ) );							//   y,   x,   x,   y
		__m128 sb0 = _mm_perm_ps( d0, _MM_SHUFFLE( 2, 2, 1, 1 ) );							//  y2,  y2,  z2,  z2
		__m128 sc0 = _mm_perm_ps( q0, _MM_SHUFFLE( 3, 3, 3, 2 ) );							//   z,   w,   w,   w
		__m128 sd0 = _mm_perm_ps( d0, _MM_SHUFFLE( 0, 1, 2, 2 ) );							//  z2,  z2,  y2,  x2

		sa0 = _mm_xor_ps( sa0, vector_float_first_sign_bit );

		sc0 = _mm_xor_ps( sc0, vector_float_last_three_sign_bits );							// flip stupid inverse quaternions

		__m128 ma0 = _mm_add_ps( _mm_mul_ps( sa0, sb0 ), vector_float_first_pos_half );		//  .5 - yy2,  xy2,  xz2,  yz2		//  .5 0 0 0
		__m128 mb0 = _mm_add_ps( _mm_mul_ps( sc0, sd0 ), vector_float_first_neg_half );		// -.5 + zz2,  wz2,  wy2,  wx2		// -.5 0 0 0
		__m128 mc0 = _mm_sub_ps( vector_float_first_pos_half, _mm_mul_ps( q0, d0 ) );		//  .5 - xx2, -yy2, -zz2, -ww2		//  .5 0 0 0

		__m128 mf0 = _mm_shuffle_ps( ma0, mc0, _MM_SHUFFLE( 0, 0, 1, 1 ) );					//       xy2,  xy2, .5 - xx2, .5 - xx2	// 01, 01, 10, 10
		__m128 md0 = _mm_shuffle_ps( mf0, ma0, _MM_SHUFFLE( 3, 2, 0, 2 ) );					//  .5 - xx2,  xy2,  xz2,  yz2			// 10, 01, 02, 03
		__m128 me0 = _mm_shuffle_ps( ma0, mb0, _MM_SHUFFLE( 3, 2, 1, 0 ) );					//  .5 - yy2,  xy2,  wy2,  wx2			// 00, 01, 12, 13

		__m128 ra0 = _mm_add_ps( _mm_mul_ps( mb0, vector_float_quat2mat_mad1 ), ma0 );		// 1 - yy2 - zz2, xy2 - wz2, xz2 + wy2,					// - - + -
		__m128 rb0 = _mm_add_ps( _mm_mul_ps( mb0, vector_float_quat2mat_mad2 ), md0 );		// 1 - xx2 - zz2, xy2 + wz2,          , yz2 - wx2		// - + - -
		__m128 rc0 = _mm_add_ps( _mm_mul_ps( me0, vector_float_quat2mat_mad3 ), md0 );		// 1 - xx2 - yy2,          , xz2 - wy2, yz2 + wx2		// + - - +


		__m128 ta0 = _mm_shuffle_ps( ra0, t0, _MM_SHUFFLE( 0, 0, 2, 2 ) );
		__m128 tb0 = _mm_shuffle_ps( rb0, t0, _MM_SHUFFLE( 1, 1, 3, 3 ) );
		__m128 tc0 = _mm_shuffle_ps( rc0, t0, _MM_SHUFFLE( 2, 2, 0, 0 ) );

		ra0 = _mm_shuffle_ps( ra0, ta0, _MM_SHUFFLE( 2, 0, 1, 0 ) );						// 00 01 02 10
		rb0 = _mm_shuffle_ps( rb0, tb0, _MM_SHUFFLE( 2, 0, 0, 1 ) );						// 01 00 03 11
		rc0 = _mm_shuffle_ps( rc0, tc0, _MM_SHUFFLE( 2, 0, 3, 2 ) );	

		_mm_store_ps( mat, ra0 );
		_mm_store_ps( mat+4, rb0 );
		_mm_store_ps( mat+8, rc0 );
	}
}
