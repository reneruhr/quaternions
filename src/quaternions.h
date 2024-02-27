#pragma once
#include "common.h"
#include "vec_math.h"

struct fquat32
{
	f32 r{1},x{0},y{0},z{0};
/*
	bool operator==(const fquat32& other) const {
        return x == other.x && y == other.y && z == other.z && r == other.r;
    }
*/
	bool operator==(const fquat32& other) const 
	{
		f32 eps = 1e-6f;
        return  std::fabs(x - other.x) < eps &&
				std::fabs(y - other.y) < eps &&
				std::fabs(z - other.z) < eps &&
				std::fabs(r - other.r) < eps ;
    }
	bool operator<(const fquat32& other) const 
	{
		f32 eps = 1e-6f;
		if(x < other.x - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && y < other.y - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && std::fabs(y-other.y)<eps && z < other.z - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && std::fabs(z-other.z)<eps && std::fabs(y-other.y)<eps  
										   && r < other.r - eps)
			return true;
		return false;
    }
};

inline vec4_f32 to_vec4(fquat32 q)
{
	return { q.r, q.x, q.y, q.z };
}

inline fquat32 to_quat(vec4_f32 q)
{
	return { q.x, q.y, q.z, q.w };
}

struct quat_s32
{
	int r{1},x{0},y{0},z{0};
};

using quat = fquat32;
using quatz = quat_s32;


inline quat im(quat a)
{
	return {0, a.x, a.y, a.z};
}

inline quat operator~(quat a)
{
	return {a.r, -a.x, -a.y, -a.z};
}

inline quat operator*(f32 t, quat a)
{
	return {t*a.r, t*a.x, t*a.y, t*a.z};
}

inline quat operator+(quat a, quat b)
{
	return {a.r+b.r, a.x+b.x, a.y+b.y, a.z+b.z};
}

inline quat operator-(quat a, quat b)
{
	return {a.r-b.r, a.x-b.x, a.y-b.y, a.z-b.z};
}

inline quat operator^(quat a, quat b)
{
	return {0, a.y*b.z-a.z*b.y, b.x*a.z-b.z*a.x, a.x*b.y-a.y*b.x};
} // 6 mults 3 adds

inline f32 operator,(quat a, quat b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
} // 3 mults 3 adds

inline quat operator*(quat a, quat b)
{
	quat q{a.r*b.r - (a,b), 0,0,0};
	return q + a.r*im(b)+b.r*im(a)+ (a^b);
} // 12 mults 14 adds 

inline quat to_quatf(quatz q, f32 norm)
{
	return { q.r*norm, q.x*norm, q.y*norm, q.z*norm};
}

inline quatz im(quatz a)
{
	return {0, a.x, a.y, a.z};
}

inline quatz operator~(quatz a)
{
	return {a.r, -a.x, -a.y, -a.z};
}

inline quatz operator*(int t, quatz a)
{
	return {t*a.r, t*a.x, t*a.y, t*a.z};
}

inline quatz operator+(quatz a, quatz b)
{
	return {a.r+b.r, a.x+b.x, a.y+b.y, a.z+b.z};
}

inline quatz operator^(quatz a, quatz b)
{
	return {0, a.y*b.z-a.z*b.y, b.x*a.z-b.z*a.x, a.x*b.y-a.y*b.x};
}

inline int operator,(quatz a, quatz b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline quatz operator*(quatz a, quatz b)
{
	quatz q{a.r*b.r - (a,b), 0,0,0};
	return q + a.r*im(b)+b.r*im(a)+ (a^b);
} 

inline quat make_quat(quatz a, u32 norm)
{
	f32 n = fast_inv_sqrt((f32)norm);
	return { a.r * n, a.x * n, a.y * n, a.z * n };
}

inline f32 norm2(quat q)
{
	return q.r*q.r + q.x*q.x + q.y*q.y + q.z*q.z;
}

inline f32 dot(quat v, quat w)
{
	return v.r*w.r + v.x*w.x + v.y*w.y + v.z*w.z;
}

inline quat normalize(quat q)
{
	return (1.f / std::sqrt(norm2(q))) * q;
}

inline quat normalize_fast(quat q)
{
	return fast_inv_sqrt(norm2(q)) * q;
}

inline void print(quat q)
{
	printf("(%f, %f, %f, %f)\n", q.r,q.x,q.y,q.z);
}

inline void print(quatz q)
{
	printf("(%i, %i, %i, %i)\n", q.r,q.x,q.y,q.z);
}

inline quat angle_axis(f32 angle, f32 x, f32 y, f32 z)
{
	angle /= 2;
	f32 s = std::sin(angle);
	f32 c = std::cos(angle);
	return { c, x * s, y * s, z * s };
}

struct mat3f32
{
	f32 data[9]{1,0,0, 0,1,0, 0,0,1};

	vec3_f32 operator[](u32 i) const
	{
		return { data[3 * i + 0], data[3 * i + 1], data[3 * i + 2] };
	}
};

inline mat3f32 rotation(quat q) 
{
	f32 x2 = -2*q.x*q.x;
	f32 y2 = -2*q.y*q.y;
	f32 z2 = -2*q.z*q.z;

	f32 xy = 2*q.x*q.y;
	f32 zy = 2*q.z*q.y;
	f32 xz = 2*q.x*q.z;

	f32 rx = 2*q.r*q.x;
	f32 ry = 2*q.r*q.y;
	f32 rz = 2*q.r*q.z;
	mat3f32 k{ 1+y2+z2, xy-rz, xz+ry,
			   xy+rz, 1+x2+z2, zy-rx,
			   xz-ry, zy+rx, 1+x2+y2 };

	return k;
} // 9 mults 12 adds 9 shifts 

inline void quat_to_vec3_3(quat q, vec3_f32* v) // column vectors
{
	f32 x2 = -2*q.x*q.x;
	f32 y2 = -2*q.y*q.y;
	f32 z2 = -2*q.z*q.z;

	f32 xy = 2*q.x*q.y;
	f32 zy = 2*q.z*q.y;
	f32 xz = 2*q.x*q.z;

	f32 rx = 2*q.r*q.x;
	f32 ry = 2*q.r*q.y;
	f32 rz = 2*q.r*q.z;
	v[0] = {1+y2+z2, xy+rz, xz-ry};
	v[1] = {xy-rz, 1+x2+z2, zy+rx};
	v[2] = {xz+ry, zy-rx, 1+x2+y2};
} 

inline void quat_to_vec4_3(quat q, vec4_f32* v) // column vectors
{
	f32 x2 = -2*q.x*q.x;
	f32 y2 = -2*q.y*q.y;
	f32 z2 = -2*q.z*q.z;

	f32 xy = 2*q.x*q.y;
	f32 zy = 2*q.z*q.y;
	f32 xz = 2*q.x*q.z;

	f32 rx = 2*q.r*q.x;
	f32 ry = 2*q.r*q.y;
	f32 rz = 2*q.r*q.z;
	v[0] = {1+y2+z2, xy+rz, xz-ry, 0};
	v[1] = {xy-rz, 1+x2+z2, zy+rx, 0};
	v[2] = {xz+ry, zy-rx, 1+x2+y2, 0};
} 

inline vec4_f32 quat_to_vec4(quat q) // column vectors
{
	f32 x2 = -2*q.x*q.x;
	f32 y2 = -2*q.y*q.y;
	f32 z2 = -2*q.z*q.z;

	f32 xy = 2*q.x*q.y;
	f32 zy = 2*q.z*q.y;
	f32 xz = 2*q.x*q.z;

	f32 rx = 2*q.r*q.x;
	f32 ry = 2*q.r*q.y;
	f32 rz = 2*q.r*q.z;
	return { 1 + y2 + z2, xy + rz, xz - ry, 1.f };
} 

inline void quat_to_vec3_3i(quatz q, vec3_f32* v, f32 norm) 
{
	int x2 = -2*q.x*q.x;
	int y2 = -2*q.y*q.y;
	int z2 = -2*q.z*q.z;

	int xy = 2*q.x*q.y;
	int zy = 2*q.z*q.y;
	int xz = 2*q.x*q.z;

	int rx = 2*q.r*q.x;
	int ry = 2*q.r*q.y;
	int rz = 2*q.r*q.z;
	v[0] = {1.f+(y2+z2)*norm, (xy+rz)*norm, (xz-ry)*norm};
	v[1] = {(xy-rz)*norm, 1.f+(x2+z2)*norm, (zy+rx)*norm};
	v[2] = {(xz+ry)*norm, (zy-rx)*norm, 1.f+(x2+y2)*norm};
} 

inline vec3_f32 quat_to_vec3_i(quatz q, f32 norm) 
{
	int y2 = -2*q.y*q.y;
	int z2 = -2*q.z*q.z;

	int xy = 2*q.x*q.y;
	int xz = 2*q.x*q.z;

	int ry = 2*q.r*q.y;
	int rz = 2*q.r*q.z;

	return {1.f+(y2+z2)*norm, (xy+rz)*norm, (xz-ry)*norm};
} 

inline vec3_f32 quat_to_vec3_long(quatz q, f32 norm) 
{
	f32 y2 = -2.f*q.y*q.y;
	f32 z2 = -2.f*q.z*q.z;

	f32 xy = 2.f*q.x*q.y;
	f32 xz = 2.f*q.x*q.z;

	f32 ry = 2.f*q.r*q.y;
	f32 rz = 2.f*q.r*q.z;
	return {1.f+(y2+z2)*norm, (xy+rz)*norm, (xz-ry)*norm};
} 

inline void quat_to_vec3_3i_long(quatz q, vec3_f32* v, f32 norm)
{
	f32 x2 = -2.f*q.x*q.x;
	f32 y2 = -2.f*q.y*q.y;
	f32 z2 = -2.f*q.z*q.z;

	f32 xy = 2.f*q.x*q.y;
	f32 zy = 2.f*q.z*q.y;
	f32 xz = 2.f*q.x*q.z;

	f32 rx = 2.f*q.r*q.x;
	f32 ry = 2.f*q.r*q.y;
	f32 rz = 2.f*q.r*q.z;
	v[0] = {1.f+(y2+z2)*norm, (xy+rz)*norm, (xz-ry)*norm};
	v[1] = {(xy-rz)*norm, 1.f+(x2+z2)*norm, (zy+rx)*norm};
	v[2] = {(xz+ry)*norm, (zy-rx)*norm, 1.f+(x2+y2)*norm};
} 

inline void quat_to_vec4_3_transpose(quat q, vec4_f32* v) // row vectors
{
	f32 x2 = -2*q.x*q.x;
	f32 y2 = -2*q.y*q.y;
	f32 z2 = -2*q.z*q.z;

	f32 xy = 2*q.x*q.y;
	f32 zy = 2*q.z*q.y;
	f32 xz = 2*q.x*q.z;

	f32 rx = 2*q.r*q.x;
	f32 ry = 2*q.r*q.y;
	f32 rz = 2*q.r*q.z;
	v[0] = {1+y2+z2, xy-rz, xz+ry, 0};
	v[1] = {xy+rz, 1+x2+z2, zy-rx, 0};
	v[2] = {xz-ry, zy+rx, 1+x2+y2, 0};
} 

inline vec3_f32 make_vec3(quat q)
{
	f32 y2 = -2*q.z*q.z;
	f32 z2 = -2*q.r*q.r;

	f32 xy = 2*q.y*q.z;
	f32 xz = 2*q.y*q.r;

	f32 ry = 2*q.x*q.z;
	f32 rz = 2*q.x*q.r;
	 
	return {1+y2+z2,   xy-rz,   xz+ry};
} 

struct vec3_s32 { int x{}, y{}, z{}; };
struct mat3s32
{
	s32 data[9]{1,0,0, 0,1,0, 0,0,1};

	vec3_s32 operator[](u32 i) const
	{
		return { data[3 * i + 0], data[3 * i + 1], data[3 * i + 2] };
	}

	vec3_f32 normalized_col(u32 i, s32 norm2) const
	{
		f32 s = 1.f / norm2;
		return { s*data[3 * i + 0], s*data[3 * i + 1], s*data[3 * i + 2] };
	}
};

inline mat3s32 rotation_scale(quatz q, s32 norm2) 
{
	s32 x2 = -2*q.x*q.x;
	s32 y2 = -2*q.y*q.y;
	s32 z2 = -2*q.z*q.z;

	s32 xy = 2*q.x*q.y;
	s32 zy = 2*q.z*q.y;
	s32 xz = 2*q.x*q.z;

	s32 rx = 2*q.r*q.x;
	s32 ry = 2*q.r*q.y;
	s32 rz = 2*q.r*q.z;
	mat3s32 k{ norm2+y2+z2, xy-rz, xz+ry,
			   xy+rz, norm2+x2+z2, zy-rx,
			   xz-ry, zy+rx, norm2+x2+y2 };

	return k;
} 

struct quat_orbit
{
	u32   size;
	quat *T;
};

quat_orbit make_hecke_orbit(u32 n, const quat *T, u32 p);

inline constexpr u64 size_hecke_sphere(u32 p, u32 level)
{
	if(!level) return 1;
	u64 s = 1;
	while(--level)
		s*= p;
	return s*(u64)(p+1);
}

inline constexpr u64 size_hecke_tree(u32 p, u32 level)
{
	if(!level) return 1;
	u64 s = (u64)p+1;
	u64 t = s + 1;
	while(--level)
	{
		s*= p;
		t+= s;
	}
	return t;
}

//Assumption on quat table T: T[i]*T[(i+(p+1)/2)%(p+1)] = id
quat_orbit make_tree(u32 level, const quat *T, u32 p);
void make_tree(const quat *T, u32 p, quat* v, u32 n);

const f32 cto_float = s2 / 1023;
const f32 cto_u32= s2inv * 1023;

// Mark Zarb-Adami - Quaternion Compression: Smallest Three
inline quat decode(u32 w)
{
	u32 x =  (w >>  2 & 1023);
	u32 y =  (w >> 12 & 1023);
	u32 z =  (w >> 22 & 1023);

	f32 xf = static_cast<float>(x) * cto_float - s2inv;
	f32 yf = static_cast<float>(y) * cto_float - s2inv;
	f32 zf = static_cast<float>(z) * cto_float - s2inv;
	f32 xyz =  xf * xf + yf * yf + zf * zf;
	f32 rf = std::sqrt(1.f - xyz);

	quat q{};

	(&q.r)[w + 0 & 3] = rf;
	(&q.r)[w + 1 & 3] = xf;
	(&q.r)[w + 2 & 3] = yf;
	(&q.r)[w + 3 & 3] = zf;

	return q;
}


inline u32 encode(quat q)
{
	u32 i{};
	f32 s{std::abs(q.r)};
	f32 t;
	if ((t = std::abs(q.x)) > s)
	{
		i = 1;
		s = t;
	}
	if ((t = std::abs(q.y)) > s)
	{
		i = 2;
		s = t;
	}
	if ((t = std::abs(q.z)) > s)
	{
		i = 3;
		s = t;
	}
	if ((&q.r)[i] < 0.f) q = quat{0.f,0.f,0.f,0.f} - q;

	u32 x = static_cast<u32>(((&q.r)[i + 1 & 3] + s2inv) * cto_u32);
	u32 y = static_cast<u32>(((&q.r)[i + 2 & 3] + s2inv) * cto_u32);
	u32 z = static_cast<u32>(((&q.r)[i + 3 & 3] + s2inv) * cto_u32);

	u32 w = i << 0 | x << 2 | y << 12 | z << 22;

	return w;
}

inline mat4_f32 quat_to_mat4(quat q)
{
	f32 x2 = -2 * q.x * q.x;
	f32 y2 = -2 * q.y * q.y;
	f32 z2 = -2 * q.z * q.z;

	f32 xy = 2 * q.x * q.y;
	f32 zy = 2 * q.z * q.y;
	f32 xz = 2 * q.x * q.z;

	f32 rx = 2 * q.r * q.x;
	f32 ry = 2 * q.r * q.y;
	f32 rz = 2 * q.r * q.z;

	return 
	{	
		1 + y2 + z2,   	xy + rz,	xz - ry, 0,
			xy - rz,1 + x2 + z2,	zy + rx, 0,
			xz + ry,	zy - rx,1 + x2 + y2, 0,
				  0,		  0,		0,	 1
	};
}

const f32 s5 = 1.f/std::sqrt(5.f);
struct S_5_biased_reordered
{	// This table layout works for the Bernoulli, non-backtracking and biased walk from random_int
	const quat s[8]
	{			s5 * quat{1, 2, 0, 0},
				s5 * quat{1, 0, 2, 0},
				s5 * quat{1, 0, 0, 2},
				s5 * quat{1,-2, 0, 0},
				s5 * quat{1, 0,-2, 0},
				s5 * quat{1, 0, 0,-2},
				s5 * quat{1, 0, 2, 0},
				s5 * quat{1, 0, 0, 2},
	};
};

struct S_5ib
{
	const quatz s[8]
	{ quatz{1, 2,0,0},
				quatz{1,0, 2,0},
				quatz{1,0,0, 2},
				quatz{1,-2,0,0},
				quatz{1,0,-2,0},
				quatz{1,0,0,-2},
				quatz{1, 2,0,0},
				quatz{1,0, 2,0},
	};
};

const f32 s13 = 1.f/std::sqrt(13.f);
struct S_13
{
	const quat s[14] {
				s13*quat{1, 2, 2, 2},
				s13*quat{1, -2, 2, 2},
				s13*quat{1, 2, -2, 2},
				s13*quat{1, 2, 2, -2},
				s13*quat{1, -2, -2, 2},
				s13*quat{1, -2, 2, -2},
				s13*quat{1, 2, -2, -2},
				s13*quat{1, -2, -2, -2},
				s13*quat{3, 2, 0, 0},
				s13*quat{3, -2, 0, 0},
				s13*quat{3, 0, 2, 0},
				s13*quat{3, 0, -2, 0},
				s13*quat{3, 0, 0, 2},
				s13*quat{3, 0, 0, -2},
	}; 
};

const f32 s17 = 1.f/std::sqrt(17.f);
struct S_17_reordered
{
	const quat s[18] {
		s17*quat{1,0,0,4},
		s17*quat{1,0,4,0},
		s17*quat{1,4,0,0},
		s17*quat{3,0,2,2},
		s17*quat{3,0,-2,2},
		s17*quat{3,2,0,2},
		s17*quat{3,-2,0,2},
		s17*quat{3,2,2,0},
		s17*quat{3,-2,2,0},
		s17*quat{1,0,0,-4},
		s17*quat{1,0,-4,0},
		s17*quat{1,-4,0,0},
		s17*quat{3,0,-2,-2},
		s17*quat{3,0,2,-2},
		s17*quat{3,-2,0,-2},
		s17*quat{3,2,0,-2},
		s17*quat{3,-2,-2,0},
		s17*quat{3,2,-2,0}
	};
} ;


const f32 s29 = 1.f/std::sqrt(29.f);
struct S_29_reordered
{
	const quat s[30] {
		s29*quat{3,0,-2,-4},
		s29*quat{3,0,2,-4},
		s29*quat{3,0,-4,-2},
		s29*quat{3,0,4,-2},
		s29*quat{3,-2,0,-4},
		s29*quat{3,-2,0,4},
		s29*quat{3,-2,-4,0},
		s29*quat{3,2,-4,0},
		s29*quat{3,-4,0,-2},
		s29*quat{3,-4,0,2},
		s29*quat{3,4,-2,0},
		s29*quat{3,-4,-2,0},
		s29*quat{5,0,0,-2},
		s29*quat{5,0,-2,0},
		s29*quat{5,-2,0,0},
		s29*quat{3,0,2,4},
		s29*quat{3,0,-2,4},
		s29*quat{3,0,4,2},
		s29*quat{3,0,-4,2},
		s29*quat{3,2,0,4},
		s29*quat{3,2,0,-4},
		s29*quat{3,2,4,0},
		s29*quat{3,-2,4,0},
		s29*quat{3,4,0,2},
		s29*quat{3,4,0,-2},
		s29*quat{3,4,2,0},
		s29*quat{3,-4,2,0},
		s29*quat{5,0,0,2},
		s29*quat{5,0,2,0},
		s29*quat{5,2,0,0},
	};
} ;

struct S_29i
{
	const quatz s[30] {
		quatz{3,0,2,4},
		quatz{3,0,-2,4},
		quatz{3,0,2,-4},
		quatz{3,0,-2,-4},
		quatz{3,0,4,2},
		quatz{3,0,-4,2},
		quatz{3,0,4,-2},
		quatz{3,0,-4,-2},
		quatz{3,2,0,4},
		quatz{3,-2,0,4},
		quatz{3,2,0,-4},
		quatz{3,-2,0,-4},
		quatz{3,2,4,0},
		quatz{3,-2,4,0},
		quatz{3,2,-4,0},
		quatz{3,-2,-4,0},
		quatz{3,4,0,2},
		quatz{3,-4,0,2},
		quatz{3,4,0,-2},
		quatz{3,-4,0,-2},
		quatz{3,4,2,0},
		quatz{3,-4,2,0},
		quatz{3,4,-2,0},
		quatz{3,-4,-2,0},
		quatz{5,0,0,2},
		quatz{5,0,0,-2},
		quatz{5,0,2,0},
		quatz{5,0,-2,0},
		quatz{5,2,0,0},
		quatz{5,-2,0,0}
	};
};

struct S_29i_reordered
{
	const quatz s[30] {
		quatz{3,0,-2,-4},
		quatz{3,0,2,-4},
		quatz{3,0,-4,-2},
		quatz{3,0,4,-2},
		quatz{3,-2,0,-4},
		quatz{3,-2,0,4},
		quatz{3,-2,-4,0},
		quatz{3,2,-4,0},
		quatz{3,-4,0,-2},
		quatz{3,-4,0,2},
		quatz{3,4,-2,0},
		quatz{3,-4,-2,0},
		quatz{5,0,0,-2},
		quatz{5,0,-2,0},
		quatz{5,-2,0,0},
		quatz{3,0,2,4},
		quatz{3,0,-2,4},
		quatz{3,0,4,2},
		quatz{3,0,-4,2},
		quatz{3,2,0,4},
		quatz{3,2,0,-4},
		quatz{3,2,4,0},
		quatz{3,-2,4,0},
		quatz{3,4,0,2},
		quatz{3,4,0,-2},
		quatz{3,4,2,0},
		quatz{3,-4,2,0},
		quatz{5,0,0,2},
		quatz{5,0,2,0},
		quatz{5,2,0,0},
	};
};

struct S_5_biased
{
	const quat s[8] 
	{ 			s5*quat{1, 2,0,0},
				s5*quat{1,-2,0,0},
				s5*quat{1,0, 2,0},
				s5*quat{1,0,-2,0},
				s5*quat{1,0,0, 2},
				s5*quat{1,0,0,-2},
	 			s5*quat{1, 2,0,0},
				s5*quat{1,0,0, 2},
	}; 
};

struct S_13_biased
{
	const quat s[16] {
				s13*quat{1, 2, 2, 2},
				s13*quat{1, -2, 2, 2},
				s13*quat{1, 2, -2, 2},
				s13*quat{1, 2, 2, -2},
				s13*quat{1, -2, -2, 2},
				s13*quat{1, -2, 2, -2},
				s13*quat{1, 2, -2, -2},
				s13*quat{1, -2, -2, -2},
				s13*quat{3, 2, 0, 0},
				s13*quat{3, -2, 0, 0},
				s13*quat{3, 0, 2, 0},
				s13*quat{3, 0, -2, 0},
				s13*quat{3, 0, 0, 2},
				s13*quat{3, 0, 0, -2},
				s13*quat{1, 2, 2, 2},
				s13*quat{1, -2, 2, 2},
	}; 
};

struct S_17_biased
{
	const quat s[16] {
		s17*quat{1,0,0,4},
		s17*quat{1,0,0,-4},
		s17*quat{1,0,4,0},
		s17*quat{1,0,-4,0},
		s17*quat{1,4,0,0},
		s17*quat{1,-4,0,0},
		s17*quat{3,0,2,2},
		s17*quat{3,0,-2,2},
		s17*quat{3,0,2,-2},
		s17*quat{3,2,0,2},
		s17*quat{3,-2,0,2},
		s17*quat{3,2,0,-2},
		s17*quat{3,-2,0,-2},
		s17*quat{3,2,2,0},
		s17*quat{3,-2,2,0},
		s17*quat{3,2,-2,0},
	};
};

struct S_29_biased
{
	const quat s[32] {
		s29*quat{3,0,2,4},
		s29*quat{3,0,-2,4},
		s29*quat{3,0,2,-4},
		s29*quat{3,0,-2,-4},
		s29*quat{3,0,4,2},
		s29*quat{3,0,-4,2},
		s29*quat{3,0,4,-2},
		s29*quat{3,0,-4,-2},
		s29*quat{3,2,0,4},
		s29*quat{3,-2,0,4},
		s29*quat{3,2,0,-4},
		s29*quat{3,-2,0,-4},
		s29*quat{3,2,4,0},
		s29*quat{3,-2,4,0},
		s29*quat{3,2,-4,0},
		s29*quat{3,-2,-4,0},
		s29*quat{3,4,0,2},
		s29*quat{3,-4,0,2},
		s29*quat{3,4,0,-2},
		s29*quat{3,-4,0,-2},
		s29*quat{3,4,2,0},
		s29*quat{3,-4,2,0},
		s29*quat{3,4,-2,0},
		s29*quat{3,-4,-2,0},
		s29*quat{5,0,0,2},
		s29*quat{5,0,0,-2},
		s29*quat{5,0,2,0},
		s29*quat{5,0,-2,0},
		s29*quat{5,2,0,0},
		s29*quat{5,-2,0,0},
		s29*quat{3,0,2,4},
		s29*quat{3,0,-2,4}
	};
};

extern S_5_biased_reordered T5;
extern S_5ib T5i;
extern S_13 T13;
extern S_17_reordered T17;
extern S_29_reordered T29;
extern S_29i_reordered T29i;



// GPU implementation
inline quat quat_mul(quat a, quat b)
{
	return      quat{dot(a,~b),0,0,0} + 
				a.r*im(b) + b.r * im(a) + (im(a) ^ im(b)) ;
}

inline quat construct_ts(u32 u)
{
	u = u & 7;
	float r = s5- (u&1u)*s5*2;
	u >>= 1;
	float x = u==0 ? s5*2 : 0;
	float y = u==1 ? s5*2 : 0;
	float z = u>=2 ? s5*2 : 0;
	return {r, x , y, z};
}

inline quat walk_6(quat q, u32 a)
{
	u32 u = a & 7u;
	quat p = construct_ts(u);
	q = quat_mul(p, q);
	q = normalize(q);
	return q;
}
