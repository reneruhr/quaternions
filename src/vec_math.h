#pragma once

using f32 = float;
#include <cstdio>

struct vec2_f32
{
	f32 x{}, y{};
};

struct vec3_f32
{
	f32 x{}, y{}, z{};
	vec3_f32() = default;
	constexpr vec3_f32(f32 x, f32 y, f32 z) : x(x), y(y), z(z) {}
	constexpr vec3_f32(f32 a) { x = a; y = a; z = a; }
};

struct vec4_f32
{
	f32 x{}, y{}, z{}, w{};
	
	bool operator==(const vec4_f32& other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
	vec4_f32() = default;

	constexpr vec4_f32(f32 a)
	{
		x = a; y = a; z = a; w = a;
	}
	constexpr vec4_f32(vec3_f32 t)
	{
		x = t.x; y = t.y; z = t.z; w = 1.f;
	}

	constexpr vec4_f32(f32 a, vec3_f32 t)
	{
		x = a; y = t.x; z = t.y; w = t.z;
	}

	constexpr vec4_f32(f32 x, f32 y, f32 z, f32 w) : x(x), y(y), z(z), w(w) {}

	constexpr vec3_f32 yzw() { return { y,z,w }; }
};

inline bool is_zero(vec2_f32 v)
{
	return v.x == 0 && v.y ==0;
}

inline bool is_not_zero(vec2_f32 v)
{
	return v.x != 0 || v.y !=0;
}

inline bool is_zero(vec3_f32 v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

inline bool is_not_zero(vec3_f32 v)
{
	return v.x != 0 || v.y != 0 || v.z != 0;
}

inline f32 operator,(vec3_f32 v, vec3_f32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

inline f32 dot(vec3_f32 v, vec3_f32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

inline vec3_f32 cross(vec3_f32 a, vec3_f32 b)
{
	return {a.y*b.z-a.z*b.y, b.x*a.z-b.z*a.x, a.x*b.y-a.y*b.x};
}

inline f32 operator,(vec2_f32 v, vec2_f32 w)
{
	return v.x*w.x + v.y*w.y;
}

inline f32 dot(vec4_f32 v, vec4_f32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z + v.w*w.w;
}

inline vec4_f32 operator+(vec4_f32 v, vec4_f32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z, v.w+w.w};
}

inline vec3_f32 operator+(vec3_f32 v, vec3_f32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z};
}

inline vec2_f32 operator+(vec2_f32 v, vec2_f32 w)
{
	return {v.x+w.x, v.y+w.y};
}

inline vec4_f32 operator-(vec4_f32 v, vec4_f32 w)
{
	return {v.x-w.x, v.y-w.y, v.z-w.z, v.w-w.w};
}

inline vec3_f32 operator-(vec3_f32 v, vec3_f32 w)
{
	return {v.x-w.x, v.y-w.y, v.z-w.z};
}

inline vec2_f32 operator-(vec2_f32 v, vec2_f32 w)
{
	return {v.x-w.x, v.y-w.y};
}

inline vec3_f32 operator*(f32 a, vec3_f32 w)
{
	return {a*w.x, a*w.y, a*w.z};
}

inline vec4_f32 operator*(f32 a, vec4_f32 w)
{
	return {a*w.x, a*w.y, a*w.z, a*w.w};
}

inline vec2_f32 operator*(f32 a, vec2_f32 w)
{
	return {a*w.x, a*w.y};
}

inline void print(vec3_f32 v)
{
	printf("(%f, %f, %f)\n", v.x, v.y, v.z);
}

inline void print(vec4_f32 v)
{
	printf("(%f, %f, %f, %f)\n", v.x, v.y, v.z,v.w);
}







using u32 = uint32_t;

struct vec2_u32
{
	u32 x, y;
};

struct vec3_u32
{
	u32 x, y, z;
	vec3_u32(u32 a) { x = a; y = a; z = a; };
	vec3_u32(u32 x, u32 y, u32 z) : x(x), y(y), z(z) {}
};

struct vec4_u32
{
	u32 x, y, z, w;

	vec4_u32(u32 a) { x = a; y = a; z = a; w = a; }
	vec4_u32(u32 x, u32 y, u32 z, u32 w) : x(x), y(y), z(z), w(w) {}

	vec3_u32 xyz()
	{
		return { x, y, z };
	}
};

inline u32 dot(vec4_u32 v, vec4_u32 w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z + v.w*w.w;
}

inline vec4_u32 operator+(vec4_u32 v, vec4_u32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z, v.w+w.w};
}

inline vec4_u32 operator-(vec4_u32 v, vec4_u32 w)
{
	return {v.x-w.x, v.y-w.y, v.z-w.z, v.w-w.w};
}

inline vec3_u32 operator+(vec3_u32 v, vec3_u32 w)
{
	return {v.x+w.x, v.y+w.y, v.z+w.z};
}

inline vec3_u32 operator-(vec3_u32 v, vec3_u32 w)
{
	return {v.x-w.x, v.y-w.y, v.z-w.z};
}

inline vec2_u32 operator+(vec2_u32 v, vec2_u32 w)
{
	return {v.x+w.x, v.y+w.y};
}

inline vec2_u32 operator-(vec2_u32 v, vec2_u32 w)
{
	return {v.x-w.x, v.y-w.y};
}

inline vec3_u32 operator*(u32 a, vec3_u32 w)
{
	return {a*w.x, a*w.y, a*w.z};
}

inline vec4_u32 operator*(u32 a, vec4_u32 w)
{
	return {a*w.x, a*w.y, a*w.z, a*w.w};
}

inline vec4_f32 to_float(vec4_u32 w)
{
	return vec4_f32((f32)w.x, (f32)w.y, (f32)w.z, (f32)w.w);
}
inline vec3_f32 to_float(vec3_u32 w)
{
	return vec3_f32((f32)w.x, (f32)w.y, (f32)w.z);
}

inline vec3_u32 operator+(u32 a, vec3_u32 v)
{
	return vec3_u32(a) + v;
}

inline vec3_u32 operator-(u32 a, vec3_u32 v)
{
	return vec3_u32(a) - v;
}

inline vec4_u32& operator^=(vec4_u32& a, const vec4_u32& b) 
{ 
	a.x ^= b.x;
	a.y ^= b.y;
	a.z ^= b.z;
	a.w ^= b.w;
	return a;
}

inline vec3_u32& operator^=(vec3_u32& a, const vec3_u32& b) 
{ 
	a.x ^= b.x;
	a.y ^= b.y;
	a.z ^= b.z;
	return a;
}

inline vec3_u32 operator<<(const vec3_u32& a, u32 b) 
{ 
	return {a.x << b, a.y << b, a.z << b};
}

inline vec4_u32 operator<<(const vec4_u32& a, u32 b) 
{ 
	return {a.x << b, a.y << b, a.z << b, a.w << b};
}

inline vec3_u32 operator>>(const vec3_u32& a, u32 b) 
{ 
	return {a.x >> b, a.y >> b, a.z >> b};
}

inline vec4_u32 operator>>(const vec4_u32& a, u32 b) 
{ 
	return {a.x >> b, a.y >> b, a.z >> b, a.w >> b};
}

inline vec4_f32 sin(vec4_f32 v)
{
	using std::sin;
	return { sin(v.x), sin(v.y), sin(v.z), sin(v.w) };
}
inline vec4_f32 cos(vec4_f32 v)
{
	using std::cos;
	return { cos(v.x), cos(v.y), cos(v.z), cos(v.w) };
}

inline vec4_f32 max(vec4_f32 v, vec4_f32 w)
{
	using std::max;
	return { max(v.x, w.x), max(v.y, w.y), max(v.z, w.z), max(v.w, w.w) };
}

inline vec4_f32 normalize(vec4_f32 q)
{
return 1.f / std::sqrt(dot(q,q)) * q;
}


struct mat4
{
	vec4_f32 columns[4];
	mat4(vec4_f32 a, vec4_f32 b, vec4_f32 c, vec4_f32 d)
	{
		columns[0] = a;
		columns[1] = b;
		columns[2] = c;
		columns[3] = d;
	}

};

inline vec4_f32 operator*(vec4_f32 v, mat4 m)
{
	return { dot(v,m.columns[0]), dot(v,m.columns[1]), dot(v,m.columns[2]), dot(v,m.columns[3]) };
}

inline vec4_f32 operator+(f32 a, vec4_f32 v)
{
	return vec4_f32(a) + v;
}

inline vec4_f32 operator-(f32 a, vec4_f32 v)
{
	return vec4_f32(a) - v;
}



inline vec3_u32 operator<<(vec3_u32& a, u32 b) 
{ 
	return vec3_u32(a.x << b, a.y << b, a.z << b);
}

inline vec4_u32 operator<<(vec4_u32& a, u32 b) 
{ 
	return vec4_u32(a.x << b, a.y << b, a.z << b, a.w << b);
}

inline vec3_u32 operator>>(vec3_u32& a, u32 b) 
{ 
	return vec3_u32(a.x >> b, a.y >> b, a.z >> b);
}

inline vec4_u32 operator>>(vec4_u32& a, u32 b) 
{ 
	return vec4_u32(a.x >> b, a.y >> b, a.z >> b, a.w >> b);
}


inline vec4_u32 lessThanEqual(const vec4_f32& a, vec4_f32& b) 
{ 
	return  { a.x <= b.x, a.y <= b.y ,a.z <= b.z ,a.w <= b.w };
}






// Column major
struct mat4_f32
{
	f32 m[16];
	
	vec4_f32 col(u32 i)
	{
		vec4_f32 v( m[i*4], m[i*4+1], m[i*4+2], m[i*4+3]);
		return v;
	}

	vec4_f32 row(u32 i)
	{
		vec4_f32 v( m[i], m[i + 4], m[i + 8], m[i + 12] );
		return v;
	}

	mat4_f32()
	{
		memset(m, 0, sizeof(m));
	}

	mat4_f32(f32 m00, f32 m01, f32 m02, f32 m03,
			 f32 m10, f32 m11, f32 m12, f32 m13,
			 f32 m20, f32 m21, f32 m22, f32 m23,
			 f32 m30, f32 m31, f32 m32, f32 m33)
	{
		m[0] = m00; m[0+4] = m01; m[0+8] = m02; m[0+12] = m03;
		m[1] = m10; m[1+4] = m11; m[1+8] = m12; m[1+12] = m13;
		m[2] = m20; m[2+4] = m21; m[2+8] = m22; m[2+12] = m23;
		m[3] = m30; m[3+4] = m31; m[3+8] = m32; m[3+12] = m33;
	}

	mat4_f32(f32 m00, f32 m11, f32 m22, f32 m33)
	{
		memset(m, 0, sizeof(m));
		m[0]  = m00;
		m[5]  = m11;
		m[10] = m22;
		m[15] = m33;
	}

	mat4_f32(f32 a)
	{
		memset(m, 0, sizeof(m));
		m[0]  = m[5]  = m[10] = m[15] = a;
	}
};

inline void print(mat4_f32& mat)
{
	print(mat.row(0));
	print(mat.row(1));
	print(mat.row(2));
	print(mat.row(3));
}

inline mat4_f32 transpose(mat4_f32 mat)
{
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < i; j++)
			std::swap(mat.m[4 * i + j], mat.m[4 * j + i]);
	return mat;
}

inline mat4_f32 mul(mat4_f32 a, mat4_f32 b)
{
	mat4_f32 mat;
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 4; j++)
			mat.m[4 * i + j] = dot(a.row(j), b.col(i));
	return mat;
}

inline mat4_f32 translation(vec3_f32 t)
{
	mat4_f32 mat(1.f);
	mat.m[12+0] = t.x;
	mat.m[12+1] = t.y;
	mat.m[12+2] = t.z;
	return mat;
}

inline mat4_f32 scale(vec3_f32 s)
{
	return { s.x, s.y, s.z, 1. };
}

inline mat4_f32 scale(f32 s)
{
	return { s, s, s, 1. };
}

inline mat4_f32 perspective(f32 fovy, f32 aspect, f32 near, f32 far)
{
	fovy *= 3.141592f / 180.f;
	f32 f = 1.f / tan(fovy / 2);
	f32 A = (near+far)/(near-far);
	f32 B = 2*far*near/(near-far);

	f32 perspective[16] =
	{
		f/aspect, 0, 0, 0,
		0		, f, 0, 0,
		0		, 0, A, -1, 
		0		, 0, B, 0
	};

	mat4_f32 m;
	memcpy(m.m, perspective, sizeof(perspective));
	return m;
}
