#pragma once
#include "common.h"
#include "random_int.h"
#include "quaternions.h"


template <class Sampler>
inline vec2_f32 disk_polar(Sampler& sampler)
{	
	const f32 z   =   0x1p-32f * sampler();
	const f32 phi =   tau_uint_max_inv * sampler();
	const f32 r = std::sqrt(z);
	return {r*std::cos(phi), r*std::sin(phi)};
}

template <class Sampler>
inline vec2_f32 disk_rejection(Sampler& sampler)
{	
	vec2_f32 p;
	do{ 
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler();
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}

template <class Sampler>
inline vec2_f32 disk_rejection2(Sampler& sampler)
{	
	vec2_f32 p;
	do{ 
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler(1);
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}

template <class Sampler>
inline vec3_f32 ball_rejection(Sampler& sampler)
{	
	vec3_f32 p;
	do{ 
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler();
		p.z = 1.f-0x1p-31f*sampler();
	  }     
	while  ( (p,p) >= 1.f);
  	
  	return p;
}

template <class Sampler>
inline vec3_f32 sphere_rejection(Sampler& sampler)
{	
	vec3_f32 p;
	f32 s;
	do{ 
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler();
		p.z = 1.f-0x1p-31f*sampler();
	  }     
	while  ( (( s=dot(p,p) ) >= 1.f ) || (s == 0));
	
	p = fast_inv_sqrt(s)*p;
  	return p;
}

template <class Sampler>
inline vec4_f32 ball4_rejection(Sampler& sampler)
{	
	vec4_f32 p;
	do{ 
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler();
		p.z = 1.f-0x1p-31f*sampler();
		p.w = 1.f-0x1p-31f*sampler();
	  }     
	while  ( dot(p,p) >= 1.f);
  	
  	return p;
}
template <class Sampler>
inline quat sphere4_rejection(Sampler& sampler)
{	
	quat p;
	f32 s;
	do{ 
		p.r = 1.f-0x1p-31f*sampler();
		p.x = 1.f-0x1p-31f*sampler();
		p.y = 1.f-0x1p-31f*sampler();
		p.z = 1.f-0x1p-31f*sampler();
	  }     
	while  ( (( s=norm2(p) ) >= 1.f ) || (s == 0));
	
	p = fast_inv_sqrt(s)*p;
  	return p;
}

inline vec3_f32 disk_to_sphere(vec2_f32 p)
{	
	const f32 s = (p,p);
	const f32 t = 2*std::sqrt(std::fmax(0.f,1.f-s));
	const f32 z = 1.f-2*s;
  	return {p.x*t,p.y*t, z};
}

// Marsaglia Choosing a point on the surface of a sphere 1972 Annals of Statistic
template <class Sampler>
inline vec3_f32 sphere_marsaglia(Sampler& sampler)
{	
	f32 x,y,s;
	do{ x = 1.f-0x1p-31f*sampler();
		y = 1.f-0x1p-31f*sampler();
	  }     while  ((s=x*x+y*y) >= 1.f);
  	
	const f32 t = std::sqrt(1.f-s);
	const f32 z = 1.f-2*s;
  	return {2*x*t,2*y*t, z};
}

template <class Sampler>
inline vec3_f32 sphere_marsaglia_polar(Sampler& sampler)
{	
	const f32 z   = 1.f - 0x1p-31f * sampler();
	const f32 phi = tau_uint_max_inv * sampler();
	const f32 r = std::sqrt(std::fmax(0.f,1.f-z*z));
	return {r*std::cos(phi), r*std::sin(phi), z};
}

// Marsaglia Choosing a point on the surface of a sphere 1972 Annals of Statistic
template <class Sampler>
inline quat sphere4_marsaglia(Sampler& sampler)
{	
	f32 x,y,z,w,s1,s2;
	do{ x = 1.f-0x1p-31f*sampler();
		y = 1.f-0x1p-31f*sampler();
	  }     while  ((s1=x*x+y*y) >= 1.f);
	do{ z = 1.f-0x1p-31f*sampler();
		w = 1.f-0x1p-31f*sampler();
	  }     while  ((s2=z*z+w*w) >= 1.f);
  	
	const f32 t = s2 == 0.f ? 0.f : std::sqrt((1.f-s1)/s2);
	return {x, y, z*t, w*t};
}

//Replacing rejection with trig functions turns it into Shoemake's subgroup algorithm.
template <class Sampler>
inline quat sphere4_marsaglia_polar(Sampler& sampler)
{	
	const f32 phi  =   tau_uint_max_inv * sampler();
	const f32 phi2 =   tau_uint_max_inv * sampler();
	const f32 z    =   0x1p-32f * sampler();
	const f32 r    =   std::sqrt(z);
	const f32 t    =   std::sqrt(1.f-z);
	return {r*std::cos(phi), r*std::sin(phi), t*std::cos(phi2), t*std::sin(phi2)};
}

//GPU implementations
template <class Sampler>
inline quat polar_1(Sampler& sampler)
{
	float phi = tau_uint_max_inv*sampler();
	float the = tau_uint_max_inv*sampler();

	float z = 0x1p-32f*sampler();
	float r = sqrt(z);
	float t = sqrt(1.f-z);
	return {r*cos(phi), r*sin(phi), t*cos(the), t*sin(the)};
}

template <class Sampler3d>
inline quat polar_2(Sampler3d& sampler)
{
	vec3_u32 x = sampler();

	vec3_f32 v = 0x1p-32f * vec3_f32(to_float(x));

	float phi = tau*v.x;
	float the = tau*v.y;
	float z = v.z;
	float r = sqrt(z);
	float t = sqrt(1.f-z);

	return {r*cos(phi), r*sin(phi), t*cos(the), t*sin(the)};
}

inline void spherical_fibonacci(vec3_f32* v, u32 n)
{
	const f32 dphi = std::numbers::pi_v<f32> * (3.f - std::sqrt(5.f));
	f32 phi = 0.f;
	const f32 dz = 2.f / n;
	f32 z = 1.f - dz / 2;

	for (u32 u = 0; u < n; u++)
	{
		f32 sin_phi = std::sin(phi);
		f32 cos_phi = std::cos(phi);
		f32 theta   = std::acos(z);
		f32 sin_theta = std::sin(theta);
		v[u] = { sin_phi * sin_theta, cos_phi * sin_theta, z };
		z -= dz;
		phi += dphi;
		phi = fmod(phi, 2 * std::numbers::pi_v<f32>);
	}
}

inline void spherical_fibonacci(quat* v, u32 n)
{
//	const f32 dphi = std::numbers::pi * (3.f - std::sqrt(5.f));
//	const f32 dtheta= std::numbers::pi * (3.f - std::sqrt(5.f));
// Alexa
	const float dphi = tau / 1.4142135623730951f;
	const float dtheta = tau / 1.533751168755204288118041f;
	f32 phi  = 0.f;
	f32 theta = 1.f;
	const f32 dz = 1.f / n;
	f32 z = dz/2;

	for (u32 u = 0; u < n; u++)
	{
		f32 r    =   std::sqrt(std::max(0.f,z));
		f32 t    =   std::sqrt(std::max(0.f,1.f-z));
		f32 sin_phi   = std::sin(phi);
		f32 cos_phi   = std::cos(phi);
		f32 sin_theta = std::sin(theta);
		f32 cos_theta = std::cos(theta);
		v[u] = { r*sin_phi , r*cos_phi, t*sin_theta, t*cos_theta };
		z += dz;
		phi += dphi;
		theta += dtheta;
		phi = fmod(phi, 2 * std::numbers::pi_v<f32>);
		theta = fmod(theta, 2 * std::numbers::pi_v<f32>);
	}
}
