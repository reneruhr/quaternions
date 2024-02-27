#pragma once

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <climits>

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

#include <numbers>
constexpr f32 pi_f = std::numbers::pi_v<f32>;
constexpr f32 tau = 2*std::numbers::pi_v<f32>;

constexpr f32 s2 = std::numbers::sqrt2_v<f32>; 
constexpr f32 s2inv = s2/2;

constexpr f32 tau_uint_max_inv = std::numbers::pi_v<f32> * 0x1p-31;

// Mike Day: Generalizing the Fast Reciprocal Square Root Algorithm
inline f32 fast_inv_sqrt(f32 x)	 
{
	u32 X = *( u32 *) & x ;
	u32 Y = 0x5F11107D - ( X >> 1) ;
	f32 y = *( f32 *) &Y ;
	f32 z = x * y * y ;
	return  y * (2.2825186f + z *( z -2.253305f ) );
}

#include <chrono>
using stdclock = std::chrono::steady_clock;

struct chrono_timer_scoped
{
	std::chrono::time_point<stdclock> start;
	const char* label;
	u64 *store;

	chrono_timer_scoped(const char* label, u64 *store = nullptr) 
		: start(stdclock::now()), label(label), store(store) {}


	~chrono_timer_scoped()
	{
		auto stop = stdclock::now();
    	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		if(store)
			*store = duration.count();
		printf("%s: %llu ms\n", label, duration.count());
	}
};

struct chrono_timer
{
	std::chrono::time_point<stdclock> start_time;
	const char* label;
	u64 *store;

	chrono_timer(const char* label, u64 *store = nullptr) : label(label), store(store) {}

	void start()
	{
		this->start_time = stdclock::now();
	}

	u64 stop()
	{
		auto stop_time = stdclock::now();
    	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time);
		if(store)
			*store = duration.count();
		return duration.count();
	}
};


#ifdef __MACH__
#include <mach/mach_time.h>
inline u64 cpu_timer()
{
	return mach_absolute_time();
}

inline u64 ticks_to_ns(u64 time)
{
	static mach_timebase_info_data_t timebase_info;
    if (timebase_info.denom == 0) 
        mach_timebase_info(&timebase_info);
    return time* timebase_info .numer / timebase_info.denom;
}
#endif

#ifdef __linux__
#include <x86intrin.h>
inline u64 cpu_timer()
{
	return __rdtsc();
}
#endif

#ifdef _WIN32
#include <intrin.h>
inline u64 cpu_timer()
{
	return __rdtsc();
}
#endif

constexpr u64 wait_time{100};

// ticks per seconds
inline u64 cpu_frequency()
{
	static u64 freq{0};
	if(!freq){
		u64 s = cpu_timer();
		auto start = stdclock::now();
		auto end   = start + std::chrono::milliseconds(wait_time);
		while(stdclock::now() < end) 
		{
		};
	 	u64 e = cpu_timer();	
		freq = (e-s)* (u64)(1000. / wait_time); 
	}	
	return freq;
}

inline void print_ticks(u64 ticks)
{
	printf("\tTime taken: %llu cycles in %f ms\n", ticks, (f64)ticks/cpu_frequency()*1.e-3);
}
