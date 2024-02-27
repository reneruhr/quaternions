struct experiment_info
{
	char* label;
	u64 count;
	u64 iterations;
	u64 size;
};

struct experiment_result
{
	u32 n_samples;
	f64 *times;
};

struct experiment_records
{
	u32 n_experiments;
	experiment_info* info;
	experiment_result* results;
};

struct statistics
{
	f64 mean{};
	f64 msqrt{};
	f64 min{};
	f64 max{};
};

statistics mean_statistics(f64* a, u32 n)
{
	f64 sum{};
	f64 sum2{};
	f64 min{a[0]};
	f64 max{a[0]};
	for (u32 i{}; i < n; i++)
	{
		min = std::min(min, a[i]);
		max = std::max(max, a[i]);
		sum  += a[i];
		sum2 += a[i]*a[i];
	}
	f64 mean = sum / n;
	f64 msqrt = sum2 / n - mean * mean;

	return { mean , std::sqrt(msqrt) , min, max};
}

template <class info>
statistics mean_statistics(f64* a, u32 n, f64(*f)(f64, info), info p)
{
	f64 sum{};
	f64 sum2{};
	f64 min{f(a[0],p)};
	f64 max{f(a[0],p)};
	for (u32 i{}; i < n; i++)
	{
		f64 b = f(a[i], p);
		min = std::min(min, b);
		max = std::max(max, b);
		sum  += b;
		sum2 += b*b;
	}
	f64 mean = sum / n;
	f64 msqrt = sum2 / n - mean * mean;

	return { mean , std::sqrt(msqrt) , min, max};
}


void print_statistics(const experiment_records& records)
{
	for (u32 u{}; u < records.n_experiments; u++)
	{
		auto& results = records.results[u];
		auto& info    = records.info[u];

		auto per_quat = [](f64 ns, u64 n) { return ns / (f64)n;  };
		auto [mean, msqrt, min, max] = 
			mean_statistics<u64>(results.times, results.n_samples, (f64(*)(f64, u64))per_quat, info.count*info.iterations);
		auto to_bw = [](f64 ns, u64 bytes) { return (f64)bytes / ns;  };
		auto [mean_bw, msqrt_bw, min_bw, max_bw] = mean_statistics<u64>(results.times, results.n_samples, (f64(*)(f64, u64))to_bw, (u64)info.count * info.iterations * info.size);

		printf("\n\n%s", info.label);
		printf("\t\t\tCount %u", info.count);
		printf("\tMean ns/quat : %f  (%f)\n", mean, msqrt);
		printf("\tMin/Max ns/quat : [%f  %f]\n", min, max);
		printf("\t%s: %f (%f)\n",info.iterations==1 ? "Bandwidth GB/s " : "GFlops ", mean_bw, msqrt_bw);
		printf("\tMin/Max: [%f  %f]\n", min_bw, max_bw);
	}
}