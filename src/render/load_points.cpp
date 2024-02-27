struct points
{
	const char* label;
	u32 n_vertices;
	vec4_f32* pos;
	enum class typ {vec3, quat} typ{typ::vec3};
};
points read_point_set(const char* file_name, u32 max_points = 100'000);

struct set_of_points
{
	u32 n;
	points* pts;
};

const u32 n_max_point_sets = 100;
set_of_points read_from_dir(const char* dir_name, u32 max_point_sets = n_max_point_sets);
set_of_points convert_points(set_of_points);

points read_point_set(const char* file_name, u32 max_points)
{
	printf("Reading %s\n", file_name);
	FILE* file = fopen(file_name, "rb");
	assert(file && "Failed to read.");
	
	u32 n_label;
	fread(&n_label, sizeof(u32), 1, file);
	char* label = new char[n_label+1];
	fread(label, sizeof(char), n_label, file);
	label[n_label] = '\0';
	u32 n_points;
	fread(&n_points, sizeof(u32), 1, file);
	n_points = std::min(n_points, max_points);
	vec4_f32* pos = new vec4_f32[n_points];
	fread(pos, sizeof(vec4_f32), n_points, file);
	fclose(file);
	return { label, n_points, pos };
}

set_of_points read_from_dir(const char* dir_name, u32 max_point_sets)
{
	u32 n{};
	for (const auto& entry : std::filesystem::directory_iterator(dir_name))
	{
		if (entry.is_regular_file() && entry.path().filename().string().find("points_") == 0)
			n++;
	}
	
	n = std::min(n, max_point_sets);
	auto pts = new points[n];
	
	u32 m{};
	for (const auto& entry : std::filesystem::directory_iterator(dir_name))
	{
		if (m == n) break;
		if (entry.is_regular_file() && entry.path().filename().string().find("points_") == 0)
		{
			pts[m] = read_point_set(entry.path().string().c_str());
			if (entry.path().string().find("v4"))
				pts[m].typ = points::typ::quat;
			m++;
		}
	}

	return { n, pts };
}

set_of_points convert_points(set_of_points all_points)
{
	u32 m{};
	for (u32 u{}; u < all_points.n; u++)
	{
		auto pts = all_points.pts[u];
		if (pts.typ == points::typ::quat) m++;
	}

	set_of_points converted_pts{};
	if (m) converted_pts = { m, new points[m] };
	u32 n_converted_pts = m; 
	m = {};

	for (u32 u{}; u < all_points.n; u++)
	{
		auto pts = all_points.pts[u];
		if (pts.typ == points::typ::quat)
		{
			printf("Converting %s\n", pts.label);
			points image_sphere
			{
				.label = pts.label,
				.n_vertices = pts.n_vertices,
				.pos = new vec4_f32[pts.n_vertices],
				.typ = points::typ::vec3
			};

			for (u32 i{}; i < pts.n_vertices; i++)
			{
				auto v = pts.pos[i];
				image_sphere.pos[i] = quat_to_vec4(quat(v.x, v.y, v.z, v.w));
			}

			converted_pts.pts[m++] = image_sphere;
		}
	}
	return converted_pts;
}
