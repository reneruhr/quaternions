

struct mesh
{
	u32 n_indices;
	u32 n_vertices;
	u32* idx;
	vec4_f32* pos;
	vec4_f32* col;
};

mesh sphere_parametric(u32 n_latitude, u32 n_longitude);

// n_latitude species the number of horizontal (latitude) lines on the sphere. 
// n_latitude = 1 gives one line around the equator
// n_longitude specifies number of vertical (longitude) lines crossing each horizontal one
// indexing scheme: 
// north_pole:		  0 (i=0)
// 1st latitude line: 0			  + (1,..., n_longitude) (i=1)
// 2st latitude line: n_longitude + (1,..., n_longitude) (i=2)
// ...
// ith latitude line: (i-1)*n_longitude + (1, ..., n_longitude) (i=i)
// ...
// last latitude line: (n_latitude-1)*n_longitude + (1,..., n_longitude)  (i=n_latitude)
// south_pole: n_latitude*n_longitude + 1 						  (i=n_latitude+1)
mesh sphere_parametric(u32 n_latitude, u32 n_longitude)
{
	u32 n_triangles = (n_latitude - 1) * n_longitude * 2 + 2 * n_longitude;
	u32 n_black = (2 + n_latitude * n_longitude);
	u32 n_white = (2 + n_latitude * n_longitude);
	u32 n_vertices  = n_black + n_white;
	auto pos = new vec4_f32[n_vertices];
	auto col = new vec4_f32[n_vertices];
	auto idx = new u32[3*n_triangles];
	
	const u32 north_pole = 0;
	const u32 south_pole = n_latitude*n_longitude+1; 

	auto add_triangle = [&idx, count = 0](u32 A, u32 B, u32 C, u32 offset) mutable
		{
			idx[count++] = A+offset; idx[count++] = B+offset; idx[count++] = C+offset;
		};

	f32 phi = 0;
	f32 theta = 0;
	f32 pi = std::numbers::pi_v<f32>; 
	f32 delta_latitude = pi / n_latitude;
	f32 delta_longitude= 2*pi / n_longitude;

	auto xyz = [](f32 phi, f32 theta)
		{
			return vec4_f32{ std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta), 1.f };
		};

	pos[0] = xyz(0.f, 0.f);
	pos[n_vertices-1] = xyz(pi, 0.f);
	
	for (u32 i = 0; i <= n_latitude; i++)
	{
		phi = 0;
		for (u32 j = 1; j <= n_longitude; j++)
		{
			u32 col_offset = (i+j) % 2 == 1 ? 0 : n_black;
			if (i == 0)
			{
				u32 index_A = north_pole;
				u32 index_B = j;
				u32 index_C = j  % n_longitude +1;
				add_triangle(index_A, index_B, index_C, col_offset);
			}
			else if (i == n_latitude)
			{
				u32 index_A = (i-1) * n_longitude + j;
				u32 index_B = south_pole;
				u32 index_C = (i-1) * n_longitude + j % n_longitude+1;
				add_triangle(index_A, index_B, index_C, col_offset);
				pos[index_A] = xyz(phi, theta);
			}
			else
			{
				u32 index_A = (i - 1) * n_longitude + j;
				u32 index_B = i * n_longitude + j;
				u32 index_C = (i - 1) * n_longitude + j % n_longitude + 1;
				add_triangle(index_A, index_B, index_C, col_offset);

				u32 index_D = index_C;
				u32 index_E = index_B;
				u32 index_F = i * n_longitude + j % n_longitude + 1;
				add_triangle(index_D, index_E, index_F, col_offset);

				pos[index_A] = xyz(phi, theta);
			}
			phi += delta_longitude;
		}
		theta += delta_latitude;
	}

	for (u32 n = 0; n < n_white; n++)
	{
		pos[n + n_black] = pos[n];
		col[n] = vec4_f32(1, 1, 1, 1.);
		col[n+n_black] = vec4_f32(0.0, 0.0, 0.00, 1.0);
	}
	return { 3*n_triangles, n_vertices, idx, pos, col };
}


