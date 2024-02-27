#include "quaternions.h"

S_5_biased_reordered T5;
S_5ib T5i;
S_13 T13;
S_17_reordered T17;
S_29_reordered T29;
S_29i_reordered T29i;

//Assumption on quat table T: T[i]*T[(i+(p+1)/2)%(p+1)] = id
quat_orbit make_tree(u32 level, const quat *T, u32 p)
{
	struct quat_coord
	{
		quat q{};
		u32  last{}; //last choice
	};

	quat q;
	u32 n_last_level = p+1;
	quat_coord* last_level = new quat_coord[n_last_level];
	u32 u = 0;

	u32 n_tree = (u32)size_hecke_tree(p, level) - 1; // skip identity element
	quat *tree = new quat[n_tree]; 

	// Initial sphere has all p+1 neighbours
	for(u32 c=0; c<p+1; c++)
		last_level[c] = {T[c], c};

	for(u32 c=0; c<n_last_level; c++)
		tree[u++] = last_level[c].q;

	// Recursively add p new neighbours to each quat
	while(--level)
	{
		u32 n_next_level = p*n_last_level;
		quat_coord* next_level = new quat_coord[n_next_level];
		assert(next_level);

		u32 j = 0;
		for(u32 i=0; i<n_last_level; i++)
		{
			for(u32 c=0; c< p+1; c++)
				if( (c+(p+1)/2) % (p+1) != last_level[i].last )
					next_level[j++] = {T[c]*last_level[i].q, c};
		}

		for(u32 c=0; c<n_next_level; c++)
			tree[u++] = next_level[c].q;

		n_last_level = n_next_level;
		delete[] last_level;
		last_level = next_level;
	}

	delete[] last_level;

	return {n_tree, tree};
}

void make_tree(const quat *T, u32 p, quat* v, u32 n)
{
	struct quat_coord
	{
		quat q;
		u32  last; //last choice
	};

	quat q;
	u32 n_last_level = p+1;
	quat_coord* last_level = new quat_coord[n_last_level];
	u32 u = 0;

	// Initial sphere has all p+1 neighbours
	for(u32 c=0; c<p+1; c++)
		last_level[c] = {T[c], c};

	for(u32 c=0; c<n_last_level && u<n; c++)
		v[u++] = last_level[c].q;

	// Recursively add p new neighbours to each quat
	while(u<n)
	{
		u32 n_next_level = p*n_last_level;
		quat_coord* next_level = new quat_coord[n_next_level];
		assert(next_level);

		u32 j = 0;
		for(u32 i=0; i<n_last_level; i++)
		{
			for(u32 c=0; c< p+1; c++)
				if( (c+(p+1)/2) % (p+1) != last_level[i].last )
					next_level[j++] = {T[c]*last_level[i].q, c};
		}

		for(u32 c=0; c<n_next_level && u<n; c++)
			v[u++] = next_level[c].q;

		n_last_level = n_next_level;
		delete[] last_level;
		last_level = next_level;
	}

	delete[] last_level;
}

quat_orbit make_hecke_orbit(u32 n, const quat *T, u32 p)
{
	u32 size = static_cast<u32>(std::pow(p+1,n));
	quat *orbit = new quat[size];

	u32 skip{1};
	for(u32 k{0}; k < n; k++){
		for(u32 i{0}; i<size; i++){
			orbit[i] = T[(i/skip)%(p+1)] * orbit[i];
		}
		skip*=(p+1);
	}
	return {size, orbit};
}