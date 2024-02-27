quat arcball_init{};
const f32 arcball_radius = 0.5f;
quat arcball_cur{};

quat calculate_arcball(f32 x, f32 y) ;

quat update_arcball_pressed(f32 x, f32 y)
{
		arcball_cur = calculate_arcball(x, y);
		return arcball_cur * arcball_init;
}

void update_arcball_first_press(quat arcball, f32 x, f32 y)
{
	arcball_init = ~calculate_arcball(x,y)*arcball;
}

quat calculate_arcball(f32 x, f32 y)
{
	x = 2*x-1;
	y = -2*y+1;
	x /= arcball_radius;
	y /= arcball_radius;
	f32 z = x*x+y*y;
	if (z <= 1.f) {
		return quat{ 0,x,y,std::sqrt(std::max(0.f,1.f - z)) };
	}
	else {
		z = std::sqrt(std::max(0.f,z));
		return quat{0,x/z,y/z,0};
	}
};
