#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <utility>
#include <numbers>
#include <filesystem>
#include <string>
#include <span>

#include "common.h"
#include "vec_math.h"

#include "random_int.h"
#include "quaternions.h"
#include "random_sphere.h"

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#include "render/gl_backend.cpp"
#include "render/load_textfile.cpp"

#include "render/checkered_sphere.cpp"
#include "render/arcball.cpp"

// events 

struct mouse
{
	f64 x{},y{};
	bool left_pressed{};
	bool right_pressed{};
} mouse;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_callback		  (GLFWwindow* window, f64 x, f64 y);
void key_callback         (GLFWwindow* window, int key, int scancode, int action, int mode);

struct key_event { int key; int action; };
const u32 n_queue{ 128 };

struct events
{
	u32 begin{};
	u32 end{};
	key_event queue[n_queue];
} events{};

void register_event(key_event event);
bool has_events();
key_event poll_event();
void event_loop(GLFWwindow* window);



// render

const GLuint WIDTH = 2*800, HEIGHT = 2*600;
void update_mvp(mvp& mvp, opengl_context* context);
void draw(opengl_context *context);

color pt_color = color_table[14];
void next_color();

quat arcball{};


int main(void)
{
	auto window = create_window(WIDTH, HEIGHT, "Sphere points");

    glfwSetKeyCallback			(window, key_callback);
	glfwSetCursorPosCallback	(window, mouse_callback); 
	glfwSetMouseButtonCallback	(window, mouse_button_callback);

	// Mesh

	const u32 n_lat = 24;
	const u32 n_long = 48;
	auto sphere = sphere_parametric(n_lat, n_long);

	opengl_context context_mesh = context_for_colored_mesh(sphere.n_vertices, sphere.n_indices);
	upload_mesh(context_mesh, sphere.pos, sphere.n_vertices, sphere.col, sphere.n_indices, sphere.idx);
	
	// OpenGL setup
	
	mvp mvp
	{
		.m = {1.f},
		.v = translation(vec3_f32(0.f, 0.f, -2.f)),
		.p = perspective(90.f, 1.f * WIDTH / HEIGHT, 1.f, 10.),
	};
	auto mvp_pts = mvp;
	map_buffer(mvp.data(), sizeof(mvp), context_mesh.uniform);
	bind_uniform_block(context_mesh.uniform, 0);

	auto make_smaller = [&mvp](opengl_context *context, f32 eps)
	{
		auto s = scale(1.f-eps); 
		mvp.m = mul(s, mvp.m);
		map_buffer(mvp.data(), sizeof(mvp), context->uniform);
	};
	make_smaller(&context_mesh, 0.01);

// Spheres
	u32 level = 8;
	u32 n_points_tree 	   = size_hecke_tree(5,level);
	u32 offset 	 = 0;//size_hecke_tree(5,level-1);
	u32 n_points = n_points_tree - offset;
	auto points = new quat[n_points_tree];
	make_tree(T5.s, 5, points, n_points_tree);

// Convolution Orbit
/*  
	auto orbit = make_hecke_orbit(level, T5br.s, 5);
	points = orbit.T;
	n_points = orbit.size;
	offset = 0;
*/

// Marsaglia
/* 
	u32 level = 8;
	u32 n_points= size_hecke_tree(5,level);// - size_hecke_tree(5,level-1);
	auto points = new quat[n_points];
	xorshift_sampler sampler;
	u32 offset = 0;	
	for(u32 u{}; u<n_points; u++)
	{
		points[u] = sphere4_marsaglia_polar(sampler);
	}
*/

	auto update_cv = [](u32 uniform, u32 i)
	{
		cv0 cvs[3] = {
						{pt_color.c, vec4_f32{1.f,0.f,0.f,1.f}},
						{pt_color.c, vec4_f32{0.f,1.f,0.f,1.f}},
						{pt_color.c, vec4_f32{0.f,0.f,1.f,1.f}}
					};
		map_buffer(cvs[i].data(), sizeof(cv0), uniform);
	};

	opengl_context context_points = context_for_points(n_points, context_mesh.uniform);
	bind_uniform_block(context_mesh.uniform, 0);
	upload_points(context_points, (vec4_f32*)points+offset, n_points);
	bind_uniform_block(context_points.uniform2, 1);
	update_cv(context_points.uniform2, 0);
	glUseProgram(context_mesh.program);

	glDisable(GL_PROGRAM_POINT_SIZE);
	glPointSize(9.-level);
	const GLfloat black[]    = { 0.0f, 0.0f, 0.0f, 1.0f };
	const GLfloat blog_bg[] = {246.f/255, 241.f/255, 241.f/255, 1.f };
	const float far_value = 1.0f;
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

    while (!glfwWindowShouldClose(window)) 
	{
		event_loop(window);

        glClearBufferfv(GL_COLOR, 0, blog_bg);
		glClearBufferfv(GL_DEPTH, 0, &far_value);


		update_mvp(mvp, &context_mesh);
		draw(&context_mesh);

		update_mvp(mvp_pts, &context_points);
		update_cv(context_points.uniform2,0);
		draw(&context_points);
		update_cv(context_points.uniform2,1);
		draw(&context_points);
		update_cv(context_points.uniform2,2);
		draw(&context_points);

		glfwSwapBuffers(window);
    }

    glfwTerminate();

    return 0;
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	register_event({ key, action });
}

void mouse_callback(GLFWwindow *window, f64 x, f64 y)
{
	if (mouse.left_pressed)
	{
		arcball = update_arcball_pressed(x / WIDTH, y / HEIGHT);
	}
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	register_event({ button, action });
}

void register_event(key_event event)
{
	events.queue[events.end] = event;
	events.end = (events.end + 1u) % n_queue;
}

bool has_events()
{
	return events.begin != events.end;
}

void event_loop(GLFWwindow *window)
{
	glfwPollEvents();

	while (has_events())
	{
		auto event = poll_event();
		switch(event.key)
		{
			case GLFW_KEY_R:
			{
				break;
			}
			case GLFW_KEY_C:
			{
				if(event.action == GLFW_PRESS)
					next_color();
				break;
			}
			case GLFW_KEY_T:
			{
				break;
			}
			case GLFW_KEY_ESCAPE:
			{
				glfwSetWindowShouldClose(window, GL_TRUE);
				break;
			}
			case GLFW_KEY_ENTER:
			{
				break;
			}
			case GLFW_KEY_RIGHT:
			{
				break;
			}
			case GLFW_KEY_LEFT:
			{
				break;
			}
			case GLFW_KEY_SPACE:
			{
				break;
			}
			case GLFW_KEY_B:
			{
				break;
			}
			case GLFW_MOUSE_BUTTON_LEFT:
			{
				if(event.action == GLFW_PRESS)
				{
					if(!mouse.left_pressed)
					{
						glfwGetCursorPos(window, &mouse.x, &mouse.y); 
						mouse.x/=WIDTH;
						mouse.y/=HEIGHT;
						mouse.left_pressed = true;
						
						update_arcball_first_press(arcball, mouse.x, mouse.y);
					}
				} 
				else if (event.action == GLFW_RELEASE)
				{
					mouse.left_pressed = false;
				}
				break;
			}	
			default:
				break;
		}
	}
}

key_event poll_event()
{
	assert(has_events() && "polled for events but no events");
	key_event event = events.queue[events.begin];
	events.begin = (events.begin + 1u) % n_queue;
	return event;
}

void update_mvp(mvp& mvp, opengl_context* context)
{
		auto arc   = quat_to_mat4(arcball); 
		auto trans = translation(vec3_f32(0, 0, -2.)); 
		mvp.v      = mul(trans, arc);
		map_buffer(mvp.data(), sizeof(mvp), context->uniform);
}

void draw(opengl_context *context)
{
	glBindVertexArray(context->vao);
	glUseProgram(context->program);

	if (context->draw_mode == opengl_context::draw_mode::array)
	{
		glDrawArrays(context->mode, context->first, context->count);
	}
	else if (context->draw_mode == opengl_context::draw_mode::elements)
	{
		glDrawElements(context->mode, context->count, GL_UNSIGNED_INT, 0);
	}
}

void next_color()
{
	static u32 u = 0;
	pt_color = color_table[u];
	u++;
	u %= sizeof(color_table)/sizeof(color);
}