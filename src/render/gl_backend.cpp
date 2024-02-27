//#define GLFW_INCLUDE_GLEXT
#include <glad/gl.h>
#include <GLFW/glfw3.h>


struct opengl_constants
{
	int shared_memory_size; 
	int work_group_count_x; 
	int work_group_count_y; 
	int work_group_count_z; 
	int local_size; 
	int local_size_x; 
	int local_size_y; 
	int local_size_z; 

	int storage_block_size;
};


void query_opengl_constants(opengl_constants& constants)
{
/*
void GetBooleanv( enum pname, boolean *data );
void GetIntegerv( enum pname, int *data );
void GetInteger64v( enum pname, int64 *data );
void GetFloatv( enum pname, float *data );
void GetDoublev( enum pname, double *data );
*/

glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &constants.shared_memory_size);

glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0, &constants.work_group_count_x);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,1, &constants.work_group_count_y);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,2, &constants.work_group_count_z);


glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &constants.local_size);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0, &constants.local_size_x);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,1, &constants.local_size_y);
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,2, &constants.local_size_z);


glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &constants.storage_block_size);

}


struct storage_info
{
	u32 buffer;
	u32 binding;
	u32 size;
	u32 *data;
};

struct opengl_context
{
	GLuint program;

	GLenum mode;
	GLenum first;
	GLenum count;
	enum class draw_mode { array, elements} draw_mode;

	GLuint vao;
	GLuint vbo;
	GLuint vbo2;
	GLuint ebo;

	GLuint uniform;
	GLuint uniform2;

	u32 n_storages;
	storage_info *storages;

	u32 n_work_groups_x;
	u32 n_work_groups_y;
	u32 n_work_groups_z;
	u32 local_size_x;
	u32 local_size_y;
	u32 local_size_z;
};


// Vertex Array structures 
GLuint create_vao();
void   create_vertexarray(opengl_context*, u32 location = 0);
void   bind_ebo(opengl_context* context);

// Shaders 
GLuint compile_shaders(const GLchar* frag, const GLchar* vert);
GLuint compile_shaders(const GLchar* comp);


// Mutable data buffers
GLuint create_buffer(u32 size, GLenum mode = GL_STATIC_DRAW);
void   map_buffer(void *data, u32 size, GLuint buffer);

// Immutable data buffers
void   bind_uniform_block(u32 uniform, u32 bind_point);
void   bind_storage_buffer(u32 buffer, u32 bind_point);

struct mvp 
{
	mat4_f32 m{1.f};
	mat4_f32 v{1.f};
	mat4_f32 p{1.f};

	f32* data() { return static_cast<f32*>(m.m); }
};

struct color
{
	vec4_f32 c{1.f};
	f32* data() { return static_cast<f32*>(&c.x); }
};

struct cv0 
{
	vec4_f32 c{1.f};
	vec4_f32 v0{1.f,0.f,0.f,1.f};
	f32* data() { return static_cast<f32*>(&c.x); }
};

opengl_context context_for_points(u32 n_vertices, u32 uniform = 0)
{
		return {
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 q;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											layout (std140, binding = 1) uniform Data 
												{
												vec4 c;
												vec4 v0;
												};
											out vec4 vert_color;

											vec3 quat_mult(vec4 q, vec3 v)
											{
											return v + 2*cross(q.yzw, (q.x*v + cross(q.yzw, v)));
											}

											void main(void)
											{
												vec4 w = vec4(quat_mult(q,v0.xyz), 1);
												gl_Position = p*(v*(m*w));
												vert_color = c;
											}
											)", 
											R"(
											#version 460 core
											in  vec4 vert_color;
											out vec4 color;
											void main(void)
											{
												color = vert_color;
											}
											)"),
		.mode			= GL_POINTS,
		.first			= 0,
		.count			= n_vertices,
		.draw_mode		= opengl_context::draw_mode::array,
		.vao			= create_vao(),
		.vbo			= create_buffer(4*sizeof(f32)*n_vertices),
		.vbo2			= 0,
		.ebo			= 0,
		.uniform		= uniform > 0 ? uniform : create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
		.uniform2		= create_buffer(sizeof(cv0), GL_DYNAMIC_DRAW),
	};
}

void upload_points(opengl_context& context, vec4_f32 *pos, u32 n_vertices)
{
	create_vertexarray(&context);
	map_buffer(pos, 4 * sizeof(f32) * n_vertices, context.vbo);
}

opengl_context context_for_colored_mesh(u32 n_vertices, u32 n_indices, u32 uniform = 0)
{
	return 
	{
		.program		= compile_shaders( R"(
											#version 460 core 
											layout (location = 0) in vec4 pos;
											layout (location = 1) in vec4 col;
											layout (std140, binding = 0) uniform Transform
												{
												mat4 m;
												mat4 v;
												mat4 p;
												};
											out vec4 frag_color;
											void main(void)
											{
												gl_Position = p*(v*(m*pos));
												frag_color = col;
											}
											)", 
											R"(
											#version 460 core
											in vec4 frag_color;
											out vec4 color;
											void main(void)
											{
												color = frag_color;
											}
											)"),
		.mode			= GL_TRIANGLES,
		.first			= 0,
		.count			= n_indices,
		.draw_mode		= opengl_context::draw_mode::elements,
		.vao			= create_vao(),
		.vbo			= create_buffer(4*sizeof(f32)*n_vertices),
		.vbo2			= create_buffer(4*sizeof(f32)*n_vertices),
		.ebo			= create_buffer(sizeof(u32)*n_indices),
		.uniform		= uniform > 0 ? uniform : create_buffer(sizeof(mvp), GL_DYNAMIC_DRAW),
	};
}


void upload_mesh(opengl_context& context, vec4_f32 *pos, u32 n_vertices, vec4_f32 *col, u32 n_indices, u32 *idx)
{
	map_buffer(pos, 4*sizeof(f32)*n_vertices, context.vbo);
	map_buffer(col, 4*sizeof(f32)*n_vertices, context.vbo2);
	map_buffer(idx,	 sizeof(u32)*n_indices, context.ebo);

	bind_ebo(&context);
	create_vertexarray(&context, 0);
	create_vertexarray(&context, 1);
}
	

GLFWwindow* create_window(u32 w, u32 h, const char* title)
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(w, h, title, NULL, NULL);
    glfwMakeContextCurrent(window);

    int version = gladLoadGL(glfwGetProcAddress);
    printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

	return window;
}


void create_vertexarray(opengl_context *context, u32 location)
{
	u32 binding = location;
	glEnableVertexArrayAttrib(context->vao, location);
	glVertexArrayAttribFormat(context->vao, location, 4, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(context->vao, location, binding );
	glVertexArrayVertexBuffer(context->vao, binding, location == 0 ? context->vbo : context->vbo2, 0, sizeof(vec4_f32));
}

void bind_uniform_block(GLuint uniform, u32 bind_point = 0)
{
	glBindBufferBase(GL_UNIFORM_BUFFER, bind_point, uniform);
}

void map_buffer(void *data, u32 size, GLuint buffer)
{
	void* gpu_handle = glMapNamedBuffer(buffer, GL_WRITE_ONLY);
	memcpy(gpu_handle, data, size);
	glUnmapNamedBuffer(buffer);
}

GLuint create_buffer(u32 size, GLenum mode)
{
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferData(handle, size, 0, mode);
	return handle;
}

GLuint buffer_storage(u32 size, void  *data, GLenum mode)
{
	GLuint handle;
	glCreateBuffers(1, &handle);
	glNamedBufferStorage(handle, size, data, mode);
	return handle;
}

void bind_storage_buffer(GLuint buffer, u32 bind_point = 0)
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_point, buffer);
}

void* gpu_to_cpu_persistent(u32 buffer, u32 size)
{
	u32 scratch_buffer = buffer_storage(size, nullptr, GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_READ_BIT);
	glCopyNamedBufferSubData(buffer, scratch_buffer, 0, 0, size);
	auto p = glMapNamedBufferRange(	scratch_buffer, 0, size, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	return p;
}

void gpu_to_cpu(u32 buffer, void* cpu_buffer, u32 size)
{
	glGetNamedBufferSubData(buffer, 0, size, cpu_buffer);
}


GLuint create_vao()
{
	GLuint handle;
	glCreateVertexArrays(1, &handle);
	return handle;
}

void bind_ebo(opengl_context* context)
{
	if(context->draw_mode == opengl_context::draw_mode::elements)
		glVertexArrayElementBuffer(context->vao, context->ebo);
}

enum class shader_log { shader, program };

void compile_info(GLuint handle, const char* debug_name, shader_log type){
    char buffer[8192];
	GLsizei length = 0;
	if(type == shader_log::shader )
		glGetShaderInfoLog(handle, sizeof(buffer), &length, buffer);
	else
		glGetProgramInfoLog(handle, sizeof(buffer), &length, buffer);
    if (length)
	{
		printf("%s (File: %s)\n", buffer, debug_name);
		assert(false);
	}
};

GLuint compile_shaders(const GLchar* vert_shader_src, const GLchar* frag_shader_src)
{
	GLuint vert_shader;
	GLuint frag_shader;
	GLuint program;

	vert_shader = glCreateShader(GL_VERTEX_SHADER);
	frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vert_shader, 1, &vert_shader_src, 0);
	compile_info(vert_shader, "Vertex shader", shader_log::shader);
	glShaderSource(frag_shader, 1, &frag_shader_src, 0);
	compile_info(frag_shader, "Fragment shader", shader_log::shader);
	glCompileShader(vert_shader);
	glCompileShader(frag_shader);
	program = glCreateProgram();
	glAttachShader(program, vert_shader);
	glAttachShader(program, frag_shader);
	glLinkProgram(program);
	compile_info(program, "Program", shader_log::program);
	
	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);

	return program;
}

GLuint compile_shaders(const GLchar* comp_shader_src)
{
	GLuint comp_shader;
	GLuint program;

	comp_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(comp_shader, 1, &comp_shader_src, 0);
	compile_info(comp_shader, "Compute shader", shader_log::shader);
	glCompileShader(comp_shader);
	program = glCreateProgram();
	glAttachShader(program, comp_shader);
	glLinkProgram(program);
	compile_info(program, "Program", shader_log::program);

	glDeleteShader(comp_shader);

	return program;
}


struct opengl_profiler
{
	const u32 lag_query{8};
	const u32 n_queries{16};
	u32 queries[16];
	u32 current_query{};

	u32 n_data{};
	u64* data{};
};

void init(opengl_profiler* p)
{
	glGenQueries(p->n_queries, p->queries);
}

// https://registry.khronos.org/OpenGL/extensions/ARB/ARB_timer_query.txt	
void profile_compute(opengl_context *context, opengl_profiler *p)
{
	u32& u = p->current_query;
	glUseProgram(context->program);

	glBeginQuery(GL_TIME_ELAPSED, p->queries[u % p->n_queries]);

	glDispatchCompute(context->n_work_groups_x, context->n_work_groups_y, context->n_work_groups_z);

	glEndQuery(GL_TIME_ELAPSED);

	if (u >= p->lag_query)
	{
		GLint available{};
		while (!available)
		{
			glGetQueryObjectiv(p->queries[(u - p->lag_query) % p->n_queries], GL_QUERY_RESULT_AVAILABLE, &available);
		}

		u64 ns_elapsed{};
		glGetQueryObjectui64v(p->queries[(u - p->lag_query) % p->n_queries], GL_QUERY_RESULT, &ns_elapsed);
		if (u < p->n_data + p->lag_query)
			p->data[u - p->lag_query] = ns_elapsed;
	}

	u++;
}



color color_table[] =
{
 	vec4_f32(0.7254901960784313, 0.5647058823529412, 0.5843137254901961, 1.0),
    vec4_f32(0.9882352941176471, 0.7098039215686275, 0.6745098039215687, 1.0),
    vec4_f32(0.7098039215686275, 0.8980392156862745, 0.8117647058823529, 1.0),
    vec4_f32(0.23921568627450981, 0.3568627450980392, 0.34901960784313724, 1.0),
    vec4_f32(0.9372549019607843, 0.48627450980392156, 0.5568627450980392, 1.0),
    vec4_f32(0.9803921568627451, 0.9098039215686274, 0.8784313725490196, 1.0),
    vec4_f32(0.7137254901960784, 0.8862745098039215, 0.8274509803921568, 1.0),
    vec4_f32(0.8470588235294118, 0.6549019607843137, 0.6941176470588235, 1.0),
    vec4_f32(0.4627450980392157, 0.7254901960784313, 0.2784313725490196, 1.0),
	vec4_f32(0.6941176470588235, 0.8470588235294118, 0.7176470588235294, 1.0),
    vec4_f32(0.1843137254901961, 0.3215686274509804, 0.2, 1.0),
    vec4_f32(0.5803921568627451, 0.788235294117647, 0.45098039215686275, 1.0),
    vec4_f32(0.1843137254901961, 0.9529411764705882, 0.8784313725490196, 1.0),
    vec4_f32(0.9725490196078431, 0.8235294117647058, 0.06274509803921569, 1.0),
    vec4_f32(0.9803921568627451, 0.14901960784313725, 0.6274509803921569, 1.0),
    vec4_f32(0.9607843137254902, 0.09019607843137255, 0.12549019607843137, 1.0),
    vec4_f32(0.9921568627450981, 0.4980392156862745, 0.12549019607843137, 1.0),
    vec4_f32(0.9882352941176471, 0.1803921568627451, 0.12549019607843137, 1.0),
    vec4_f32(0.9921568627450981, 0.7176470588235294, 0.3137254901960784, 1.0),
    vec4_f32(0.00392156862745098, 0.00392156862745098, 0.0, 1.0),
    vec4_f32(0.8980392156862745, 0.8666666666666667, 0.7843137254901961, 1.0),
    vec4_f32(0.00392156862745098, 0.5803921568627451, 0.6039215686274509, 1.0),
    vec4_f32(0.0, 0.2627450980392157, 0.4117647058823529, 1.0),
    vec4_f32(0.8588235294117647, 0.12156862745098039, 0.2823529411764706, 1.0)
};