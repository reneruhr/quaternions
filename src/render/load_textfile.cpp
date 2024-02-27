#include <filesystem>

struct text_file
{
	u64 size{};
	char *data{};
	char* name{};
};

// Returns zero terminated string
text_file read_file(const char *file_name)
{
	printf("Reading %s\n", file_name);
	FILE* file = fopen(file_name, "r");
	assert(file && "Failed to read.");
	auto size = std::filesystem::file_size(std::filesystem::path(file_name));
	auto data = new char[size + 1];

	fread(data, sizeof(char), size, file);
	fclose(file);

	data[size] = 0;
	auto file_name_ = std::filesystem::path(file_name).filename().string();
	auto name = new char[strlen(file_name_.c_str()) + 1];
    strcpy(name, file_name_.c_str()); 
	return {size, data, name};
}



u32 count_files_in_dir(const char* dir_name, const char* contains)
{
	u32 n_text_files{};

	for (const auto& entry : std::filesystem::directory_iterator(dir_name))
	{
		printf("Parsing %s. Contains %s at %u\n",
			entry.path().filename().string().c_str(), contains,
			entry.path().filename().string().find(contains));

		if (entry.is_regular_file() && entry.path().filename().string().find(contains) != std::string::npos)
			n_text_files++;
	}
	return n_text_files;
}

void read_from_dir(const char* dir_name, text_file** text_files, u32& n_text_files, const char* contains)
{
	if (n_text_files == 0)
	{
		n_text_files = count_files_in_dir(dir_name, contains);
		*text_files = new text_file[n_text_files];
	}

	u32 m{};
	for (const auto& entry : std::filesystem::directory_iterator(dir_name))
	{
		if (m == n_text_files) break;
		if (entry.is_regular_file() && entry.path().filename().string().find(contains) != std::string::npos)
		{
			(*text_files)[m]= read_file(entry.path().string().c_str());
			m++;
		}
	}
}
