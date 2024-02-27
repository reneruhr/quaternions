for %%f in (vk_*.comp) do (
    glslangValidator -e main -V -o %%f_debug.spv --target-env vulkan1.3 %%f -DDEBUG
)