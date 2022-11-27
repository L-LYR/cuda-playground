cpu_pp_source_file = files(
    'cpu/spp.cc',
    'graph.cc',
)

libcpu_pp = library(
    'cpu_pp',
    cpu_pp_source_file,
    dependencies: deps + dependency('boost'),
    include_directories: incs,
)

gpu_pp_source_file = files(
    'gpu/spp.cu',
)

libgpu_pp = library(
    'gpu_pp',
    'graph.cc',
    gpu_pp_source_file,
    cuda_args: cuda_args,
    dependencies: deps + dependency('cuda'),
    include_directories: incs,
)
