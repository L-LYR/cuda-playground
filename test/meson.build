gtest = subproject('gtest')

test_deps = deps
test_deps += gtest.get_variable('gtest_dep')


executable(
    'cpu_spp_test',
    'cpu_spp_test.cc',
    include_directories: incs,
    dependencies: test_deps,
    link_with: libcpu_pp,
)

executable(
    'boost_spp_test',
    'boost_spp_test.cc',
    include_directories: incs,
    dependencies: test_deps + dependency('boost'),
    link_with: libcpu_pp,
)

executable(
    'gpu_spp_test',
    'gpu_spp_test.cu',
    cuda_args: cuda_args,
    include_directories: incs,
    dependencies: test_deps + dependency('cuda'),
    link_with: libgpu_pp,
)
