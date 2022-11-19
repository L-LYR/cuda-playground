project(
    'pssp',
    'cpp',
    'cuda',
    version: '0.1',
    default_options: [
        'warning_level=3',
        'cpp_std=c++17',
    ],
)

add_project_arguments('-Wno-pedantic', language: ['cpp'])
add_project_arguments('-w', language: ['cuda'])

cuda_args = [
    '-extended-lambda',
    '-gencode=arch=compute_50,code=sm_50',
]

fmt = subproject('fmt')
spdlog = subproject('spdlog')

deps = []
deps += fmt.get_variable('fmt_dep')
deps += spdlog.get_variable('spdlog_dep')

incs = []
incs += include_directories('include')

subdir('src')
subdir('test')
