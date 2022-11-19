.PHONY: build

export CC=gcc-11
export CXX=g++-11

build:
	meson build && meson compile -C build