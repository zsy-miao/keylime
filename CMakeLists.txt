cmake_minimum_required(VERSION 3.25)
project(keylime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT: our runtime never shares a CRT heap with the title's (host) msvcp140/vcruntime140.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

file(GLOB_RECURSE KEYLIME_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(miniz STATIC third_party/miniz/miniz.c)
target_include_directories(miniz PUBLIC third_party/miniz)
# AMD's address library (MIT, from Mesa 25.0.0): the console GPU's texture layouts.
file(GLOB_RECURSE ADDRLIB_SOURCES third_party/addrlib/src/*.cpp)
add_library(addrlib STATIC ${ADDRLIB_SOURCES})
target_include_directories(addrlib PUBLIC third_party/addrlib/inc third_party/addrlib/src
    third_party/addrlib/src/core third_party/addrlib/src/chip/gfx9 third_party/addrlib/src/chip/gfx10
    third_party/addrlib/src/chip/gfx11 third_party/addrlib/src/chip/gfx12 third_party/addrlib/src/chip/r800)
target_compile_definitions(addrlib PUBLIC ADDR_FASTCALL= LITTLEENDIAN_CPU DEBUG=0)
target_compile_options(addrlib PRIVATE /w)
add_executable(keylime ${KEYLIME_SOURCES})
target_include_directories(keylime PRIVATE src third_party/vulkan)
target_compile_definitions(keylime PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
target_compile_options(keylime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zi /utf-8 /wd4100>)
target_link_options(keylime PRIVATE /DEBUG /LARGEADDRESSAWARE /DYNAMICBASE)
target_link_libraries(keylime PRIVATE addrlib miniz d3d12 dxgi ole32 user32)
