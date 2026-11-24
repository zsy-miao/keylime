cmake_minimum_required(VERSION 3.25)
project(keylime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT: our runtime never shares a CRT heap with the title's (host) msvcp140/vcruntime140.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

file(GLOB_RECURSE KEYLIME_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(miniz STATIC third_party/miniz/miniz.c)
target_include_directories(miniz PUBLIC third_party/miniz)
add_executable(keylime ${KEYLIME_SOURCES})
target_include_directories(keylime PRIVATE src)
target_compile_definitions(keylime PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
target_compile_options(keylime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zi /utf-8 /wd4100>)
target_link_options(keylime PRIVATE /DEBUG /LARGEADDRESSAWARE /DYNAMICBASE)
target_link_libraries(keylime PRIVATE miniz d3d12 dxgi ole32 user32)
