cmake_minimum_required(VERSION 3.20)

project(sdl3-gpu-imgui)

find_package(imgui CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)

add_executable(sdl3-gpu-imgui main.cpp)
target_link_libraries( sdl3-gpu-imgui PUBLIC
    imgui::imgui
    SDL3::SDL3
)
