# Указываем минимальную версию CMake
cmake_minimum_required(VERSION 3.20)

# Указываем имя проекта
project(AmeliBase LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Устанавливаем флаги компиляции для Debug и Release
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Включаем папку include/ для поиска заголовочных файлов
include_directories(${CMAKE_SOURCE_DIR}/include)

# Сбор всех .cpp файлов из src/
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Устанавливаем тип сборки по умолчанию на Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Set default build type to Release" FORCE)
endif()

# Указываем директории для output в зависимости от типа сборки
set(OUTPUT_BASE_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/Release)

# Создаем исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Выводим текущий режим сборки
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")