cmake_minimum_required(VERSION 3.16)
# Set CUDA compiler path (adjust the path to match your CUDA installation)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

project(CUDAVideoProcessing LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Try to find Multimedia and MultimediaWidgets, but don't require them
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Multimedia MultimediaWidgets QUIET)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Define sources, headers, forms, resources
set(PROJECT_SOURCES
    main.cpp
    videosuite.cpp
    videosuite.h
    videosuite.ui
    cuda_functions.h
    cuda_functions.cu
)

# Compile CUDA files
cuda_compile(CUDA_OBJ ${CUDA_SOURCES})

# Qt configuration
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(${PROJECT_NAME}
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        ${CUDA_OBJ}
    )
else()
    add_executable(${PROJECT_NAME}
        ${PROJECT_SOURCES}
        ${CUDA_OBJ}
    )
endif()

# Define libraries to link
set(LINK_LIBRARIES
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
)

# Add Multimedia libraries only if they were found
if(Qt${QT_VERSION_MAJOR}Multimedia_FOUND)
    list(APPEND LINK_LIBRARIES Qt${QT_VERSION_MAJOR}::Multimedia)
endif()

if(Qt${QT_VERSION_MAJOR}MultimediaWidgets_FOUND)
    list(APPEND LINK_LIBRARIES Qt${QT_VERSION_MAJOR}::MultimediaWidgets)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARIES})

# Qt finalization
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${PROJECT_NAME})
endif()
