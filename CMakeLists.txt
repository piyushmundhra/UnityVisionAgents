cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(depth_estimation)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++14-extensions")

set(ONNXRUNTIME_DIR "onnxruntime")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
list(APPEND CMAKE_LIBRARY_PATH ${ONNXRUNTIME_LIBRARY_DIR})
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_LIBRARY_DIR})

set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("./eigen-3.4.0")

find_package(OpenCV REQUIRED)

# add_executable(dqn src/dqn.cpp)
# target_link_libraries(dqn PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_include_directories(dqn PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
# set_property(TARGET dqn PROPERTY CXX_STANDARD 20)

# add_executable(preprocessor src/preprocessor.cpp)
# target_link_libraries(preprocessor PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_include_directories(preprocessor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
# set_property(TARGET preprocessor PROPERTY CXX_STANDARD 20)

# add_executable(preprocessor src/preprocessor.cpp)
# target_link_libraries(preprocessor PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_include_directories(preprocessor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "./include")
# set_property(TARGET preprocessor PROPERTY CXX_STANDARD 20)

# add_library(preprocessor_lib SHARED src/preprocessor.cpp)
# target_link_libraries(preprocessor_lib PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_include_directories(preprocessor_lib PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "./include") 
# set_property(TARGET preprocessor_lib PROPERTY CXX_STANDARD 20)

file(GLOB SOURCES "src/*.cpp")

# add_executable(agent ${SOURCES})
# target_link_libraries(agent PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
# target_include_directories(agent PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "./include")
# set_property(TARGET agent PROPERTY CXX_STANDARD 20)

add_library(agent_lib SHARED ${SOURCES})
target_link_libraries(agent_lib PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_include_directories(agent_lib PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "./include") 
set_property(TARGET agent_lib PROPERTY CXX_STANDARD 20)
