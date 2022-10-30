# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (video_traffic)


# Create a sources variable with a link to all cpp files to compile
set (VideoFlows_sou 
    src/VideoTraffic/main_video_flows.cpp
    src/VideoTraffic/video_flows_set.cpp
)

set(flow_HT_sou
    src/VideoTraffic/main_flow_HT.cpp
    src/VideoTraffic/flow_HT.cpp
    src/_TLS/TLS_tools.cpp
    )

# Add an executable with the above sources
add_executable(VideoFlows ${VideoFlows_sou})
add_executable(FlowHttpTls ${flow_HT_sou})


# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(VideoFlows PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(FlowHttpTls PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(VideoFlows ${PROJECT_SOURCE_DIR}/lib/liblibconfig++.dll.a )
target_link_libraries(VideoFlows ${PROJECT_SOURCE_DIR}/lib/libFilterSE.a )
target_link_libraries(VideoFlows ${PROJECT_SOURCE_DIR}/lib/libHashSE.a )
target_link_libraries(VideoFlows ${PROJECT_SOURCE_DIR}/lib/libPcapSE.a )
target_link_libraries(VideoFlows ${PROJECT_SOURCE_DIR}/lib/libPacketSE.a )

target_link_libraries(FlowHttpTls ${PROJECT_SOURCE_DIR}/lib/liblibconfig++.dll.a )
target_link_libraries(FlowHttpTls ${PROJECT_SOURCE_DIR}/lib/libFlow2SE.a)
target_link_libraries(FlowHttpTls ${PROJECT_SOURCE_DIR}/lib/libHashSE.a)
target_link_libraries(FlowHttpTls ${PROJECT_SOURCE_DIR}/lib/libPcapSE.a)
target_link_libraries(FlowHttpTls ${PROJECT_SOURCE_DIR}/lib/libPacketSE.a)

