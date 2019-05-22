cmake_minimum_required(VERSION 3.13)
project(FastDesp)
set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( SOURCE_FILES main.cpp)
include_directories(include/)
include_directories(src/)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


ADD_LIBRARY(${PROJECT_NAME} SHARED src/ComputeFeatures.cpp)

# eigen
include_directories( "/usr/include/eigen3/" )

# pcl
find_package( PCL 1.8 REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS} )
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

#YAML
find_package(  yaml-cpp REQUIRED )
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

#Goole-test
find_package(GTest REQUIRED)
target_link_libraries(${PROJECT_NAME} ${GTEST_INCLUDE_DIRS} )


# CGAL and its components
#[[find_package( CGAL )

if ( CGAL_FOUND )
    include( ${CGAL_USE_FILE} )
else()
    message(STATUS "NOTICE: This program requires the CGAL library, and will not be compiled.")
endif()]]

add_executable(main main.cpp)
target_link_libraries (main ${PROJECT_NAME})

add_executable(runGTest UnitTest/GTest.cpp)
target_link_libraries(runGTest gtest gtest_main)
target_link_libraries(runGTest ${PROJECT_NAME})

add_executable(TEST_DebugExporter TestFile/TestDebugExporter.cpp)
target_link_libraries(TEST_DebugExporter ${PROJECT_NAME})

add_executable(viewResult src/viewResult.cpp)
target_link_libraries(viewResult ${PROJECT_NAME})

add_executable(ResultVisual src/ResultVisual.cpp)
target_link_libraries (ResultVisual ${PROJECT_NAME})

add_executable(ComputeFeatures src/ComputeFeatures.cpp)
target_link_libraries (ComputeFeatures ${PROJECT_NAME})

#[[

add_executable(extractWall extract_walls.cpp)
target_link_libraries (extractWall ${PROJECT_NAME})

add_executable(testCGAL testCGAL.cpp)
target_link_libraries(testCGAL ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})]]
