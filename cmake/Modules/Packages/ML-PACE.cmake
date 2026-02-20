# PACE library support for ML-PACE package
find_package(pace QUIET)

if(pace_FOUND)
    find_package(pace)
    target_link_libraries(lammps PRIVATE pace::pace)
else()
    # set policy to silence warnings about timestamps of downloaded files. review occasionally if it may be set to NEW
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 OLD)
    endif()

    set(PACELIB_URL "https://github.com/thermoatoms/lammps-user-pace/archive/refs/heads/main.tar.gz" CACHE STRING "URL for PACE evaluator library sources")
    mark_as_advanced(PACELIB_URL)

    # LOCAL_ML-PACE points to top-level dir with local lammps-user-pace repo,
    # to make it easier to check local build without going through the public github releases.
    # Auto-detect a fork at the sibling directory lammps-user-pace if LOCAL_ML-PACE is not set.
    if(NOT LOCAL_ML-PACE)
      get_filename_component(_default_pace_dir "${CMAKE_SOURCE_DIR}/../../lammps-user-pace" ABSOLUTE)
      if(EXISTS "${_default_pace_dir}/ML-PACE")
        set(LOCAL_ML-PACE "${_default_pace_dir}")
        message(STATUS "Auto-detected local lammps-user-pace fork: ${LOCAL_ML-PACE}")
      endif()
    endif()
    if(LOCAL_ML-PACE)
     set(lib-pace "${LOCAL_ML-PACE}")
    else()
      # always download fresh from fork to pick up latest changes
      message(STATUS "Downloading ${PACELIB_URL}")
      file(DOWNLOAD ${PACELIB_URL} ${CMAKE_BINARY_DIR}/libpace.tar.gz STATUS DL_STATUS SHOW_PROGRESS)
      if(NOT DL_STATUS EQUAL 0)
        message(FATAL_ERROR "Download of PACE library from ${PACELIB_URL} failed")
      endif()


      # uncompress downloaded sources
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove_directory lammps-user-pace*
        COMMAND ${CMAKE_COMMAND} -E tar xzf libpace.tar.gz
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      get_newest_file(${CMAKE_BINARY_DIR}/lammps-user-pace-* lib-pace)
    endif()

    # some preinstalled yaml-cpp versions don't provide a namespaced target
    find_package(yaml-cpp QUIET)
    if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
      add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()

    # fixup yaml-cpp/emitterutils.cpp for GCC 15+ until patch is applied
    file(READ ${lib-pace}/yaml-cpp/src/emitterutils.cpp yaml_emitterutils)
    string(REPLACE "#include <sstream>" "#include <sstream>\n#include <cinttypes>" yaml_tmp_emitterutils "${yaml_emitterutils}")
    string(REPLACE "#include <cinttypes>\n#include <cinttypes>" "#include <cinttypes>" yaml_emitterutils "${yaml_tmp_emitterutils}")
    file(WRITE ${lib-pace}/yaml-cpp/src/emitterutils.cpp "${yaml_emitterutils}")

    add_subdirectory(${lib-pace} build-pace EXCLUDE_FROM_ALL)
    set_target_properties(pace PROPERTIES CXX_EXTENSIONS ON OUTPUT_NAME lammps_pace${LAMMPS_MACHINE})

    if(CMAKE_PROJECT_NAME STREQUAL "lammps")
      target_link_libraries(lammps PRIVATE pace)
    endif()
endif()
