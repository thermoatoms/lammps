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

# GRACE/TensorFlow support (compiled by default unless NO_GRACE_TF is set)
if(NOT DEFINED NO_GRACE_TF)
  # Check if TF_LIB_FILE is provided directly
  if(TF_LIB_FILE)
    message("User-defined TF_LIB_FILE is provided: ${TF_LIB_FILE}")
  else()
    # 1) Try to find TensorFlow library from Python installation
    if(NOT PACE_PYTHON_EXEC)
      find_package(Python COMPONENTS Interpreter QUIET)
      set(PACE_PYTHON_EXEC ${Python_EXECUTABLE})
    endif()
    message("-- Python interpreter found: ${PACE_PYTHON_EXEC}")
    execute_process(
      COMMAND ${PACE_PYTHON_EXEC} -c "import os;import pkgutil;package = pkgutil.get_loader('tensorflow');print(os.path.dirname(package.get_filename()))"
      OUTPUT_VARIABLE TF_DISCOVER
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(STRIP "${TF_DISCOVER}" TF_DISCOVER)
    set(TF_PATH ${TF_DISCOVER})

    if(APPLE)
      set(TF_LIB_FILE "${TF_PATH}/libtensorflow_cc.2.dylib")
    elseif(WIN32)
      set(TF_LIB_FILE "${TF_PATH}/tensorflow.dll")
    else()
      set(TF_LIB_FILE "${TF_PATH}/libtensorflow_cc.so.2")
    endif()
    set(TF_INCLUDE_PATH "${TF_PATH}/include")

    # 2) If not found, download it
    if(NOT EXISTS ${TF_LIB_FILE})
      set(TF_URL_WINDOWS "https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-windows-x86_64.zip")
      set(TF_URL_LINUX   "https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-gpu-linux-x86_64.tar.gz")
      set(TF_URL_MACOS   "https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-cpu-darwin-arm64.tar.gz")

      set(TF_SHA256_WINDOWS "28acdcea6c6b34828cf0e95e67802b0f3577d51bc2e8915de811b7aa0b04452d")
      set(TF_SHA256_LINUX   "6ca25aae03548cf76f6f68f00bdf53ec39710f08cee23bf6419b9e6e27feca5c")
      set(TF_SHA256_MACOS   "462257d2792730dcb131fcf21bc826192ae5a2c418535f6347d051f10fc8be8a")

      set(TF_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/tensorflow-library-download")
      message(STATUS "TensorFlow library not found via Python discovery. Attempting to download.")

      if(WIN32)
        set(TF_URL ${TF_URL_WINDOWS})
        set(TF_SHA256 ${TF_SHA256_WINDOWS})
        set(TF_ARCHIVE "${CMAKE_BINARY_DIR}/libtensorflow.zip")
        set(EXTRACT_COMMAND ${CMAKE_COMMAND} -E tar xf)
      elseif(APPLE)
        set(TF_URL ${TF_URL_MACOS})
        set(TF_SHA256 ${TF_SHA256_MACOS})
        set(TF_ARCHIVE "${CMAKE_BINARY_DIR}/libtensorflow.tar.gz")
        set(EXTRACT_COMMAND ${CMAKE_COMMAND} -E tar xzf)
      else()
        set(TF_URL ${TF_URL_LINUX})
        set(TF_SHA256 ${TF_SHA256_LINUX})
        set(TF_ARCHIVE "${CMAKE_BINARY_DIR}/libtensorflow.tar.gz")
        set(EXTRACT_COMMAND ${CMAKE_COMMAND} -E tar xzf)
      endif()

      if(NOT EXISTS ${TF_ARCHIVE})
        message(STATUS "Downloading TensorFlow C library from ${TF_URL}")
        file(DOWNLOAD ${TF_URL} ${TF_ARCHIVE}
                SHOW_PROGRESS
                EXPECTED_HASH SHA256=${TF_SHA256}
                STATUS DL_STATUS
        )
        list(GET DL_STATUS 0 DL_CODE)
        list(GET DL_STATUS 1 DL_MSG)
        if(NOT DL_CODE EQUAL 0)
          message(FATAL_ERROR "Failed to download TensorFlow from ${TF_URL}. Error: ${DL_MSG}")
        endif()
      else()
        message(STATUS "Using already downloaded archive ${TF_ARCHIVE} (Hash verified)")
      endif()

      message(STATUS "Extracting TensorFlow library archive ${TF_ARCHIVE}")
      execute_process(
        COMMAND ${EXTRACT_COMMAND} ${TF_ARCHIVE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      set(TF_PATH ${CMAKE_BINARY_DIR})

      if(WIN32)
        set(TF_LIB_FILE "${TF_PATH}/lib/tensorflow.dll")
        string(REPLACE ".dll" ".lib" TF_IMPORTS_LIB_FILE "${TF_LIB_FILE}")
      elseif(APPLE)
        set(TF_LIB_FILE "${TF_PATH}/lib/libtensorflow.2.dylib")
      else()
        set(TF_LIB_FILE "${TF_PATH}/lib/libtensorflow.so.2")
      endif()
      set(TF_INCLUDE_PATH "${TF_PATH}/include")
    endif()
  endif()

  # 3) Import library or fail
  if(EXISTS ${TF_LIB_FILE})
    message("-- TensorFlow library is FOUND at ${TF_LIB_FILE}")
    add_library(tensorflow SHARED IMPORTED)
    if(WIN32)
      set_target_properties(tensorflow PROPERTIES
              IMPORTED_LOCATION "${TF_LIB_FILE}"
              IMPORTED_IMPLIB "${TF_IMPORTS_LIB_FILE}"
              INTERFACE_INCLUDE_DIRECTORIES "${TF_INCLUDE_PATH}")
    else()
      set_target_properties(tensorflow PROPERTIES
              IMPORTED_LOCATION "${TF_LIB_FILE}"
              INTERFACE_INCLUDE_DIRECTORIES "${TF_INCLUDE_PATH}")
    endif()

    # cppflow: use local path if provided, otherwise download
    if(DEFINED CPPFLOW_PATH AND EXISTS "${CPPFLOW_PATH}")
      message(STATUS "Using provided local cppflow at: ${CPPFLOW_PATH}")
    else()
      set(CPPFLOW_VERSION "2.0.3aw")
      set(CPPFLOW_URL "https://github.com/ACEworksGmbH/cppflow/archive/refs/tags/v${CPPFLOW_VERSION}.tar.gz" CACHE STRING "URL for cppflow")
      set(CPPFLOW_SHA256 "f1144030aa6d6ed8f1a843f6e5fb5ae4b8e25383620096e2d086f8c27c0a6ef0")

      set(CPPFLOW_ARCHIVE "${CMAKE_BINARY_DIR}/libcppflow.tar.gz")

      if(EXISTS ${CPPFLOW_ARCHIVE})
        file(SHA256 ${CPPFLOW_ARCHIVE} CURRENT_CPPFLOW_SHA256)
        if(NOT CURRENT_CPPFLOW_SHA256 STREQUAL CPPFLOW_SHA256)
          message(WARNING "Existing cppflow archive hash mismatch. Deleting and re-downloading...")
          file(REMOVE ${CPPFLOW_ARCHIVE})
        endif()
      endif()

      if(NOT EXISTS ${CPPFLOW_ARCHIVE})
        message(STATUS "Downloading ${CPPFLOW_URL}")
        file(DOWNLOAD ${CPPFLOW_URL} ${CPPFLOW_ARCHIVE}
                EXPECTED_HASH SHA256=${CPPFLOW_SHA256}
                STATUS DL_CPPFLOW_STATUS
        )
        list(GET DL_CPPFLOW_STATUS 0 DL_CPPFLOW_CODE)
        list(GET DL_CPPFLOW_STATUS 1 DL_CPPFLOW_MSG)
        if(NOT DL_CPPFLOW_CODE EQUAL 0)
          message(FATAL_ERROR "Failed to download cppflow from ${CPPFLOW_URL}. Error: ${DL_CPPFLOW_MSG}")
        endif()
      else()
        message(STATUS "Using already downloaded cppflow archive (Hash verified)")
      endif()

      execute_process(
              COMMAND ${CMAKE_COMMAND} -E remove_directory cppflow-${CPPFLOW_VERSION}
              COMMAND ${CMAKE_COMMAND} -E tar xzf ${CPPFLOW_ARCHIVE}
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      set(CPPFLOW_PATH "${CMAKE_BINARY_DIR}/cppflow-${CPPFLOW_VERSION}")
    endif()

    add_library(cppflow INTERFACE)
    target_include_directories(cppflow
            INTERFACE
            ${tensorflow_INCLUDE_DIRS}
            $<BUILD_INTERFACE:${CPPFLOW_PATH}/include>
    )
    target_compile_features(cppflow INTERFACE cxx_std_17)
    target_link_libraries(cppflow INTERFACE tensorflow)

    set(PACE_TP ON)
    find_package(OpenMP)
  else()
    message("-- TensorFlow library is NEITHER found at ${TF_LIB_FILE} NOR downloaded/extracted")
  endif()
else()
  message("-- NO GRACE/TensorFlow will be compiled (because flag NO_GRACE_TF is set)")
  add_definitions(-DNO_GRACE_TF=1)
endif()

if(CMAKE_PROJECT_NAME STREQUAL "lammps")
  if(DEFINED PACE_TP)
    add_definitions(-DPACE_TP)
    target_link_libraries(lammps PRIVATE tensorflow)
    target_link_libraries(lammps PRIVATE cppflow)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(lammps PUBLIC OpenMP::OpenMP_CXX)
    endif()
  endif()

  if(WIN32)
    if(BUILD_SHARED_LIBS)
      message(STATUS "ML-PACE: Configuring 'lammps' for shared library import (YAML_CPP_DLL)")
      target_compile_definitions(lammps PRIVATE YAML_CPP_DLL)
    else()
      message(STATUS "ML-PACE: Configuring 'lammps' for static library link (YAML_CPP_STATIC_DEFINE)")
      target_compile_definitions(lammps PRIVATE YAML_CPP_STATIC_DEFINE)
    endif()
  endif()
endif()
