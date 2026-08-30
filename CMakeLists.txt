cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------------
# Single-file deal.II superbuild
#
# Normal use:
#   cmake -S . -B build -DDEAL_II_ROOT=/path/to/dealii/install
#   cmake --build build -j
#
# To download and build deal.II when it is not installed:
#   cmake -S . -B build -DIAH_BOOTSTRAP_DEAL_II=ON
#   cmake --build build -j
#
# The outer build:
#   1. Reuses an existing compatible deal.II installation when one is found.
#   2. Optionally initializes ./dealii as a Git submodule, or downloads deal.II.
#   3. Builds and installs deal.II once under build/_deps/dealii-install.
#   4. Configures and incrementally builds every .cc program.
#
# The IAH_INNER_BUILD option is internal. The outer superbuild invokes this same
# file a second time to configure the actual program targets after deal.II is
# available.
# -----------------------------------------------------------------------------

option(IAH_INNER_BUILD "Configure only the program targets" OFF)

# =============================================================================
# Inner project: create one deal.II executable for every .cc file.
# =============================================================================
if(IAH_INNER_BUILD)
  set(IAH_DEAL_II_MIN_VERSION "9.7.1" CACHE STRING
      "Minimum acceptable deal.II version")
  set(DEAL_II_ROOT "" CACHE PATH
      "deal.II installation prefix or directory containing deal.IIConfig.cmake")

  find_package(deal.II ${IAH_DEAL_II_MIN_VERSION} REQUIRED CONFIG
    HINTS
      "${DEAL_II_ROOT}"
      "$ENV{DEAL_II_ROOT}"
      "$ENV{DEAL_II_DIR}"
  )

  # deal.II intentionally initializes its compiler/cache settings before the
  # application project() command.
  deal_ii_initialize_cached_variables()
  project(IAH_programs LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(IAH_PROGRAM_DIRECTORIES
    accuracy
    lid_driven_cavity
    channel_flow
  )

  set(_iah_target_names "")

  foreach(_iah_directory IN LISTS IAH_PROGRAM_DIRECTORIES)
    file(GLOB _iah_sources CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${_iah_directory}/*.cc"
    )

    # Keep post-processing scripts beside the executables. The synchronization
    # target runs on every build, so a deleted output directory or script is
    # recreated without requiring a clean configure.
    file(GLOB _iah_python_scripts CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/${_iah_directory}/*.py"
    )

    set(_iah_output_directory
        "${CMAKE_BINARY_DIR}/${_iah_directory}")

    string(MAKE_C_IDENTIFIER
      "sync_${_iah_directory}_runtime" _iah_sync_target)
    add_custom_target("${_iah_sync_target}" ALL
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_iah_output_directory}"
      COMMENT "Synchronizing ${_iah_directory} runtime directory"
      VERBATIM
    )

    foreach(_iah_python_script IN LISTS _iah_python_scripts)
      get_filename_component(_iah_python_name "${_iah_python_script}" NAME)
      add_custom_command(TARGET "${_iah_sync_target}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_iah_python_script}"
                "${_iah_output_directory}/${_iah_python_name}"
        COMMENT "Copying ${_iah_python_name}"
        VERBATIM
      )
    endforeach()

    foreach(_iah_source IN LISTS _iah_sources)
      get_filename_component(_iah_target "${_iah_source}" NAME_WE)

      if(TARGET "${_iah_target}")
        message(FATAL_ERROR
          "Duplicate executable target '${_iah_target}'. "
          "Each .cc filename must be unique across all program directories.")
      endif()

      add_executable("${_iah_target}" "${_iah_source}")
      deal_ii_setup_target("${_iah_target}")
      add_dependencies("${_iah_target}" "${_iah_sync_target}")

      # Mirror the source layout inside the build tree:
      #   build/programs/accuracy/accuracy_IAH
      #   build/programs/lid_driven_cavity/cavity_IAH
      #   build/programs/channel_flow/channel_IAH
      set_target_properties("${_iah_target}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_iah_output_directory}"
      )

      # Prevent multi-configuration generators from adding an extra
      # Debug/, Release/, ... directory.
      foreach(_iah_config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_iah_config}" _iah_config_upper)
        set_target_properties("${_iah_target}" PROPERTIES
          "RUNTIME_OUTPUT_DIRECTORY_${_iah_config_upper}"
          "${_iah_output_directory}"
        )
      endforeach()

      list(APPEND _iah_target_names "${_iah_target}")
      message(STATUS
        "Added ${_iah_target} -> ${_iah_output_directory}/${_iah_target}")
    endforeach()
  endforeach()

  if(NOT _iah_target_names)
    message(FATAL_ERROR
      "No .cc files were found in: ${IAH_PROGRAM_DIRECTORIES}")
  endif()

  # A convenient aggregate target inside the program build.
  add_custom_target(all_programs DEPENDS ${_iah_target_names})

  return()
endif()

# =============================================================================
# Outer project: find or build deal.II, then configure the inner project.
# =============================================================================
project(IAH_superbuild LANGUAGES C CXX)

include(ExternalProject)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type for the IAH programs" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

set(IAH_DEAL_II_VERSION "9.7.1" CACHE STRING
    "deal.II version downloaded when no usable installation is found")
set(IAH_DEAL_II_MIN_VERSION "9.7.1" CACHE STRING
    "Minimum acceptable preinstalled deal.II version")
set(IAH_DEAL_II_GIT_REPOSITORY
    "https://github.com/dealii/dealii.git" CACHE STRING
    "deal.II Git repository used by the fallback download")
set(IAH_DEAL_II_GIT_TAG "v${IAH_DEAL_II_VERSION}" CACHE STRING
    "deal.II Git tag, branch, or commit used by the fallback download")
set(IAH_DEAL_II_BUILD_TYPE "Release" CACHE STRING
    "Build type used for the bootstrapped deal.II installation")

option(IAH_FORCE_BOOTSTRAP_DEAL_II
       "Ignore installed deal.II packages and build the local copy" OFF)
option(IAH_BOOTSTRAP_DEAL_II
       "Download and build deal.II when no compatible installation is found" OFF)
option(IAH_DEAL_II_WITH_MPI
       "Build the bootstrapped deal.II with MPI support" OFF)
option(IAH_DEAL_II_WITH_P4EST
       "Build the bootstrapped deal.II with p4est support" OFF)
option(IAH_DEAL_II_UPDATE_DISCONNECTED
       "Do not contact the network after deal.II has been downloaded" ON)

# Additional deal.II options may be supplied as a semicolon-separated list, for
# example:
#   -DIAH_DEAL_II_EXTRA_CMAKE_ARGS="-DDEAL_II_WITH_UMFPACK=ON;-DUMFPACK_DIR=/opt/suitesparse"
set(IAH_DEAL_II_EXTRA_CMAKE_ARGS "" CACHE STRING
    "Additional CMake arguments passed to the deal.II build")
set(DEAL_II_ROOT "" CACHE PATH
    "deal.II installation prefix or directory containing deal.IIConfig.cmake")

if(IAH_DEAL_II_WITH_P4EST AND NOT IAH_DEAL_II_WITH_MPI)
  message(FATAL_ERROR
    "IAH_DEAL_II_WITH_P4EST=ON requires IAH_DEAL_II_WITH_MPI=ON.")
endif()

set(_iah_dependency_root "${CMAKE_BINARY_DIR}/_deps")
set(_iah_dealii_install "${_iah_dependency_root}/dealii-install")
set(_iah_dealii_build   "${_iah_dependency_root}/dealii-build")
set(_iah_dealii_prefix  "${_iah_dependency_root}/dealii-prefix")

# Prefer an explicitly supplied package, then the installation created by an
# earlier run of this superbuild, then any package visible in normal CMake paths.
if(NOT IAH_FORCE_BOOTSTRAP_DEAL_II)
  find_package(deal.II ${IAH_DEAL_II_MIN_VERSION} QUIET CONFIG
    HINTS
      "${DEAL_II_ROOT}"
      "$ENV{DEAL_II_ROOT}"
      "$ENV{DEAL_II_DIR}"
      "${_iah_dealii_install}"
      "${_iah_dealii_install}/lib/cmake/deal.II"
      "${_iah_dealii_install}/share/deal.II"
  )
endif()

if(deal.II_FOUND AND NOT IAH_FORCE_BOOTSTRAP_DEAL_II)
  # find_package sets deal.II_DIR to the directory containing
  # deal.IIConfig.cmake. Pass that exact location to the inner project.
  set(_iah_dealii_package_hint "${deal.II_DIR}")
  add_custom_target(dealii_external)

  message(STATUS
    "Using existing deal.II ${deal.II_VERSION} from ${deal.II_DIR}")
elseif(NOT IAH_BOOTSTRAP_DEAL_II AND NOT IAH_FORCE_BOOTSTRAP_DEAL_II)
  message(FATAL_ERROR
    "A compatible deal.II installation (version ${IAH_DEAL_II_MIN_VERSION} or newer) "
    "was not found.\n"
    "Set deal.II_DIR to the directory containing deal.IIConfig.cmake, set "
    "DEAL_II_ROOT to the installation prefix, or configure with "
    "-DIAH_BOOTSTRAP_DEAL_II=ON.")
else()
  find_package(Git REQUIRED)

  # Source selection, in order:
  #   1. An already initialized ./dealii submodule/source tree.
  #   2. An uninitialized ./dealii submodule, initialized during the build.
  #   3. A shallow fallback clone under build/_deps.
  if(EXISTS "${CMAKE_SOURCE_DIR}/dealii/CMakeLists.txt")
    set(_iah_dealii_source "${CMAKE_SOURCE_DIR}/dealii")
    set(_iah_dealii_download_args DOWNLOAD_COMMAND "")
    message(STATUS "Using initialized deal.II source: ${_iah_dealii_source}")

  elseif(EXISTS "${CMAKE_SOURCE_DIR}/.gitmodules")
    set(_iah_dealii_source "${CMAKE_SOURCE_DIR}/dealii")
    set(_iah_dealii_download_args
      DOWNLOAD_COMMAND
        "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}"
        submodule update --init --recursive --depth 1 --recommend-shallow dealii
    )
    message(STATUS
      "The deal.II submodule will be initialized during the first build.")

  else()
    set(_iah_dealii_source "${_iah_dependency_root}/dealii-src")
    set(_iah_dealii_download_args
      GIT_REPOSITORY "${IAH_DEAL_II_GIT_REPOSITORY}"
      GIT_TAG        "${IAH_DEAL_II_GIT_TAG}"
      GIT_SHALLOW    TRUE
      GIT_PROGRESS   TRUE
    )
    message(STATUS
      "No deal.II submodule was found; ${IAH_DEAL_II_GIT_TAG} will be downloaded.")
  endif()

  set(_iah_dealii_cmake_args
    "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
    "-DCMAKE_BUILD_TYPE:STRING=${IAH_DEAL_II_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
    "-DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF"
    "-DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF"
    "-DDEAL_II_COMPONENT_PACKAGE:BOOL=OFF"
    "-DDEAL_II_COMPONENT_PYTHON_BINDINGS:BOOL=OFF"
    "-DDEAL_II_ALLOW_BUNDLED:BOOL=ON"
    "-DDEAL_II_WITH_MPI:BOOL=${IAH_DEAL_II_WITH_MPI}"
    "-DDEAL_II_WITH_P4EST:BOOL=${IAH_DEAL_II_WITH_P4EST}"
  )

  if(CMAKE_TOOLCHAIN_FILE)
    list(APPEND _iah_dealii_cmake_args
      "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}")
  endif()

  if(CMAKE_OSX_ARCHITECTURES)
    list(APPEND _iah_dealii_cmake_args
      "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}")
  endif()

  if(CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _iah_dealii_cmake_args
      "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()

  if(IAH_DEAL_II_EXTRA_CMAKE_ARGS)
    list(APPEND _iah_dealii_cmake_args ${IAH_DEAL_II_EXTRA_CMAKE_ARGS})
  endif()

  ExternalProject_Add(dealii_external
    PREFIX      "${_iah_dealii_prefix}"
    SOURCE_DIR  "${_iah_dealii_source}"
    BINARY_DIR  "${_iah_dealii_build}"
    INSTALL_DIR "${_iah_dealii_install}"

    ${_iah_dealii_download_args}

    UPDATE_DISCONNECTED ${IAH_DEAL_II_UPDATE_DISCONNECTED}
    UPDATE_COMMAND ""
    CMAKE_ARGS ${_iah_dealii_cmake_args}
    TEST_COMMAND ""

    # No BUILD_ALWAYS here: ExternalProject's stamp files ensure that deal.II
    # is configured, compiled, and installed only when necessary.
    USES_TERMINAL_DOWNLOAD  TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
  )

  # Passing the installation prefix is sufficient; the inner find_package()
  # searches the standard deal.II package locations below this prefix.
  set(_iah_dealii_package_hint "${_iah_dealii_install}")
endif()

set(_iah_program_build "${CMAKE_BINARY_DIR}/programs")

set(_iah_program_cmake_args
  "-DIAH_INNER_BUILD:BOOL=ON"
  "-DIAH_DEAL_II_MIN_VERSION:STRING=${IAH_DEAL_II_MIN_VERSION}"
  "-DDEAL_II_ROOT:PATH=${_iah_dealii_package_hint}"
  "-DCMAKE_PREFIX_PATH:STRING=${_iah_dealii_package_hint}"
  "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
)

if(CMAKE_BUILD_TYPE)
  list(APPEND _iah_program_cmake_args
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_TOOLCHAIN_FILE)
  list(APPEND _iah_program_cmake_args
    "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

if(CMAKE_OSX_ARCHITECTURES)
  list(APPEND _iah_program_cmake_args
    "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}")
endif()

if(CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND _iah_program_cmake_args
    "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

ExternalProject_Add(iah_programs
  PREFIX     "${CMAKE_BINARY_DIR}/iah-programs-prefix"
  SOURCE_DIR "${CMAKE_SOURCE_DIR}"
  BINARY_DIR "${_iah_program_build}"

  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND   ""
  CMAKE_ARGS       ${_iah_program_cmake_args}
  INSTALL_COMMAND  ""
  TEST_COMMAND     ""

  DEPENDS dealii_external

  # Always ask the inner build system to build. Its own dependency graph is
  # incremental, so unchanged .cc files are not recompiled. This is necessary
  # because ExternalProject cannot otherwise see edits in the parent source tree.
  BUILD_ALWAYS TRUE

  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD     TRUE
)

message(STATUS "Program executables will be placed under:")
message(STATUS "  ${_iah_program_build}/accuracy")
message(STATUS "  ${_iah_program_build}/lid_driven_cavity")
message(STATUS "  ${_iah_program_build}/channel_flow")