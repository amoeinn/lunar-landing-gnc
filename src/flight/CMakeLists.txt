# Flight code of PLAN.md phase 4: the guidance and navigation step in C++17
# with Eigen 3.4, the convex solve through Clarabel.cpp.
#
# Build (examples/build_flight.sh runs exactly this):
#   cmake -S src/flight -B work/build/flight -DCMAKE_BUILD_TYPE=Release \
#         -DPython_EXECUTABLE=.venv/bin/python
#   cmake --build work/build/flight -j
# The pybind11 module lands in src/lunar_landing_gnc as _flight; the Python
# tests that need it skip when it is not built.
cmake_minimum_required(VERSION 3.16)
project(lunar_flight C CXX)

# Compiler flags as recorded in PLAN.md (step 4.0 revisions): C++17 without
# GNU extensions, -O2, no multiply-add contraction (GCC contracts by default in
# gnu++17 mode, which changes rounding against numpy), no -march=native, no
# fast-math.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(LLG_FLIGHT_FLAGS -ffp-contract=off)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CLARABEL_CPP_DIR "${REPO_ROOT}/src/extern/Clarabel.cpp")

find_package(Eigen3 3.4 REQUIRED CONFIG)
find_package(Threads REQUIRED)

# Clarabel.cpp, through add_subdirectory for its configuration and headers.
# Its own cargo target is not used: it runs `cargo install cbindgen` (an
# install from the network) and `cargo build` without --locked or --offline.
# The Rust library is built here instead, offline against the recorded lock
# file, with the build products kept out of the submodule.
add_subdirectory("${CLARABEL_CPP_DIR}" "${CMAKE_BINARY_DIR}/clarabel" EXCLUDE_FROM_ALL)

set(CARGO_LOCK "${CLARABEL_CPP_DIR}/rust_wrapper/Cargo.lock")
set(CARGO_LOCK_SHA256 "b5e47392fba6493cc13968eaa10a6e34162edc93f1ea6b90fff106cd987e97e5")
if(NOT EXISTS "${CARGO_LOCK}")
  message(FATAL_ERROR "Cargo.lock missing in ${CLARABEL_CPP_DIR}/rust_wrapper")
endif()
file(SHA256 "${CARGO_LOCK}" lock_hash)
if(NOT lock_hash STREQUAL CARGO_LOCK_SHA256)
  message(FATAL_ERROR "Cargo.lock is not the recorded one: ${lock_hash}")
endif()
find_program(CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
file(STRINGS "${CLARABEL_CPP_DIR}/Clarabel.rs/Cargo.toml" rs_version REGEX "^version = " LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([^\"]+)\".*" "\\1" CLARABEL_RS_VERSION "${rs_version}")
message(STATUS "Clarabel.rs ${CLARABEL_RS_VERSION}, Cargo.lock ${lock_hash}")

set(CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
set(CLARABEL_STATIC_LIB "${CARGO_TARGET_DIR}/release/libclarabel_c.a")
add_custom_command(
  OUTPUT "${CLARABEL_STATIC_LIB}"
  COMMAND "${CARGO}" build --release --locked --offline --target-dir "${CARGO_TARGET_DIR}"
  WORKING_DIRECTORY "${CLARABEL_CPP_DIR}/rust_wrapper"
  DEPENDS "${CARGO_LOCK}" "${CLARABEL_CPP_DIR}/rust_wrapper/Cargo.toml"
  COMMENT "cargo build --release --locked --offline (Clarabel.rs ${CLARABEL_RS_VERSION})"
  VERBATIM)
add_custom_target(clarabel_cargo DEPENDS "${CLARABEL_STATIC_LIB}")
add_library(clarabel_static STATIC IMPORTED GLOBAL)
set_target_properties(clarabel_static PROPERTIES
  IMPORTED_LOCATION "${CLARABEL_STATIC_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${CLARABEL_CPP_DIR}/include"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
add_dependencies(clarabel_static clarabel_cargo)

# The flight library.
add_library(llg_flight STATIC
  src/socp.cpp
  src/roundtrip.cpp
  src/build_info.cpp
  src/nav_filter.cpp
  src/trn.cpp
  src/guidance.cpp
  src/triggers.cpp
  src/ode.cpp
  src/lsq.cpp
  src/abort.cpp
  src/descent.cpp)
target_include_directories(llg_flight PUBLIC include)
target_compile_options(llg_flight PRIVATE ${LLG_FLIGHT_FLAGS})
target_compile_definitions(llg_flight PRIVATE
  LLG_CLARABEL_RS_VERSION="${CLARABEL_RS_VERSION}"
  LLG_CARGO_LOCK_SHA256="${lock_hash}"
  LLG_CXX_FLAGS="${CMAKE_CXX_FLAGS_RELEASE} ${LLG_FLIGHT_FLAGS}")
target_link_libraries(llg_flight PUBLIC Eigen3::Eigen clarabel_static)
set_target_properties(llg_flight PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The pybind11 module lunar_landing_gnc._flight. NO_EXTRAS keeps pybind11 from
# adding its own optimisation flags (link-time optimisation), so the flags
# above are the only ones.
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE pyb_rc)
if(NOT pyb_rc EQUAL 0)
  message(FATAL_ERROR "pybind11 not importable by ${Python_EXECUTABLE}")
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.12 REQUIRED CONFIG)
pybind11_add_module(_flight NO_EXTRAS python/module.cpp python/nav_module.cpp python/trn_module.cpp python/gnc_module.cpp
  python/num_module.cpp python/abort_module.cpp)
target_compile_options(_flight PRIVATE ${LLG_FLIGHT_FLAGS})
target_link_libraries(_flight PRIVATE llg_flight)
target_compile_definitions(_flight PRIVATE LLG_PYBIND11_VERSION="${pybind11_VERSION}")
set_target_properties(_flight PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${REPO_ROOT}/src/lunar_landing_gnc")

# GoogleTest unit tests against hand-computed cases.
find_package(GTest 1.14 REQUIRED CONFIG)
enable_testing()
add_executable(llg_flight_tests "${REPO_ROOT}/tests/cpp/test_skeleton.cpp" "${REPO_ROOT}/tests/cpp/test_nav_filter.cpp"
  "${REPO_ROOT}/tests/cpp/test_trn.cpp"
  "${REPO_ROOT}/tests/cpp/test_guidance.cpp"
  "${REPO_ROOT}/tests/cpp/test_triggers.cpp"
  "${REPO_ROOT}/tests/cpp/test_ode.cpp"
  "${REPO_ROOT}/tests/cpp/test_lsq.cpp"
  "${REPO_ROOT}/tests/cpp/test_abort.cpp"
  "${REPO_ROOT}/tests/cpp/test_descent.cpp")
target_compile_options(llg_flight_tests PRIVATE ${LLG_FLIGHT_FLAGS})
target_link_libraries(llg_flight_tests PRIVATE llg_flight GTest::gtest_main)
add_test(NAME llg_flight_tests COMMAND llg_flight_tests)

# Single-thread timing of the flight-step calls (PLAN.md phase 4 bar 4).
add_executable(llg_flight_bench bench/bench.cpp)
# examples/compare_n1.py bench-input writes the recorded flight it times.
target_compile_options(llg_flight_bench PRIVATE ${LLG_FLIGHT_FLAGS})
target_link_libraries(llg_flight_bench PRIVATE llg_flight)
