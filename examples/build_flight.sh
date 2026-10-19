#!/usr/bin/env bash
# Build the phase 4 flight code (src/flight): Clarabel.rs offline against the
# recorded Cargo.lock, the llg_flight library, the pybind11 module
# lunar_landing_gnc._flight, the GoogleTest unit tests and the timing bench.
# Release, C++17, -O2 -ffp-contract=off. Nothing is downloaded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/work/build/flight"
export PATH="${HOME}/.cargo/bin:${PATH}"
cmake -S "${ROOT}/src/flight" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="${ROOT}/.venv/bin/python"
cmake --build "${BUILD}" -j "${JOBS:-8}"
"${BUILD}/llg_flight_tests"
