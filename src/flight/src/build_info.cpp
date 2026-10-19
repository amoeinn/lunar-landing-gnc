#include "llg/roundtrip.hpp"

#include <Eigen/Core>

#include <string>

namespace llg {

std::map<std::string, std::string> build_info() {
    return {
        {"clarabel_rs", LLG_CLARABEL_RS_VERSION},
        {"cargo_lock_sha256", LLG_CARGO_LOCK_SHA256},
        {"eigen", std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
                      std::to_string(EIGEN_MINOR_VERSION)},
        {"compiler", __VERSION__},
        {"cxx_standard", std::to_string(__cplusplus)},
        {"flags", LLG_CXX_FLAGS},
#ifdef __FP_FAST_FMA
        {"fp_fast_fma", "defined"},
#else
        {"fp_fast_fma", "not defined"},
#endif
#ifdef __FAST_MATH__
        {"fast_math", "on"},
#else
        {"fast_math", "off"},
#endif
    };
}

}  // namespace llg
