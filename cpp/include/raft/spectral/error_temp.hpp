#pragma once

#include <stdexcept>
#include <string>

#define STRINGIFY_DETAIL(x) #x
#define RAFT_STRINGIFY(x) STRINGIFY_DETAIL(x)

///#define RAFT_EXPECT(cond, reason)
inline void RAFT_EXPECT(bool cond, std::string const& reason) {
  if (!cond) throw std::runtime_error(reason.c_str());
}

#define RAFT_TRY(expression) (expression)

//assume RAFT_FAIL() can take a std::string `reason`
//
#define RAFT_FAIL(reason)

#define CUDA_TRY(call) (call)

#define CUDA_CHECK_LAST()

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)
#define WARNING(message)                                                  \
  do {                                                                    \
    std::stringstream ss;                                                 \
    ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message; \
    CERR() << ss.str() << std::endl;                                      \
  } while (0)
#else  // DEBUG
#define WARNING(message)
#endif
