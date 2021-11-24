#pragma once

#include <stdexcept>
#include <string>

#define STRINGIFY_DETAIL(x) #x
#define RAFT_STRINGIFY(x) STRINGIFY_DETAIL(x)

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)

//nope:
//
#define WARNING(message)                                                  \
  do {                                                                    \
    std::stringstream ss;                                                 \
    ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message; \
    CERR() << ss.str() << std::endl;                                      \
  } while (0)
#else  // DEBUG
#define WARNING(message)
#endif
