#pragma once

#define STRINGIFY_DETAIL(x) #x
#define RAFT_STRINGIFY(x) STRINGIFY_DETAIL(x)


#define RAFT_EXPECT(cond, reason)

#define RAFT_TRY(error_expression)

#define RAFT_FAIL(reason)

#define CUDA_TRY(call)

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

