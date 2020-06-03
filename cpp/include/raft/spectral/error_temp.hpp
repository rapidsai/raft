#pragma once

#define STRINGIFY_DETAIL(x) #x
#define RAFT_STRINGIFY(x) STRINGIFY_DETAIL(x)


#define RAFT_EXPECT(cond, reason)

#define RAFT_TRY(error_expression)

#define RAFT_FAIL(reason)

#define CUDA_TRY(call)

#define CUDA_CHECK_LAST()
