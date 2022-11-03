#pragma once
#include <exception>

namespace raft {
#ifdef RAFT_ENABLE_CUDA
auto constexpr static const CUDA_ENABLED = true;
#else
auto constexpr static const CUDA_ENABLED = false;
#endif

struct cuda_unsupported : std::exception {
  cuda_unsupported() : cuda_unsupported("CUDA functionality invoked in non-CUDA build") {}
  explicit cuda_unsupported(char const* msg) : msg_{msg} {}
  [[nodiscard]] virtual auto what() const noexcept -> char const* { return msg_; }

 private:
  char const* msg_;
};

}  // namespace raft
