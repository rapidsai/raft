/**
 * @brief Exception thrown when a NCCL error is encountered.
 */
struct nccl_error : public raft::exception {
  explicit nccl_error(char const *const message) : raft::exception(message) {}
  explicit nccl_error(std::string const &message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for NCCL runtime API functions.
 *
 * Invokes a NCCL runtime API function call, if the call does not return ncclSuccess, throws an
 * exception detailing the NCCL error that occurred
 */
#define NCCL_TRY(call)                                                        \
  do {                                                                        \
    ncclResult_t const status = (call);                                       \
    if (ncclSuccess != status) {                                              \
      std::string msg{};                                                      \
      SET_ERROR_MSG(msg,                                                      \
                    "NCCL error encountered at: ", "call='%s', Reason=%d:%s", \
                    #call, status, ncclGetErrorString(status));               \
      throw raft::nccl_error(msg);                                            \
    }                                                                         \
  } while (0);

#define NCCL_CHECK_NO_THROW(call)                         \
  do {                                                    \
    ncclResult_t status = call;                           \
    if (ncclSuccess != status) {                          \
      printf("NCCL call='%s' failed. Reason:%s\n", #call, \
             ncclGetErrorString(status));                 \
    }                                                     \
  } while (0)

namespace raft {
namespace comms {

static size_t get_datatype_size(const datatype_t datatype) {
  switch (datatype) {
    case datatype_t::CHAR:
      return sizeof(char);
    case datatype_t::UINT8:
      return sizeof(uint8_t);
    case datatype_t::INT32:
      return sizeof(int);
    case datatype_t::UINT32:
      return sizeof(unsigned int);
    case datatype_t::INT64:
      return sizeof(int64_t);
    case datatype_t::UINT64:
      return sizeof(uint64_t);
    case datatype_t::FLOAT32:
      return sizeof(float);
    case datatype_t::FLOAT64:
      return sizeof(double);
    default:
      RAFT_FAIL("Unsupported datatype.");
  }
}

static ncclDataType_t get_nccl_datatype(const datatype_t datatype) {
  switch (datatype) {
    case datatype_t::CHAR:
      return ncclChar;
    case datatype_t::UINT8:
      return ncclUint8;
    case datatype_t::INT32:
      return ncclInt;
    case datatype_t::UINT32:
      return ncclUint32;
    case datatype_t::INT64:
      return ncclInt64;
    case datatype_t::UINT64:
      return ncclUint64;
    case datatype_t::FLOAT32:
      return ncclFloat;
    case datatype_t::FLOAT64:
      return ncclDouble;
    default:
      throw "Unsupported";
  }
}

static ncclRedOp_t get_nccl_op(const op_t op) {
  switch (op) {
    case op_t::SUM:
      return ncclSum;
    case op_t::PROD:
      return ncclProd;
    case op_t::MIN:
      return ncclMin;
    case op_t::MAX:
      return ncclMax;
    default:
      throw "Unsupported";
  }
}
