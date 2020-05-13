#include <nccl.h>

namespace raft {
namespace comms {
inline void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId, int size) {
  memcpy(id->internal, uniqueId, size);
}

inline void get_unique_id(char *uid, int size) {
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, size);
}
}  // namespace comms
}  // namespace raft
