# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# If libucx was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libucx
except ModuleNotFoundError:
    pass
else:
    libucx.load_library()
    del libucx

# If libraft was installed as a wheel, we must request it to load the library
# symbols. Otherwise, we assume that the library was installed in a system path that ld
# can find.
try:
    import libraft
except ModuleNotFoundError:
    pass
else:
    libraft.load_library()
    del libraft

from raft_dask._version import __git_commit__, __version__
