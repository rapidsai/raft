# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

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

from pylibraft._version import __git_commit__, __version__
