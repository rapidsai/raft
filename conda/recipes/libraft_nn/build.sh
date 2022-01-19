if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    # C++11 requires 10.9
    # but cudatoolkit 8 is build for 10.11
    export MACOSX_DEPLOYMENT_TARGET=10.11
fi

./build.sh clean libraft -v --allgpuarch --compilelibs

# TODO: Somehow extract just the libraft_nn.so artiface