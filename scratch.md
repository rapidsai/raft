```shell
docker run \
    --rm \
    -v $(pwd):/opt/work \
    -w /opt/work \
    -it rapidsai/ci-conda:cuda12.2.2-ubuntu22.04-py3.9-amd64 \
    bash

# do a 'librmm' conda build
ci/build_cpp.sh
```
