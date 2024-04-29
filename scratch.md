```shell
docker run \
    --rm \
    -v $(pwd):/opt/work \
    -w /opt/work \
    -it rapidsai/ci-conda:cuda12.2.2-ubuntu22.04-py3.10 \
    bash

# do a 'libraft' conda build
ci/build_cpp.sh
```

* https://github.com/mozilla/sccache/pull/1403
* https://github.com/mozilla/sccache/blob/887ae9086146d1cfdbdd0a3bc568c77c80ac3a08/src/config.rs#L659-L664
* https://github.com/rapidsai/gha-tools/pull/57/files#r1268405346

Supposedly:

```text
/opt/conda/lib/python3.10/site-packages/conda_build/environ.py:551: UserWarning: The environment variable 'AWS_ACCESS_KEY_ID' specified in script_env is undefined.
  warnings.warn(
```

> If a listed environment variable is missing from the environment seen by the conda-build process itself, a UserWarning is emitted during the build process and the variable remains undefined.

https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html#use-environment-variables

that's coming from here: https://github.com/conda/conda-build/blob/3e9fe719aecc61c92cdfc20fe131cde1c36e3a3c/conda_build/environ.py#L532
