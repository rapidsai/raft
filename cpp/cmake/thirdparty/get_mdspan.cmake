function(find_and_configure_mdspan VERSION)
  rapids_cpm_find(
    mdspan ${VERSION}
    GLOBAL_TARGETS std::mdspan
    BUILD_EXPORT_SET    raft-exports
    INSTALL_EXPORT_SET  raft-exports
    CPM_ARGS
      EXCLUDE_FROM_ALL TRUE
      GIT_REPOSITORY https://github.com/rapidsai/mdspan.git
      GIT_TAG b3042485358d2ee168ae2b486c98c2c61ec5aec1
      OPTIONS "MDSPAN_ENABLE_CUDA ON"
              "MDSPAN_CXX_STANDARD ON"
  )
endfunction()

find_and_configure_mdspan(0.2.0)
