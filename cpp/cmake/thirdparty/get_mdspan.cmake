function(find_and_configure_mdspan VERSION)
  rapids_cpm_find(
    mdspan ${VERSION}
    GLOBAL_TARGETS std::mdspan
    BUILD_EXPORT_SET    raft-exports
    INSTALL_EXPORT_SET  raft-exports
    CPM_ARGS
      GIT_REPOSITORY https://github.com/trivialfis/mdspan
      GIT_TAG 0193f075e977cc5f3c957425fd899e53d598f524
      OPTIONS "MDSPAN_ENABLE_CUDA ON"
              "MDSPAN_CXX_STANDARD ON"
  )
endfunction()

find_and_configure_mdspan(0.2.0)
