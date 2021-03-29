set(CPM_DOWNLOAD_VERSION 0.31.1)

if(CPM_SOURCE_CACHE)
  # Expand relative path. This is important if the provided path contains a tilde (~)
  get_filename_component(CPM_SOURCE_CACHE ${CPM_SOURCE_CACHE} ABSOLUTE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "RAFT: Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# If a target is installed, found by the `find_package` step of CPMFindPackage,
# and marked as IMPORTED, make it globally accessible to consumers of our libs.
function(fix_cmake_global_defaults target)
    if(TARGET ${target})
        get_target_property(_is_imported ${target} IMPORTED)
        get_target_property(_already_global ${target} IMPORTED_GLOBAL)
        if(_is_imported AND NOT _already_global)
            set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endfunction()

function(raft_save_if_enabled var)
    if(RAFT${var})
        unset(${var} PARENT_SCOPE)
        unset(${var} CACHE)
    endif()
endfunction()

function(raft_restore_if_enabled var)
    if(RAFT${var})
        set(${var} ON CACHE INTERNAL "" FORCE)
    endif()
endfunction()

