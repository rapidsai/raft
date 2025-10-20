
include(metabench)

function(add_cxx_comparison name template range)

    set(all_datasets)
    foreach(std IN ITEMS 11 14 17)
      metabench_add_dataset(
          ${name}_${std}
          ${template}
          ${range}
          MEDIAN_OF 3
      )
      target_link_libraries(${name}_${std} mdspan)
      set_property(TARGET ${name}_${std} PROPERTY CXX_STANDARD ${std})
      set(all_datasets ${all_datasets} ${name}_${std})
    endforeach()

    metabench_add_chart(${name} DATASETS ${all_datasets})
endfunction()

add_cxx_comparison(submdspan_chart "cbench_submdspan.cpp.erb" "[2, 4, 8, 16, 32]")
