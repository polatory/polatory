set(TARGET _core)

# pybind11 would LTO-compile only this module, which crashes lld-link.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

pybind11_add_module(${TARGET} python_binding.cpp)

target_compile_definitions(${TARGET} PRIVATE
    -DPOLATORY_VERSION=${POLATORY_VERSION}
)

target_link_libraries(${TARGET} PRIVATE
    polatory
)

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()
