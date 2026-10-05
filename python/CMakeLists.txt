set(TARGET _core)

# pybind11 would LTO-compile only this module, which crashes lld-link.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

pybind11_add_module(${TARGET} python_binding.cpp)

target_compile_definitions(${TARGET} PRIVATE
    -DJIZAI_VERSION=${JIZAI_VERSION}
)

target_link_libraries(${TARGET} PRIVATE
    jizai
)

if(MSVC)
    jizai_target_contents(${TARGET} ${JIZAI_DLLS})
endif()
