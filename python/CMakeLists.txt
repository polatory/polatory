set(TARGET _core)

pybind11_add_module(${TARGET} python_binding.cpp)

set(STRUCTURAL_TARGET _structural)
pybind11_add_module(${STRUCTURAL_TARGET} structural_binding.cpp)

foreach(PY_TARGET ${TARGET} ${STRUCTURAL_TARGET})
    target_compile_definitions(${PY_TARGET} PRIVATE
        -DPOLATORY_VERSION=${POLATORY_VERSION}
    )

    target_link_libraries(${PY_TARGET} PRIVATE
        polatory
    )

    if(MSVC)
        polatory_target_contents(${PY_TARGET} ${POLATORY_DLLS})
    endif()
endforeach()

# setup.py builds the _core target explicitly. Make that build the companion
# structural extension into the same Python package output directory as well.
add_dependencies(${TARGET} ${STRUCTURAL_TARGET})
