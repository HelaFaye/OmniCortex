# Drop this directory into llama.cpp/tools/cett-dump/ and add
#   add_subdirectory(cett-dump)
# to llama.cpp/tools/CMakeLists.txt

set(TARGET llama-cett-dump)
add_executable(${TARGET} cett-dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
# The common library was renamed: `common` on older revisions, `llama-common`
# since llama.cpp 0.4.0-dev. Link whichever exists.
if (TARGET llama-common)
    set(OC_COMMON llama-common)
else()
    set(OC_COMMON common)
endif()
target_link_libraries(${TARGET} PRIVATE ${OC_COMMON} llama ${CMAKE_THREAD_LIBS_INIT})
# Some llama.cpp revisions do not export common/ from the `common` target, so
# common.h fails to resolve. Naming it explicitly is harmless when it does.
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
