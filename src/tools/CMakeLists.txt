# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(mtmd)
    add_subdirectory(gguf-split)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
    if (GGML_RPC)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
    endif()
    if (GGML_METAL)
    endif()
endif()
