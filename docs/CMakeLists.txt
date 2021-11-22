find_package(Doxygen)

if(DOXYGEN_FOUND STREQUAL "NO")
    message(STATUS "Doxygen not found. Please get a copy http://www.doxygen.org")
else()

    if("${DOXYGEN_VERSION}" VERSION_LESS 1.8.3.1)
        message(
                STATUS "Doxygen version too old. Please get a copy http://www.doxygen.org"
        )
    endif()

    # prepare doxygen configuration file
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )

endif()
