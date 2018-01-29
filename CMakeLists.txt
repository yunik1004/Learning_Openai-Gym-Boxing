cmake_minimum_required(VERSION 3.5)

find_program(PYTHON "python")

if(PYTHON)
    set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()