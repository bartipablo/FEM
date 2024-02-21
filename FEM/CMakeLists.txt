cmake_minimum_required(VERSION 3.27)
project(example)

# Ustaw standard C++.
set(CMAKE_CXX_STANDARD 17)

# Znajdź biblioteki Pythona.
find_package(Python COMPONENTS Development REQUIRED)

# Dodaj plik wykonywalny projektu.
add_executable(example main.cpp)

# Dołącz katalogi nagłówków Pythona.
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})

# Linkuj biblioteki Pythona z twoim plikiem wykonywalnym.
target_link_libraries(example Python::Python)
