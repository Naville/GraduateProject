# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.12.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.12.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/naville/Desktop/L4Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/naville/Desktop/L4Project/GPU

# Include any dependencies generated for this target.
include CMakeFiles/FT.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FT.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FT.dir/flags.make

CMakeFiles/FT.dir/FT/ft.cpp.o: CMakeFiles/FT.dir/flags.make
CMakeFiles/FT.dir/FT/ft.cpp.o: ../FT/ft.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/naville/Desktop/L4Project/GPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FT.dir/FT/ft.cpp.o"
	/Users/naville/Downloads/LLVM7/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FT.dir/FT/ft.cpp.o -c /Users/naville/Desktop/L4Project/FT/ft.cpp

CMakeFiles/FT.dir/FT/ft.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FT.dir/FT/ft.cpp.i"
	/Users/naville/Downloads/LLVM7/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/naville/Desktop/L4Project/FT/ft.cpp > CMakeFiles/FT.dir/FT/ft.cpp.i

CMakeFiles/FT.dir/FT/ft.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FT.dir/FT/ft.cpp.s"
	/Users/naville/Downloads/LLVM7/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/naville/Desktop/L4Project/FT/ft.cpp -o CMakeFiles/FT.dir/FT/ft.cpp.s

# Object files for target FT
FT_OBJECTS = \
"CMakeFiles/FT.dir/FT/ft.cpp.o"

# External object files for target FT
FT_EXTERNAL_OBJECTS =

FT: CMakeFiles/FT.dir/FT/ft.cpp.o
FT: CMakeFiles/FT.dir/build.make
FT: libCore.a
FT: CMakeFiles/FT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/naville/Desktop/L4Project/GPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FT"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FT.dir/build: FT

.PHONY : CMakeFiles/FT.dir/build

CMakeFiles/FT.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FT.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FT.dir/clean

CMakeFiles/FT.dir/depend:
	cd /Users/naville/Desktop/L4Project/GPU && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/naville/Desktop/L4Project /Users/naville/Desktop/L4Project /Users/naville/Desktop/L4Project/GPU /Users/naville/Desktop/L4Project/GPU /Users/naville/Desktop/L4Project/GPU/CMakeFiles/FT.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FT.dir/depend

