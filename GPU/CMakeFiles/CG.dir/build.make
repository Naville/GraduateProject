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
include CMakeFiles/CG.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CG.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CG.dir/flags.make

CMakeFiles/CG.dir/CG/main.cpp.o: CMakeFiles/CG.dir/flags.make
CMakeFiles/CG.dir/CG/main.cpp.o: ../CG/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/naville/Desktop/L4Project/GPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CG.dir/CG/main.cpp.o"
	/Users/naville/Downloads/LLVM7/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CG.dir/CG/main.cpp.o -c /Users/naville/Desktop/L4Project/CG/main.cpp

CMakeFiles/CG.dir/CG/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CG.dir/CG/main.cpp.i"
	/Users/naville/Downloads/LLVM7/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/naville/Desktop/L4Project/CG/main.cpp > CMakeFiles/CG.dir/CG/main.cpp.i

CMakeFiles/CG.dir/CG/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CG.dir/CG/main.cpp.s"
	/Users/naville/Downloads/LLVM7/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/naville/Desktop/L4Project/CG/main.cpp -o CMakeFiles/CG.dir/CG/main.cpp.s

# Object files for target CG
CG_OBJECTS = \
"CMakeFiles/CG.dir/CG/main.cpp.o"

# External object files for target CG
CG_EXTERNAL_OBJECTS =

CG: CMakeFiles/CG.dir/CG/main.cpp.o
CG: CMakeFiles/CG.dir/build.make
CG: libCore.a
CG: CMakeFiles/CG.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/naville/Desktop/L4Project/GPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CG"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CG.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CG.dir/build: CG

.PHONY : CMakeFiles/CG.dir/build

CMakeFiles/CG.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CG.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CG.dir/clean

CMakeFiles/CG.dir/depend:
	cd /Users/naville/Desktop/L4Project/GPU && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/naville/Desktop/L4Project /Users/naville/Desktop/L4Project /Users/naville/Desktop/L4Project/GPU /Users/naville/Desktop/L4Project/GPU /Users/naville/Desktop/L4Project/GPU/CMakeFiles/CG.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CG.dir/depend

