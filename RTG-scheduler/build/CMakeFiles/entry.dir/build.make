# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build

# Include any dependencies generated for this target.
include CMakeFiles/entry.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/entry.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/entry.dir/flags.make

CMakeFiles/entry.dir/source/RTGS-entry.c.o: CMakeFiles/entry.dir/flags.make
CMakeFiles/entry.dir/source/RTGS-entry.c.o: ../source/RTGS-entry.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/entry.dir/source/RTGS-entry.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/entry.dir/source/RTGS-entry.c.o   -c /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/source/RTGS-entry.c

CMakeFiles/entry.dir/source/RTGS-entry.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/entry.dir/source/RTGS-entry.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/source/RTGS-entry.c > CMakeFiles/entry.dir/source/RTGS-entry.c.i

CMakeFiles/entry.dir/source/RTGS-entry.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/entry.dir/source/RTGS-entry.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/source/RTGS-entry.c -o CMakeFiles/entry.dir/source/RTGS-entry.c.s

CMakeFiles/entry.dir/source/RTGS-entry.c.o.requires:

.PHONY : CMakeFiles/entry.dir/source/RTGS-entry.c.o.requires

CMakeFiles/entry.dir/source/RTGS-entry.c.o.provides: CMakeFiles/entry.dir/source/RTGS-entry.c.o.requires
	$(MAKE) -f CMakeFiles/entry.dir/build.make CMakeFiles/entry.dir/source/RTGS-entry.c.o.provides.build
.PHONY : CMakeFiles/entry.dir/source/RTGS-entry.c.o.provides

CMakeFiles/entry.dir/source/RTGS-entry.c.o.provides.build: CMakeFiles/entry.dir/source/RTGS-entry.c.o


# Object files for target entry
entry_OBJECTS = \
"CMakeFiles/entry.dir/source/RTGS-entry.c.o"

# External object files for target entry
entry_EXTERNAL_OBJECTS =

bin/entry: CMakeFiles/entry.dir/source/RTGS-entry.c.o
bin/entry: CMakeFiles/entry.dir/build.make
bin/entry: CMakeFiles/entry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bin/entry"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/entry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/entry.dir/build: bin/entry

.PHONY : CMakeFiles/entry.dir/build

CMakeFiles/entry.dir/requires: CMakeFiles/entry.dir/source/RTGS-entry.c.o.requires

.PHONY : CMakeFiles/entry.dir/requires

CMakeFiles/entry.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/entry.dir/cmake_clean.cmake
.PHONY : CMakeFiles/entry.dir/clean

CMakeFiles/entry.dir/depend:
	cd /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build /home/zephyr/Programming/CS6235/SRTG-Schedule/RTG-scheduler/build/CMakeFiles/entry.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/entry.dir/depend

