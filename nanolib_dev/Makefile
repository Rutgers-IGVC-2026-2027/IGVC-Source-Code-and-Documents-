CXX := /usr/bin/g++

NANOLIB_SDK ?= $(HOME)/nanolib/NanoLib_cpp_1.4.3/nanolib_cpp_linux_1.4.3/nanotec_nanolib
NANOLIB_INCLUDE ?= $(NANOLIB_SDK)/inc
NANOLIB_LIB ?= /usr/local/lib/nanotec

CPPFLAGS := -I$(NANOLIB_INCLUDE)
CXXFLAGS := -std=c++17 -O0 -g3 -Wall -Wextra -pthread
LDFLAGS := -L$(NANOLIB_LIB) -Wl,-rpath,$(NANOLIB_LIB)
LDLIBS := -lnanolib -ldl -lstdc++fs -pthread

TARGETS := encoder_test sensorless_velocity_test

.PHONY: all clean check

all: check $(TARGETS)

check:
	@test -f "$(NANOLIB_INCLUDE)/accessor_factory.hpp" || (echo "NanoLib headers not found at $(NANOLIB_INCLUDE)"; echo "Edit NANOLIB_SDK in this Makefile if your SDK is elsewhere."; exit 1)
	@test -f "$(NANOLIB_LIB)/libnanolib.so" || (echo "libnanolib.so not found at $(NANOLIB_LIB)"; exit 1)

$(TARGETS): %: %.cpp | check
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)

motoManagerTest: motoManagerTest.cpp motorManager.cpp motorManager.hpp
	/usr/bin/g++ -std=c++17 -O0 -g3 -Wall -Wextra -pthread \
	-I/home/igvc/nanolib/NanoLib_cpp_1.4.3/nanolib_cpp_linux_1.4.3/nanotec_nanolib/inc \
	motorManager.cpp motoManagerTest.cpp \
	-L/usr/local/lib/nanotec \
	-Wl,-rpath,/usr/local/lib/nanotec \
	-lnanolib -ldl -lstdc++fs -pthread \
	-o motoManagerTest
