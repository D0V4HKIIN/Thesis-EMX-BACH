# compiler
CXX    = g++

SRC_DIR = src
INCLUDE_DIR = include

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -I$(INCLUDE_DIR) -DCL_HPP_TARGET_OPENCL_VERSION=300 -DCL_TARGET_OPENCL_VERSION=300

RELEASEFLAGS = -O3
DEBUGFLAGS = -g3

FLAGS = $(CXXFLAGS) $(RELEASEFLAGS)

LOADLIBES  = -lOpenCL -lCCfits -lcfitsio -fopenmp

PROFILERLIBES = -Wl,--no-as-needed,-lprofiler,--as-needed

BIN = main.o argsUtil.o bach.o bachUtil.o cdkscUtil.o clUtil.o cmvUtil.o fitsUtil.o sssUtil.o

all: $(BIN)
	$(CXX) $(FLAGS) -o EMXBACH $(BIN) $(LOADLIBES)
	cp EMXBACH bin/emxbach/emxbach
	# rm -f *.o

# requires google perftools (libgoogle-perftools-dev in ubuntu packages)
# https://gperftools.github.io/gperftools/cpuprofile.html
profile: override LOADLIBES += $(PROFILERLIBES)
profile: all

debugprofile: override LOADLIBES += $(PROFILERLIBES)
debugprofile: override FLAGS = $(CXXFLAGS) $(DEBUGFLAGS)
debugprofile: all

debug: override FLAGS = $(CXXFLAGS) $(DEBUGFLAGS)
debug: all

SHELL := /bin/bash
test:
	cp EMXBACH bin/emxbach/emxbach
	source ./tools/venv/bin/activate &&\
	python ./tools/run_test.py

run_profiler: profile
	CPUPROFILE=emxbach.prof CPUPROFILE_FREQUENCY=1000 ./EMXBACH -t ptf_m82_t_2k.fits -s ptf_m82_s_2k.fits -vt -sss mp && pprof --http=:9999 --focus conv ./EMXBACH emxbach.prof

main.o: $(SRC_DIR)/main.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/main.cpp
	
argsUtil.o: $(SRC_DIR)/argsUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/argsUtil.cpp

bach.o: $(SRC_DIR)/bach.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/bach.cpp

bachUtil.o: $(SRC_DIR)/bachUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/bachUtil.cpp

cdkscUtil.o: $(SRC_DIR)/cdkscUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/cdkscUtil.cpp

clUtil.o: $(SRC_DIR)/clUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/clUtil.cpp

cmvUtil.o: $(SRC_DIR)/cmvUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/cmvUtil.cpp

fitsUtil.o: $(SRC_DIR)/fitsUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/fitsUtil.cpp

sssUtil.o: $(SRC_DIR)/sssUtil.cpp
	$(CXX) $(FLAGS) $(LOADLIBES) -c $(SRC_DIR)/sssUtil.cpp

.PHONY: clean
clean:
	rm -f *.o EMXBACH
