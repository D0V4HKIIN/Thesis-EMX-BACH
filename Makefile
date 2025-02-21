# compiler
CXX    = g++

SRC_DIR = src
INCLUDE_DIR = include

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -I$(INCLUDE_DIR) -DCL_HPP_TARGET_OPENCL_VERSION=300 -DCL_TARGET_OPENCL_VERSION=300

RELEASEFLAGS = -O3
DEBUGFLAGS = -g3

FLAGS = $(CXXFLAGS) $(RELEASEFLAGS)

LOADLIBES  = -lOpenCL -lCCfits -lcfitsio -fopenmp

BIN = main.o argsUtil.o bach.o bachUtil.o cdkscUtil.o clUtil.o cmvUtil.o fitsUtil.o sssUtil.o

all: $(BIN)
	$(CXX) $(FLAGS) -o BACH $(BIN) $(LOADLIBES)
	rm -f *.o

debug: override FLAGS = $(CXXFLAGS) $(DEBUGFLAGS)
debug:	$(BIN)
	$(CXX) $(FLAGS) $(DEBUGFLAGS) -o BACH $(BIN) $(LOADLIBES)
	rm -f *.o

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
	rm -f *.o BACH
