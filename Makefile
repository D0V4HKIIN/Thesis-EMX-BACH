# compiler
CXX    = g++

SRC_DIR = src
INCLUDE_DIR = include

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -O3 -I$(INCLUDE_DIR)
LOADLIBES  = -lOpenCL -lCCfits -lcfitsio

BIN = main.o argsUtil.o bach.o bachUtil.o cdkscUtil.o clUtil.o cmvUtil.o fitsUtil.o sssUtil.o

all: $(BIN)
	$(CXX) $(CXXFLAGS) -o BACH $(BIN) $(LOADLIBES)
	rm -f *.o

debug: override CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -g3
debug:	$(BIN)
	$(CXX) $(CXXFLAGS) -o BACH $(BIN) $(LOADLIBES)

main.o: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/main.cpp
	
argsUtil.o: $(SRC_DIR)/argsUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/argsUtil.cpp

bach.o: $(SRC_DIR)/bach.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/bach.cpp

bachUtil.o: $(SRC_DIR)/bachUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/bachUtil.cpp

cdkscUtil.o: $(SRC_DIR)/cdkscUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/cdkscUtil.cpp

clUtil.o: $(SRC_DIR)/clUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/clUtil.cpp

cmvUtil.o: $(SRC_DIR)/cmvUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/cmvUtil.cpp

fitsUtil.o: $(SRC_DIR)/fitsUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/fitsUtil.cpp

sssUtil.o: $(SRC_DIR)/sssUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c $(SRC_DIR)/sssUtil.cpp

.PHONY: clean
clean:
	rm -f *.o BACH
