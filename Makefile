# compiler
CXX    = g++

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -O3
LOADLIBES  = -lCCfits -lcfitsio -lOpenCL

BIN =   main.o  bachUtil.o sssUtil.o cmvUtil.o cdkscUtil.o

all: $(BIN)
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -o BACH $(BIN)
	rm -f *.o

debug: override CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -g3
debug:	$(BIN)
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -o BACH $(BIN)


main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c main.cpp

bachUtil.o: bachUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c bachUtil.cpp

sssUtil.o: sssUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c sssUtil.cpp

cmvUtil.o: cmvUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c cmvUtil.cpp

cdkscUtil.o: cdkscUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c cdkscUtil.cpp

.PHONY: clean
clean:
	rm -f *.o BACH
