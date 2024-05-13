# compiler
CXX    = g++

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -O3
LOADLIBES  = -lCCfits -lcfitsio -lOpenCL

BIN = main.o argsUtil.o bach.o bachUtil.o cdkscUtil.o clUtil.o cmvUtil.o fitsUtil.o sssUtil.o

all: $(BIN)
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -o BACH $(BIN)
	rm -f *.o

debug: override CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon -g3
debug:	$(BIN)
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -o BACH $(BIN)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c main.cpp
	
argsUtil.o: argsUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c argsUtil.cpp

bach.o: bach.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c bach.cpp

bachUtil.o: bachUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c bachUtil.cpp

cdkscUtil.o: cdkscUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c cdkscUtil.cpp

clUtil.o: clUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c clUtil.cpp

cmvUtil.o: cmvUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c cmvUtil.cpp

fitsUtil.o: fitsUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c fitsUtil.cpp

sssUtil.o: sssUtil.cpp
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -c sssUtil.cpp

.PHONY: clean
clean:
	rm -f *.o BACH
