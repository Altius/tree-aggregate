CC	= g++
FLAGS	= -static -ansi -Wall -std=c++11 -pedantic -O3 -s
DFLAGS	= -static -ansi -Wall -std=c++11 -pedantic -O0 -g -s
SRCDIR	= src
BIN	= bin

SOURCE1 = random_forest.cpp

.SUFFIXES: .cpp .o

.cpp.o:; $(CC) -c $(FLAGS) $<

prog:
	mkdir -p $(BIN)
	$(CC) -o $(BIN)/rf $(FLAGS) $(SRCDIR)/$(SOURCE1)

debug:
	mkdir -p $(BIN)
	$(CC) -o $(BIN)/debug.rf $(FLAGS) $(SRCDIR)/$(SOURCE1)

clean:
	rm -rf $(BIN)/rf
	rm -rf $(BIN)/debug.rf
