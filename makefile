INCLUDE_DIR = ./
CXX = g++
CXXFLAGS = -I$(INCLUDE_DIR) -O2 -Wall -Wextra -std=c++17

.PHONY: all build run clean

all: build

build:
	$(CXX) $(CXXFLAGS) -c math/Matrix.cpp -o Matrix.o
	$(CXX) $(CXXFLAGS) -c math/Activations.cpp -o Activations.o
	$(CXX) $(CXXFLAGS) -c MLP.cpp -o MLP.o
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o
	$(CXX) $(CXXFLAGS) Matrix.o Activations.o MLP.o main.o -o main.exe

run: build
	./main.exe

clean:
	del /Q Matrix.o Activations.o MLP.o main.o main.exe 2>NUL || exit 0