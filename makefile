INCLUDE_DIR = ./
CXX = g++
CXXFLAGS = -I$(INCLUDE_DIR) -O3 -Wall -Wextra -std=c++17

.PHONY: all build run test benchmark clean

all: build

build:
	$(CXX) $(CXXFLAGS) -c math/Tensor.cpp -o Tensor.o
	$(CXX) $(CXXFLAGS) -c math/Activations.cpp -o Activations.o
	$(CXX) $(CXXFLAGS) -c MLP.cpp -o MLP.o
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o
	$(CXX) $(CXXFLAGS) Tensor.o Activations.o MLP.o main.o -o main.exe

run: build
	./main.exe

test:
	$(CXX) $(CXXFLAGS) math/Tensor.cpp test/TensorTests.cpp -o TensorTests.exe
	./TensorTests.exe

clean:
	del /Q loss.csv Matrix.o Tensor.o Activations.o MLP.o main.o main.exe TensorTests.exe 2>NUL || exit 0