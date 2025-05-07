TARGET = main

SRC = main.cpp reduction.cpp gpu_reduction.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

