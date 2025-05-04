TARGET = main

SRC = main.cpp gpu_scan_v1.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

