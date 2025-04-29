TARGET = main

SRC = main.cu gpu_histogram_v1.cu gpu_histogram_v2.cu

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^

clean:
	rm $(TARGET)

