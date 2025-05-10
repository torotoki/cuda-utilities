TARGET = main

SRC = main.cu
INCLUDE = ../common/

NVCC = nvcc

$(TARGET): $(SRC)
	$(NVCC) -o $@ -I$(INCLUDE) $^

clean:
	rm $(TARGET)

