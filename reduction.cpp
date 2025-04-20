#include <cstddef>

using namespace std;

template<typename T>
T reduce(const T* values, size_t size) {
  T sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    sum += values[i];
  }
  return sum;
}

template int reduce<int>(const int*, size_t);
//template float reduce<float>(const float*, size_t);
