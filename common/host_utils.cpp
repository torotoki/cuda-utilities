#include <random>
#include <vector>

using namespace std;

class InputGenerator {

  public:
    InputGenerator(int seed = 42)
      : seed(seed), random_generator(seed) {
    }

    template <typename T>
    vector<T> generateRandomVector(size_t size) {
      std::uniform_int_distribution<> dist(0, 32767);
      vector<T> result(size);

      for (size_t i = 0; i < size; ++i) {
        result[i] = dist(random_generator);
      }
      return result;
    }

    template <typename T>
    vector<T> generateConstantVector(size_t size) {
      vector<T> result(size, 1);
      return result;
    }
    
    template <typename T>
    pair<vector<T*>, vector<T>> generateConstantMatrix(
        size_t num_rows, size_t num_cols
    ) {
      vector<T> data(num_rows * num_cols, 1);
      vector<T*> row_ptrs(num_rows);
      for (uint i = 0; i < num_rows; ++i) {
        row_ptrs[i] = &data[i * num_cols];
      }

      return {row_ptrs, data};
    }

    template <typename T>
    vector<T> generateSortedVector(
        size_t size,
        unsigned int min_step = 1,
        unsigned int max_step = 10
    ) {
      vector<T> result(size);
      std::uniform_int_distribution<> step_dist(min_step, max_step);

      T current = 0;
      T value_limit = std::numeric_limits<T>::max() - max_step - 10;
      for (size_t i = 0; i < size; ++i) {
        assert(current < value_limit);
        unsigned int step = step_dist(random_generator);
        current += step;
        result[i] = current;
      }

      return result;
    }

  private:
    int seed;
    std::mt19937 random_generator;
};

