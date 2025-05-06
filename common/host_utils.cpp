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
      vector<T> vec(size);

      for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(random_generator);
      }
      return vec;
    }

    template <typename T>
    vector<T> generateConstantVector(size_t size) {
      vector<T> vec(size);
      for (size_t i = 0; i < size; ++i) {
        vec[i] = 1;
      }
      return vec;
    }

  private:
    int seed;
    std::mt19937 random_generator;
};

