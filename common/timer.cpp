#include <vector>
#include <iostream>
#include <chrono>
#include <cassert>

using namespace std;

class Stopwatch {
  public:
    Stopwatch() {}

    void start() {
      assert(intervals.size() == 0 && !finished);
      intervals.push_back(chrono::system_clock::now());
    }

    void pause() {
      assert(intervals.size() % 2 == 1 && !finished);
      intervals.push_back(chrono::system_clock::now());
    }

    void resume() {
      assert(intervals.size() % 2 == 0 && !finished);
      intervals.push_back(chrono::system_clock::now());
    }

    void lap() {
      assert(intervals.size() % 2 == 1 && !finished);
      chrono::system_clock::time_point now = chrono::system_clock::now();
      intervals.push_back(now);
      intervals.push_back(now);
    }

    void stop() {
      pause();
      finished = true;
    }

    double get_elapsed_time_msec() {
      assert(finished);
      return chrono::duration<double, std::milli>
        (intervals.back() - intervals.front()).count();
    }

    void pprint() {
      // Incomplete intervals are not supported yet.
      assert(intervals.size() % 2 == 0);
      double elapsed_time_msec =
        chrono::duration<double, std::milli>
        (intervals.back() - intervals.front()).count();
      cout << "Total Time: " << elapsed_time_msec << " msec" << endl;
      
      if (intervals.size() == 2)
        return;

      vector<double> time_elapsed_msecs;
      for (int i = 0; i < intervals.size(); i += 2) {
        chrono::system_clock::time_point start, end;
        double msec;
        start = intervals[i];
        end = intervals[i + 1];
        msec = chrono::duration<double, std::milli>(end - start).count();
        time_elapsed_msecs.push_back(msec);
      }

      for (int i = 0; i < time_elapsed_msecs.size(); ++i) {
        cout << "Time("<< i <<"): " << time_elapsed_msecs[i] << " msec" << endl;
      }
    }

  private:
    vector<chrono::system_clock::time_point> intervals;
    bool finished = false;
};

