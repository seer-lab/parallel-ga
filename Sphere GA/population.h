#include <stdlib.h>
#include <vector>

void initPopulation (std::vector<std::vector<double>> &population, std::vector<float> bounds) {
    for (auto i = population.begin(); i != population.end(); i++){
        for (auto j = i->begin(); j != i->end(); j++) {
            *j = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
        }
    }
}