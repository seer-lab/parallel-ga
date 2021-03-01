#include "population.h"

void initPopulation (vector<vector<double>> &population, vector<float> bounds) {
    for (auto i = population.begin(); i != population.end(); i++){
        for (auto j = i->begin(); j != i->end(); j++) {
            *j = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
        }
    }
}