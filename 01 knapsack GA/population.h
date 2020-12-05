#include <stdlib.h>
#include <vector>


void initPopulation (std::vector<std::vector<int>> &population, const int solutionsPerPopulation, const int length) {

    for (auto i = population.begin(); i != population.end(); i++){
        for (auto j = i->begin(); j != i->end(); j++)
            *j = rand() % 2;
    }


}