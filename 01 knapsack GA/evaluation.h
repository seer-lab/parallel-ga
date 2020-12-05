#include <stdio.h> 
#include <vector>

using std::vector;

void evaluation (vector<vector<int>> population, vector<int> itemWeight, vector<int> itemProfit, 
                 vector<int> &fitness, const int length, const int maxWeight) {

    int sumWeight = 0;
    int sumProfit = 0;


    for (vector<int>::size_type i = 0; i != population.size(); i++) {
        sumWeight = 0;
        sumProfit = 0;

        // calculating total weight and profit for each individual in the population
        for (vector<int>::size_type j = 0; j != population.size(); j++) {

            sumWeight += (population[i][j] * itemWeight[j]);
            sumProfit += (population[i][j] * itemProfit[j]);
        }

        if (sumWeight > maxWeight)
            fitness[i] = 0;
        else
            fitness[i] = sumProfit;
    }
} 


