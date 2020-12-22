#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <vector>

using std::vector;

int find_min(vector<int> a, int n) {
    vector<int>::size_type c, index = 0;

    for(c = 1; c != n; c++)
        if (a[c] < a[index])
            index = c;

    return index;
}

void eval (vector<int> individual, vector<int> &fitness, vector<int> itemWeight, 
           vector<int> itemProfit, const int index, const int length, const int maxWeight) {

    int sumWeight = 0;
    int sumProfit = 0;

    for(vector<int>::size_type i = 0; i != individual.size(); i++) {
        sumProfit += individual[i] * itemProfit[i];
        sumWeight += individual[i] * itemWeight[i];
    }

    if (sumWeight > maxWeight)
            fitness[index] = 0;
        else
            fitness[index] = sumProfit;

}

void replacement (vector<vector<int>> &population,  
                  vector<vector<int>> mutants, vector<vector<int>> parents, 
                  vector<int> &fitness, vector<int> itemWeight, vector<int> itemProfit, 
                  const int length, const int maxWeight) {

    vector<int> lowestIndex(4, 0);

    for(vector<int>::size_type i = 0; i != 4; i++) {

        lowestIndex[i] = find_min(fitness, length);

        for (vector<int>::size_type j = 0; j != population.size(); j++) {
          
            if (i < 2)
                population[lowestIndex[i]][j] = parents[i%2][j];
            else
                population[lowestIndex[i]][j] = mutants[i%2][j];

        }

        eval(population[i], fitness, itemWeight, itemProfit, lowestIndex[i], length, maxWeight);   
    }

}

