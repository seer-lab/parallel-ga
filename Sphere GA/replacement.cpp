#include "replacement.h"
#include "evaluation.h"

int find_min(vector<double> a, int n) {
    vector<int>::size_type c, index = 0;

    for(c = 1; c != n; c++)
        if (a[c] < a[index])
            index = c;

    return index;
}

void replacement (vector<vector<double>> temp_population, vector<vector<double>> &population,  
                  vector<double> &fitness, const int populationSize, const int p) {

    vector<int> minIndex(4, 0);

    for(vector<int>::size_type i = 0; i != 4; i++) 
        minIndex[i] = find_min(fitness, populationSize);

    for (vector<int>::size_type j = 0; j != p; j++) {
            
        population[25*1-1][j] = population[minIndex[0]][j];
        population[25*2-1][j] = population[minIndex[1]][j];
        population[25*3-1][j] = population[minIndex[2]][j];
        population[25*4-1][j] = population[minIndex[3]][j];

    }
    
    for(vector<int>::size_type i = 0; i != (populationSize/4-1); i++) {
        for (vector<int>::size_type j = 0; j != p; j++) {
            population[i][j] = temp_population[i][j];
            population[i+24][j] = temp_population[i+24][j];
            population[i+24*2][j] = temp_population[i+24*2][j];
            population[i+24*3][j] = temp_population[i+24*3][j];
        }
    }

}