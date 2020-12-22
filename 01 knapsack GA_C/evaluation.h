#include <stdio.h> 

void evaluation (int *population, int *itemWeight, int *itemProfit, int *fitness, const int length, const int maxWeight) {

    int sumWeight = 0;
    int sumProfit = 0;


    for(int i = 0; i < length; i++) {
        sumWeight = 0;
        sumProfit = 0;

        // calculating total weight and profit for each individual in the population
        for (int j = 0; j < length; j++) {
            sumWeight += (*((population+i*length) + j) * itemWeight[j]);
            sumProfit += (*((population+i*length) + j) * itemProfit[j]);
        }


        if (sumWeight > maxWeight)
            fitness[i] = 0;
        else
            fitness[i] = sumProfit;
    }
} 

int find_min(int *a, int n) {
  int c, index = 0;
 
  for (c = 1; c < n; c++)
    if (a[c] < a[index])
      index = c;

  return index;
}

void eval (int *individual, int *fitness, int *itemWeight, int *itemProfit, const int index, const int length, const int maxWeight) {

    int sumWeight = 0;
    int sumProfit = 0;

    for (size_t i = 0; i < length; i++) {
        sumProfit += (*((individual+index*length) + i) * itemProfit[i]);
        sumWeight += (*((individual+index*length) + i) * itemWeight[i]);
    }

    if (sumWeight > maxWeight)
            fitness[index] = 0;
        else
            fitness[index] = sumProfit;

}