int closeToZero(std::vector<double> a, int n) {

    std::vector<int>::size_type i, index = 0;
    double diff = DBL_MAX;
    
    for (i = 0; i < n; i++) {
        double absVal = abs(a[i]);
        if (absVal < diff) {
            index = i;
            diff = absVal;
        } else if (absVal == diff && a[i] > 0 && a[index] < 0) {
            index = i;
        }
    }

    return index;
}

void replacement(std::vector<std::vector<double>> &population, std::vector<std::vector<double>> temp_population,
                 std::vector<double> fitness, const int p, const int populationSize, const int elitism) {
    
    std::vector<int> minIndex(elitism, 0);

    for(std::vector<int>::size_type i = 0; i != elitism; i++) {
        minIndex[i] = closeToZero(fitness, populationSize);
        fitness[minIndex[i]] = DBL_MAX;
    }
    
    for(std::vector<int>::size_type i = 0; i != elitism; i++) 
        for (std::vector<int>::size_type j = 0; j != p; j++) 
            population[i][j] = population[minIndex[i]][j];

    for(std::vector<int>::size_type i = elitism; i != populationSize; i++) 
        for (std::vector<int>::size_type j = 0; j != p; j++) 
            population[i][j] = temp_population[i][j];

}