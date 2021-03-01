int bestIndividual(std::vector<double> a, std::vector<int> b, int n) {
    
    std::vector<int>::size_type i, index = b[0];
    double diff = DBL_MAX;

    for (i = 0; i < n; i++) {
        double absVal = abs(a[b[i]]);
        if (absVal < diff) {
            index = b[i];
            diff = absVal;
        } else if (absVal == diff && a[b[i]] > 0 && a[index] < 0) {
            index = b[i];
        }
    }

    return index;
}

void tournamentSelection(std::vector<std::vector<double>> &parents, std::vector<std::vector<double>> population, 
                         std::vector<double> fitness, const int p, const int populationSize, const int tournamentSize) {
    
    std::vector<int> tournamentPool(tournamentSize, 0);

    int count = 0;
    int parentIndex = 0;
    while (count < populationSize) {
        
        // selecting individuals for tournament
        for (std::vector<int>::size_type i = 0; i != tournamentSize; i++) 
            tournamentPool[i] = rand() % populationSize; 

        parentIndex = bestIndividual(fitness, tournamentPool, tournamentSize);

        
        for (std::vector<int>::size_type i = 0; i != p; i++) 
            parents[count][i] = population[parentIndex][i];

        count++;
    }


}