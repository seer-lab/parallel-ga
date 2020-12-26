int findTournamentMin(std::vector<double> a, std::vector<int> b, int n) {
    std::vector<int>::size_type c, index = b[0];

    for(c = 1; c != n; c++)
        if (a[b[c]] < a[index])
            index = b[c];

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

        parentIndex = findTournamentMin(fitness, tournamentPool, tournamentSize);

        
        for (std::vector<int>::size_type i = 0; i != p; i++) 
            parents[count][i] = population[parentIndex][i];

        count++;
    }


}