void crossover (std::vector<std::vector<double>> &temp_population, std::vector<std::vector<double>> parents, 
                const int p, const float crossoverProbablity, const int mating) {
    
    int crossoverPoint = 0;
    
    for (size_t i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbablity) {
            crossoverPoint = rand() % p;

            for (std::vector<int>::size_type j = 0; j != crossoverPoint; j++) {
                temp_population[i][j] = parents[i][j];
                temp_population[i+mating][j] = parents[i+mating][j];
            }

            for (std::vector<int>::size_type j = crossoverPoint; j != p; j++) {
                temp_population[i][j] = parents[i+mating][j];
                temp_population[i+mating][j] = parents[i][j];
            }
        } else {

            for (std::vector<int>::size_type j = 0; j != p; j++) { 
                temp_population[i][j] = parents[i][j];
                temp_population[i+mating][j] = parents[i+mating][j];
            }

        }
    }

}