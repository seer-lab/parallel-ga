double square(double x) { return x * x;}

// Sphere Optimization: https://www.sfu.ca/~ssurjano/spheref.html
double sphere(std::vector<double> individual, const int p) { 

    double sum = 0.0;
    for (std::vector<int>::size_type i = 0; i != p; i++)
        sum += square(individual[i]);

    return sum; 
}

// Evaluates the score of each individual within a population
void evaluation (std::vector<std::vector<double>> population, std::vector<double> &fitness, const int p) {

    for (std::vector<int>::size_type i = 0; i != fitness.size(); i++) 
        fitness[i] = sphere(population[i], p);
} 