#include "../include/crossover.h"

#define d 0.25

void one_point_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating) {
    int crossoverPoint = 0;
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            crossoverPoint = rand() % p;

            for (unsigned int j = 0; j < crossoverPoint; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }

            for (unsigned int j = crossoverPoint; j < p; j++) {
                temp_population[i * p + j] = parents[(i+mating) * p + j];
                temp_population[(i+mating) * p + j] = parents[i * p + j];
            }
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }
        }
    }
}

void arithmetic_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const float alpha) {
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            for(unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = alpha * parents[i * p + j] + (1-alpha) * parents[(i+mating) * p + j];
                temp_population[(i+mating) * p + j] = (1-alpha) * parents[i * p + j] + alpha * parents[(i+mating) * p + j];
            }
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }  
        }
    }
}

double calc_B(float u, int nc) {
    if (u <= 0.5f) 
        return pow(2.0*u, 1.0/(nc+1));
    else 
        return pow(1.0/(2.0*(1.0-u)), 1.0/(nc+1));
}

/**
 * https://engineering.purdue.edu/~sudhoff/ee630/Lecture04.pdf
 */
void simulated_binary_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const int nc) {
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            for(unsigned int j = 0; j < p; j++) {
                float u = (float) rand()/RAND_MAX;
                double b = calc_B(u, nc);
                
                temp_population[i * p + j] = 0.5*(((1+b) * parents[i * p + j]) + ((1-b) * parents[(i+mating) * p + j]));
                temp_population[(i+mating) * p + j] = 0.5*(((1-b) * parents[i * p + j]) + ((1+b) * parents[(i+mating) * p + j]));
            } 
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }  
        }
    }    
}


/**
 * http://jultika.oulu.fi/files/isbn9789514287862.pdf
 * http://www.geatbx.com/docu/algindex-03.html#P587_32771
 */
void line_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating) {
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            for(unsigned int j = 0; j < p; j++) {
                float alpha = -d + ((float) rand()/RAND_MAX) * ((1+d) - -d);
                float alpha2 = -d + ((float) rand()/RAND_MAX) * ((1+d) - -d);

                temp_population[i * p + j] = (parents[i * p + j]) + alpha * (parents[(i+mating) * p + j] - parents[i * p + j]);
                temp_population[(i+mating) * p + j] = (parents[(i+mating) * p + j]) + alpha2 * (parents[i * p + j] - parents[(i+mating) * p + j]);
            } 
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }  
        }
    } 
}