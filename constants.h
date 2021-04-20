#ifndef CONSTANTS_H
#define CONSTANTS_H

// Island Parameters
#define warp 64
#define islands 64

#define populationSize1 8192
#define p1 128

#define individualsPerIsland populationSize1/islands

#define mating1 ceil((populationSize1)/2)

#define elitism 2
#define tournamentSize 6
#define crossoverProbability 0.9f
#define mutationProbability 0.01f

#define alpha 0.25f
#define nc 4
#define d 0.25

#define numGen 10000

#define crossover_type 3
#define evaluation_type 4

// bounds xi ∈ [-5.12, 5.12]
// bounds xi ∈ [-32.768, 32.768]
// bounds xi ∈ [-600, 600]
#define lowerBound -600
#define upperBound 600

#endif