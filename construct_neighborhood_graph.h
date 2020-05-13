#include <utility>
#include <random>
#include <iostream>

using namespace std;


float sq_euclidean_distance(int d, int i, int j, float * X) {
    /*
        Return the squared distance between points i and j in X
        Parameters
        ----------
        d: dimensions of dataset X
        i: index of first point in X
        j: index of second point in X
        X: (m, d) dataset
    */

    float distance = 0;

    for (int k = 0; k < d; k++) {
        distance += pow(X[i * d + k] - X[j * d + k], 2);
    }

    return distance;
}


void construct_neighborhood_graph_cy(float p, int pn, int n,
                                                                   int d, 
                                                                   float eps,
                                                                   float * X,
                                                                   int * neighbors,
                                                                   float * distances,
                                                                   int * num_neighbors) {
    /*
        

    */

    float distance, tmp;
    int neighbor;
    float sq_eps = eps * eps;

    for (int i = 0; i < n; i++) {

        neighbors[i * pn + num_neighbors[i]] = i;
        num_neighbors[i]++;
      
        // To ensure neighborhood graph is symmetric, we only sample points that come after
        for (int j = 0; j < floor(p * (n - i)) - 1; j++) {

            low = i + 1;
            high = n - 1;
            neighbor = rand() % (high - low + 1) + low;

            distance = 0; //sq_euclidean_distance(d, i, neighbor, X);
            //cout << "i " << i << " j " << j << " distance " << distance << " i * pn " << i * pn << " neighbor " << neighbor << " num_neighbors[i] " << num_neighbors[i] << endl;

            for (int k = 0; k < d; k++) {
              
                tmp = X[i * d + k] - X[j * d + k];
                distance += tmp * tmp;
                if (distance > sq_eps) break;
            }

            if ((distance <= sq_eps) && (num_neighbors[i] < pn)) {
              
                // Add edge between both vertices
                neighbors[i * pn + num_neighbors[i]] = neighbor;
                neighbors[neighbor * pn + num_neighbors[neighbor]] = i;

                // distances[i * pn + num_neighbors[i]] = distance;
                // distances[neighbor * pn + num_neighbors[neighbor]] = distance;

                num_neighbors[i]++;
                num_neighbors[neighbor]++;
            }
        }
    }
}