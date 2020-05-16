#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <random>
#include <iostream>

using namespace std;


// float sq_euclidean_distance(int d, int i, int j, float * X) {
    /*
        Return the squared distance between points i and j in X
        Parameters
        ----------
        d: dimensions of dataset X
        i: index of first point in X
        j: index of second point in X
        X: (m, d) dataset
    */

//     float distance = 0;

//     for (int k = 0; k < d; k++) {
//         distance += pow(X[i * d + k] - X[j * d + k], 2);
//     }

//     return distance;
// }


void construct_neighborhood_graph(float p, int n,
                                 int d, 
                                 float eps,
                                 float * X,
                                 vector<int> & neighbors,
                                 int * num_neighbors) {
    /*
        

    */

    float distance;
    int neighbor, low, high;
    float sq_eps = eps * eps;
    int pn = p * n;

    for (int i = 0; i < n; i++) {
        neighbors[i * pn + num_neighbors[i]] = i;
        num_neighbors[i]++;
      
        // To ensure neighborhood graph is symmetric, we only sample points that come after
        for (int j = 0; j < floor(p * (n - i)) - 1; j++) {
            // cout << "i " << i << " j " << j << endl;

            low = i + 1;
            high = n - 1;
            neighbor = rand() % (high - low + 1) + low;
            // distance = sq_euclidean_distance(d, i, neighbor, X);
            
            distance = 0;
            for (int k = 0; k < d; k++) {
                distance += pow(X[i * d + k] - X[neighbor * d + k], 2);
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