#include <vector>
#include <utility>
#include <cmath>
#include <set>
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


pair< vector<int>, vector<float> > construct_neighborhood_graph_cy(int n, 
                                                                   int d,
                                                                   float p,
                                                                   float eps,
                                                                   float * X,
                                                                   int * num_neighbors) {
    /*
        

    */

    set< pair<int, float> > neighbors_distances[n];
    vector<int> neighbors;
    vector<float> distances;
    float distance;
    float sq_eps = pow(eps, 2);
    set<int> unique;

    for (int i = 0; i < n; i++) {
        neighbors_distances[i] = set< pair<int, float> >();
    }

    for (int i = 0; i < n; i++) {

        neighbors_distances[i].insert(make_pair(i, 0));
      
        // To ensure neighborhood graph is symmetric, we only sample points that come after
        while (unique.size() < floor(p * (n - i))) {
          
            int k = rand() % (n-i) + i;
            unique.insert(k);
            distance = sq_euclidean_distance(d, i, k, X);

            if (distance <= sq_eps) {
              
                // Add edge between both vertices
                neighbors_distances[i].insert(make_pair(k, distance));
                neighbors_distances[k].insert(make_pair(i, distance));
            }
        }

        num_neighbors[i] = neighbors_distances[i].size();
        unique.clear();
    }


    for (int i = 0; i < n; i++) {

        set< pair<int, float> >::iterator it;

        for (it = neighbors_distances[i].begin(); it != neighbors_distances[i].end(); ++it) {
            neighbors.push_back(it->first);
            distances.push_back(it->second);
        }
    }

    return make_pair(neighbors, distances);
}