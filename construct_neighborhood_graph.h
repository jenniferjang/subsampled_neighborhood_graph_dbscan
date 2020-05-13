#include <vector>
#include <utility>
#include <cmath>
#include <set>
#include <random>

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

    int[p * n * n] test;
    vector<vector< pair<int, float> > > neighbors_distances(n);
    float distance;
    float sq_eps = pow(eps, 2);
    int neighbor;

    for (int i = 0; i < n; i++) {
        neighbors_distances[i] = vector< pair<int, float> >();
    }

    for (int i = 0; i < n; i++) {

        neighbors_distances[i].push_back(make_pair(i, 0));
      
        // To ensure neighborhood graph is symmetric, we only sample points that come after
        for (int j = 0; j < floor(p * (n - i)); j++) {

            int neighbor = rand() % (n-i) + i;
            distance = sq_euclidean_distance(d, i, neighbor, X);

            if (distance <= sq_eps) {
              
                // Add edge between both vertices
                neighbors_distances[i].push_back(make_pair(neighbor, distance));
                neighbors_distances[neighbor].push_back(make_pair(i, distance));
            }
        }

        num_neighbors[i] = neighbors_distances[i].size();
    }


    vector<int> neighbors;
    vector<float> distances;

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < neighbors_distances[i].size(); j++) {
            neighbors.push_back(neighbors_distances[i][j].first);
            distances.push_back(neighbors_distances[i][j].second);
        }
    }

    return make_pair(neighbors, distances);
}