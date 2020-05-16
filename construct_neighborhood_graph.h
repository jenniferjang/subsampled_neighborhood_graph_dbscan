#include <vector>
#include <cmath>
#include <random>

using namespace std;


void construct_neighborhood_graph(float p, 
                                  int n,
                                  int d, 
                                  float eps,
                                  float * X,
                                  vector<int> & neighbors,
                                  vector<int> & num_neighbors) {
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
        for (int j = 0; j < int(p * (n - i)) - 1; j++) {

            low = i + 1;
            high = n - 1;
            neighbor = rand() % (high - low + 1) + low;
            
            distance = 0;
            for (int k = 0; k < d; k++) {
                distance += pow(X[i * d + k] - X[neighbor * d + k], 2);
                if (distance > sq_eps) break;
            }

            if ((distance <= sq_eps) && (num_neighbors[i] < pn)) {
              
                // Add edge between both vertices
                neighbors[i * pn + num_neighbors[i]] = neighbor;
                neighbors[neighbor * pn + num_neighbors[neighbor]] = i;

                num_neighbors[i]++;
                num_neighbors[neighbor]++;
            }
        }
    }
}