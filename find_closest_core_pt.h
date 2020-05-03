#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

float euclidean_distance_sq(int d,
                            int x1, 
                            int x2,
                            float * X) {

    float distance_sq = 0;

    for (int i = 0; i < d; i++) {
        distance_sq += pow(X[x1 * d + i] - X[x2 * d + i], 2);
    } 

    return distance_sq;
}

void find_closest_core_pt_cy(int n,
                             int d,
                             float * X,
                             int * neighbors,
                             int * num_neighbors,
                             bool * is_core_point,
                             int * closest_core_pt,
                             float * dist_sq_to_core_pt) {
    /*
        

    */

    int start_ind, end_ind = 0;
    float distance_sq = 0;

    for (int i = 0; i < n; i++) {

        start_ind = num_neighbors[max(i - 1, 0)];
        end_ind = num_neighbors[i];
        dist_sq_to_core_pt[i] = numeric_limits<float>::max();

        for (int j = start_ind; j < end_ind; j++) {
            if (is_core_point[neighbors[j]] == 0) {
                distance_sq = euclidean_distance_sq(d, i, neighbors[j], X);
                if (distance_sq < dist_sq_to_core_pt[i]) {
                    closest_core_pt[i] = neighbors[j];
                    dist_sq_to_core_pt[i] = distance_sq;
                }
            }
        }
    }
}
