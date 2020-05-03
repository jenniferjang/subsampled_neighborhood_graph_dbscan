#include <limits>

using namespace std;

void cluster_remaining_cy(int n,
                          int * neighbors,
                          float * distances,
                          int * num_neighbors,
                          bool * is_core_point,
                          int * result) {
    /*
        

    */

    int start_ind, end_ind = 0;
    float distance;

    for (int i = 0; i < n; i++) {

        end_ind = num_neighbors[i];
        distance = numeric_limits<float>::max();

        for (int j = start_ind; j < end_ind; j++) {
            if (is_core_point[neighbors[j]] == 0 && distances[j] < distance) {
                result[i] = result[neighbors[j]];
            }
        }

        start_ind = num_neighbors[i];
    }
}