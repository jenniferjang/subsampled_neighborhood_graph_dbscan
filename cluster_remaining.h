#include <limits>

using namespace std;

void cluster_remaining_cy(int n,
                          int * neighbors,
                          int * num_neighbors_cum,
                          float * distances,
                          int * is_core_pt,
                          int * result) {
    /*
        

    */

    int start_ind = 0, end_ind = 0;
    float distance;

    for (int i = 0; i < n; i++) {

        end_ind = num_neighbors_cum[i];
        distance = numeric_limits<float>::max();    

        for (int j = start_ind; j < end_ind; j++) {
            if (is_core_pt[neighbors[j]] > 0 && distances[j] < distance) {
                result[i] = result[neighbors[j]];
                distance = distances[j];
            }
        }

        start_ind = num_neighbors_cum[i];
    }
}