#include <iostream>

using namespace std;

void cluster_remaining(float p, int n,
                          vector<int> & neighbors,
                          // int * num_neighbors_cum,
                          bool * is_core_pt,
                          int * result) {
    /*
        

    */

    // int start_ind = 0;
    // int end_ind = 0;
    int neighbor;
    //float distance;

    for (int i = 0; i < n; i++) {

        // end_ind = num_neighbors_cum[i];
        //distance = numeric_limits<float>::max();

        for (int j = 0; j < p * n; j++) {

            neighbor = neighbors[i * int(p * n) + j];
            if (neighbor < 0) break;

            if (is_core_pt[neighbor]) {// && distances[i * pn + j] < distance) {
                result[i] = result[neighbor];
                //distance = distances[i * pn + j];
                break;
            }
        }

        // start_ind = num_neighbors_cum[i];
    }
}