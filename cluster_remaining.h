#include <limits>

using namespace std;

void cluster_remaining_cy(int pn, int n,
                          int * neighbors,
                          int * num_neighbors_cum,
                          int * is_core_pt,
                          int * result) {
    /*
        

    */

    int start_ind = 0;
    int end_ind = 0;
    int neighbor;
    //float distance;

    for (int i = 0; i < n; i++) {

        end_ind = num_neighbors_cum[i];
        //distance = numeric_limits<float>::max();

        for (int j = 0; j < pn; j++) {

            neighbor = neighbors[i * pn + j];
            if (neighbor < 0) break;

            if (is_core_pt[neighbor] > 0) {// && distances[i * pn + j] < distance) {
                result[i] = result[neighbor];
                //distance = distances[i * pn + j];
                break;
            }
        }

        start_ind = num_neighbors_cum[i];
    }
}