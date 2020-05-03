#include <algorithm>

using namespace std;

void find_core_neighbors_cy(int c,
                            int * neighbors,
                            int * num_neighbors,
                            bool * is_core_point,
                            int * core_neighbors,
                            int * num_core_neighbors) {
    /*
        

    */

    int start_ind, end_ind, cnt = 0;

    for (int i = 0; i < c; i++) {

        end_ind = num_neighbors[i];

        for (int j = start_ind; j < end_ind; j++) {
            if (is_core_point[neighbors[j]] == 0) {
                core_neighbors[cnt] = neighbors[j];
                cnt++;
            }
        }
        
        num_core_neighbors[i] = cnt;
        start_ind = num_neighbors[i];
    }

}
