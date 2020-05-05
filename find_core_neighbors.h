#include <algorithm>
#include <iostream>

using namespace std;

void find_core_neighbors_cy(int n,
                            int * neighbors,
                            int * num_neighbors,
                            int * is_core_pt,
                            int * core_neighbors,
                            int * num_core_neighbors) {
    /*
        

    */

    int start_ind = 0, end_ind, core_cnt, neighbor_cnt = 0;

    for (int i = 0; i < n; i++) {
        // cout << "i " << i << endl;

        // cout << "end_ind " << num_neighbors[i] << endl;
        
        end_ind = num_neighbors[i];

        if (is_core_pt[i] > 0) {

            // Add the point itself as a neighbor
            core_neighbors[neighbor_cnt] = i;
            neighbor_cnt++;

            for (int j = start_ind; j < end_ind; j++) {
                if (is_core_pt[neighbors[j]] >= 0 && neighbors[j] != i) {
                    // cout << "j " << j << endl;
                    // cout << "cnt " << cnt << endl;
                    // cout << "neighbors[j] " << neighbors[j] << endl;
                    // cout << "core_indices[neighbors[j]] " << core_indices[neighbors[j]] << endl;
                    // cout << "core_neighbors[cnt] " << core_neighbors[cnt] << endl;
                    core_neighbors[neighbor_cnt] = neighbors[j];
                    neighbor_cnt++;
                }
            }
        
            // cout << "num_core_neighbors[i] " << num_core_neighbors[i] << endl;
            num_core_neighbors[core_cnt] = neighbor_cnt;
            // cout << "start_ind " << num_neighbors[i] << endl;
        }

        start_ind = num_neighbors[i];
        core_cnt++;
    }

}
