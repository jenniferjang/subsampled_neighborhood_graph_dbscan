#include <iostream>

using namespace std;

void cluster_remaining(float p, 
                       int n,
                       vector<int> & neighbors,
                       vector<bool> & is_core_pt,
                       int * result) {
    /*
        

    */

    int neighbor;

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < p * n; j++) {

            neighbor = neighbors[i * int(p * n) + j];
            if (neighbor < 0) break;

            if (is_core_pt[neighbor]) {

                result[i] = result[neighbor];
                break;
            }
        }
    }
}