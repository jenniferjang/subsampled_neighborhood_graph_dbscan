using namespace std;

void find_core_pts_cy(int n,
                      int minPts,
                      int * num_neighbors,
                      bool * is_core_pt) {
    /*
        

    */

    for (int i = 0; i < n; i++) {
        if (num_neighbors[i] >= minPts) {
            is_core_pt[i] = 1;
        }
    }
}
