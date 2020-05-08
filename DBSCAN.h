#include <queue>

using namespace std;


void DBSCAN_cy(int n,
               int * is_core_pt,
               int * neighbors,
               int * num_neighbors_cum,
               int * result) {
    /*
        
    */

    queue<int> q = queue<int>();
    int neighbor, start_ind, end_ind, point, cnt = 0;

    for (int i = 0; i < n; i++) {

        q = queue<int>();

        if (is_core_pt[i] && result[i] == -1) {

            q.push(i);
            result[i] = cnt;

            while (!q.empty()) {

                point = q.front();
                q.pop();

                if (point != 0) {
                    start_ind = num_neighbors_cum[point - 1];
                } else {
                    start_ind = 0;
                }
                end_ind = num_neighbors_cum[point];

                for (int j = start_ind; j < end_ind; j++) {
                    neighbor = neighbors[j];

                    if (is_core_pt[neighbor] && result[neighbor] == -1) {
                        q.push(neighbor);
                        result[neighbor] = cnt;
                    }

                }

            }

            cnt ++;
        }
    }
}
