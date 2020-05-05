#include <queue>

using namespace std;


void DBSCAN_cy(int c, 
               int * core_pts,
               int * is_core_pt,
               int * neighbors,
               int * num_neighbors_cum,
               int * result) {
    /*
        
    */

    queue<int> q = queue<int>();
    int neighbor = 0, start_ind = 0, end_ind = 0, point = 0, cnt = 0;

    for (int i = 0; i < c; i++) {
        q = queue<int>();
        if (result[core_pts[i]] == -1) {
            q.push(core_pts[i]);

            while (!q.empty()) {
                point = q.front();
                q.pop();

                result[point] = cnt;

                if (point != 0) start_ind = num_neighbors_cum[point - 1];
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
