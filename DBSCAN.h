#include <queue>

using namespace std;


void DBSCAN(float p,
            int n,
            vector<bool> & is_core_pt,
            vector<int> & neighbors,
            int * result) {
    /*
        
    */

    queue<int> q = queue<int>();
    int neighbor, point;
    int cnt = 0;

    for (int i = 0; i < n; i++) {

        q = queue<int>();

        if (is_core_pt[i] && result[i] == -1) {

            q.push(i);
            result[i] = cnt;

            while (!q.empty()) {

                point = q.front();
                q.pop();

                for (int j = 0; j < p * n; j++) {
                    neighbor = neighbors[point * int(p * n) + j];
                    if (neighbor < 0) break;

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
