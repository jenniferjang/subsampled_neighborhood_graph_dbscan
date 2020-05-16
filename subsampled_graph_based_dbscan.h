#include <queue>
#include <vector>

using namespace std;


void construct_neighborhood_graph(float p, 
                                  int n,
                                  int d, 
                                  float eps,
                                  float * X,
                                  vector<int> & neighbors,
                                  vector<int> & num_neighbors) {
    /*
        

    */

    float distance;
    int neighbor, low, high;
    float sq_eps = eps * eps;
    int pn = p * n;

    for (int i = 0; i < n; i++) {
        neighbors[i * pn + num_neighbors[i]] = i;
        num_neighbors[i]++;
      
        // To ensure neighborhood graph is symmetric, we only sample points that come after
        for (int j = 0; j < int(p * (n - i)) - 1; j++) {

            low = i + 1;
            high = n - 1;
            neighbor = rand() % (high - low + 1) + low;
            
            distance = 0;
            for (int k = 0; k < d; k++) {
                distance += pow(X[i * d + k] - X[neighbor * d + k], 2);
                if (distance > sq_eps) break;
            }

            if ((distance <= sq_eps) && (num_neighbors[i] < pn)) {
              
                // Add edge between both vertices
                neighbors[i * pn + num_neighbors[i]] = neighbor;
                neighbors[neighbor * pn + num_neighbors[neighbor]] = i;

                num_neighbors[i]++;
                num_neighbors[neighbor]++;
            }
        }
    }
}


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


void SubsampledGraphBasedDBSCAN_cy(float p, 
								   int n,
		                           int d, 
		                           float eps,
		                           int minPts,
		                           float * X,
		                           int * result) {
    /*
        

    */

    vector<int> neighbors (int(p * n) * n);
    vector<int> num_neighbors (n);
    vector<bool> is_core_pt (n);
	fill(num_neighbors.begin(), num_neighbors.end(), 0);
	fill(neighbors.begin(), neighbors.end(), -1);

    // Construct the neighborhood graph
    construct_neighborhood_graph(p, n, d, eps, X, neighbors, num_neighbors);

	// Find core points
	for (int i = 0; i < n; i++) {
		is_core_pt[i] = num_neighbors[i] >= max(minPts * p, float(2));
	}

	// Cluster core points
	DBSCAN(p, n, is_core_pt, neighbors, result);
    
	// Cluster border points
	cluster_remaining(p, n, neighbors, is_core_pt, result);
}