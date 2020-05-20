#include <queue>
#include <vector>

using namespace std;


void construct_neighborhood_graph(float p, 
                                  int n,
                                  int d, 
                                  float eps,
                                  float * X,
                                  vector< vector<int> > & neighbors) {
    /*
        

    */

    float distance;
    int neighbor, low, high;
    float sq_eps = eps * eps;

    for (int i = 0; i < n; i++) {
      
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

            if (distance <= sq_eps) {
              
                // Add edge between both vertices
                neighbors[i].push_back(neighbor);
                neighbors[neighbor].push_back(i);
            }
        }
    }
}


void DBSCAN(int n,
            vector<bool> & is_core_pt,
            vector< vector<int> > & neighbors,
            int * result) {
    /*
        
    */

    queue<int> q = queue<int>();
    int neighbor, point, cnt = 0;

    for (int i = 0; i < n; i++) {

        q = queue<int>();

        if (is_core_pt[i] && result[i] == -1) {

            q.push(i);
            result[i] = cnt;

            while (!q.empty()) {

                point = q.front();
                q.pop();

                for (vector<int>::size_type j = 0; j < neighbors[point].size(); j++) {
                    neighbor = neighbors[point][j];

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


void cluster_remaining(int n,
                       vector< vector<int> > & neighbors,
                       vector<bool> & is_core_pt,
                       int * result) {
    /*
        

    */
    int neighbor;

    for (int i = 0; i < n; i++) {

        if (result[i] != -1) continue;

        for (vector<int>::size_type j = 0; j < neighbors[i].size(); j++) {
            neighbor = neighbors[i][j];
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
    vector< vector<int> > neighbors (n);
    vector<bool> is_core_pt (n);

    // Construct the neighborhood graph
    construct_neighborhood_graph(p, n, d, eps, X, neighbors);

	// Find core points
	for (int i = 0; i < n; i++) {
		is_core_pt[i] = neighbors[i].size() >= minPts * p;
	}

	// Cluster core points
	DBSCAN(n, is_core_pt, neighbors, result);
    
	// Cluster border points
	cluster_remaining(n, neighbors, is_core_pt, result);
}