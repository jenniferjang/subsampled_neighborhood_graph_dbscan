#include "construct_neighborhood_graph.h"
#include "DBSCAN.h"
#include "cluster_remaining.h"
#include <vector>
#include <iostream>


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