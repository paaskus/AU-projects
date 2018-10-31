import numpy as np

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm.

        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster.
        cost:       The cost of the clustering
    """
    n, d = X.shape

    # Initialize clusters random.
    clustering = np.random.randint(0, k, (n, ))
    centroids  = np.zeros((k, d))

    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0

    # Column names
    print("Iterations\tCost")

    for i in range(T):

        # Update centroid

        # YOUR CODE HERE
        for j in range(k):
            clusterElements = np.zeros((n, d)) #Should hold all the elements in cluster j

            #Perhaps this could have been done more efficient so we had O(n) instead of O(n * k) (however k << n so it is not really an issue)
            for element in range(n):
                #If the input belongs to our cluster, add it to cluster elements
                if (clustering[element] == j):
                    clusterElements.append(X[element]) #Add the d dimensional point from the input to the elements belonging to our cluster

            numberOfElements = len(clusterElements)
            sumOfElements = clusterElements.sum()

            #I assume numpy knows how to handle a vector divided by an integer
            centroids[j] = (sumOfElements/numberOfElements)
        # END CODE

        # Update clustering

        # YOUR CODE HERE
        for i in range(n):
            #Inform the loop that this is the first time. We assume the distance can never really become -1
            lowestDistanceSoFar = -1
            cluster = 0
            for j in range(k):
                centroid = centroids[j]
                element = X[i]
                distance = np.sqrt(np.sum(np.power(np.subtract(element, centroid), 2))) #The euklidian distance

                if (distance < lowestDistanceSoFar or distanceSoFar == -1):
                    lowestDistanceSoFar = distance
                    cluster = j #To know which cluster has to lowest distance

            #After the loop, we know which cluster has the lowest euklidian distance for the i'th input point
            clustering[i] = cluster
        # END CODE


        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]])**2
        print(i+1, "\t\t", cost)


        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost): break #TODO
        oldcost = cost

    return clustering, centroids, cost

clustering, centroids, cost = lloyds_algorithm(X, 3, 100)
