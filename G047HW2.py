'''
HW2 - Group 47
  Alessandro Lucchiari
  Lorenzo Riccò
  Francesco Pio Monaco
'''
from pyspark import SparkContext, SparkConf
import sys, time, os
import random as rand
from CountTriangles import CountTriangles
from CountTriangles2 import countTriangles2
from operator import add
from statistics import median, mean
from collections import defaultdict

def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


# Function 1
def MR_ApproxTCwithNodeColors(edges, C):

  # Variables definition
    pi = 8191
    a = rand.randint(1, pi-1)
    b = rand.randint(0, pi-1)
    dictionary = {} # To retrieve previous calculated values and remove useless calculations

  # Function that calculates the hash
    def hash(vertex):
        return ((a*vertex+b)%pi)%C

  # Function that colors the edges
    def color_per_edges(edges):
          edge = edges[1] #(i,j) tuple
          # Check if the hash was already calculated
          if edge[0] in dictionary.keys(): hash_a = dictionary[edge[0]]
          else: 
              hash_a = hash(edge[0])
              dictionary[edge[0]] = hash_a
              
          if edge[1] in dictionary.keys(): hash_b = dictionary[edge[1]]
          else:                 
             hash_b = hash(edge[1])
             dictionary[edge[1]] = hash_b
        # If the hashes of the two vertices are equal then emit (color,edge)
        # else don't emit anything
          if hash_a == hash_b: return [(hash_b, edge)]
          else: return []

    partial_count = (edges.flatMap(color_per_edges)       #Map 1
                        .groupByKey()                     #Shuffle (color,tuple)
                        .flatMap(lambda x: [(0, CountTriangles(x[1]))])   #Reduce 1 (0,partial_c)
                        .reduceByKey(add))                #Reduce 2

    # take(n) returns a list with n tuples inside, [(key,value)]
    return C**2*((partial_count.take(1)[0])[1]) 

# Function 2 
def MR_ExactTC(edges, C):
  
  # Variables definition
    pi = 8191
    a = rand.randint(1, pi-1)
    b = rand.randint(0, pi-1)
    dictionary = {} # To retrieve previous calculated values and remove useless calculations

  # Function that calculates the hash

    def hash(vertex):
        return ((a*vertex+b)%pi)%C

  # Function that colors the edges
    def color_per_edges(edges, C=16):
          edge = edges[1] #(i,j) tuple
          # Check if the hash was already calculated
          if edge[0] in dictionary.keys(): hash_a = dictionary[edge[0]]
          else: 
              hash_a = hash(edge[0])
              dictionary[edge[0]] = hash_a
              
          if edge[1] in dictionary.keys(): hash_b = dictionary[edge[1]]
          else:                 
             hash_b = hash(edge[1])
             dictionary[edge[1]] = hash_b
        # If the hashes of the two vertices are equal then emit (color,edge)
        # else don't emit anything
	
          #print([ (tuple(sorted((hash_a, hash_b, c)), edge) for c in range(C)])
          return [ (tuple(sorted((hash_a, hash_b, c))), edge) for c in range(C)]

    partial_count = (edges.flatMap(color_per_edges)       #Map 1
                        .groupByKey()                     #Shuffle (color,tuple)
                        .flatMap(lambda x: [(0, countTriangles2(x[0], x[1], a, b, pi, C))])   #Reduce 1 (0,partial_c)
                        .reduceByKey(add))                #Reduce 2

    # take(n) returns a list with n tuples inside, [(key,value)]
    return (partial_count.take(1)[0])[1] 

def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python G047HW2.py <C> <R> <file_name>"
    # SPARK SETUP
    conf = SparkConf().setAppName('TriangleCount')
    sc = SparkContext(conf=conf)
    # Check types
    assert sys.argv[1].isdigit(), "C must be an int"
    assert sys.argv[2].isdigit(), "R must be an int"
    # Take the parameters
    part = 16 # Base num of partitions
    C = int(sys.argv[1])
    R = int(sys.argv[2])
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    # Load the input as an RDD textfile
    rawData = sc.textFile(data_path)
    # Transform into a (key,edge) RDD where edge is tuple of ints
    edges = rawData.map(lambda line: ((rand.randint(0,part-1)), tuple(map(int, line.split(",")))))
    edges = edges.repartition(part).cache()
    
    # Node color partitions
      # Data structures for saving the results
    results_NC = []
    times_NC = []
      #Execution
    for i in range(R):
        start = time.time()
        results_NC.append(MR_ApproxTCwithNodeColors(edges, C))
        end = time.time()
        times_NC.append(end-start)
      #Results
    triangle_estimate_NC = median(results_NC)
        #*1000 from sec to ms, int to truncate the value
    time_estimate_NC = int(mean(times_NC)*1000)
    
    # Exact count
      # Data structures for saving the results
    results_EC = []
    times_EC = []
      #Execution
    for i in range(R):
        start = time.time()
        results_EC.append(MR_ExactTC(edges, C))
        end = time.time()
        times_EC.append(end-start)
      #Results
    triangle_exact = results_EC[-1]
        #*1000 from sec to ms, int to truncate the value
    time_estimate_EC = int(mean(times_EC)*1000)

    # Print section
    print("Dataset =", data_path)
    print("Number of Edges =", edges.count())
    print("Number of Colors =", C)
    print("Number of Repetitions =", R)
    print("Approximation through node coloring\n\
- Number of triangles (median over", R, "runs) =",\
            triangle_estimate_NC,\
               "\n- Running time (average over", R, "runs) =",\
                time_estimate_NC,"ms")

    #***EDIT WHEN THEY PUT THE FILE***
    print("Exact number of triangles\n\
- Number of triangles =",\
            triangle_exact,\
               "\n- Running time =",\
                time_estimate_EC,"ms")
  

if __name__ == "__main__":
	main()