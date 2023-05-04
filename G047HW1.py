'''
HW1 - Group 47
  Alessandro Lucchiari
  Lorenzo Riccò
  Francesco Pio Monaco
'''
from pyspark import SparkContext, SparkConf
import sys, time, os
import random as rand
from CountTriangles import CountTriangles
from operator import add
from statistics import median, mean

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
def MR_ApproxTCwithSparkPartitions(edges, C):

   # Function to move in the iterator given by mapPartitions
  def f(iterator): 
    c = list(iterator)
    if len(c) == 0: # If the partition is empty
      return [(0,0)]
    for el in c:
      if el is None:
        return [(0,0)]
      else:
        return [(0,CountTriangles(list(el[1])))]


  partial_count = (edges.groupByKey() # ((rand.randint(0,C-1), list of edges)])
                        .mapPartitions(f) #[(0, CountTriangles(x[1]))])
                        .reduceByKey(add)
                        )
  return (list(partial_count.take(5))[0][1])

def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python G047HW1.py <C> <R> <file_name>"
    # SPARK SETUP
    conf = SparkConf().setAppName('TriangleCount')
    sc = SparkContext(conf=conf)
    # Check types
    assert sys.argv[1].isdigit(), "C must be an int"
    assert sys.argv[2].isdigit(), "R must be an int"
    # Take the parameters
    C = int(sys.argv[1])
    R = int(sys.argv[2])
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    # Load the input as an RDD textfile
    rawData = sc.textFile(data_path)
    # Transform into a (key,edge) RDD where edge is tuple of ints
    edges = rawData.map(lambda line: ((rand.randint(0,C-1)), tuple(map(int, line.split(",")))))
    edges = edges.repartition(C).cache()
    
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
    
    # Spark partitions
    start = time.time()
    triangle_estimate_Spark = C**2*(MR_ApproxTCwithSparkPartitions(edges, C))
    end = time.time()
    time_estimate_Spark = int((end-start)*1000)

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

    print("Approximation through Spark partitions\n\
- Number of triangles =",\
            triangle_estimate_Spark,\
               "\n- Running time =",\
                time_estimate_Spark,"ms")
  

if __name__ == "__main__":
	main()