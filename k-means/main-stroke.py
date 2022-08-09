import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

datafile = './k-means/training.csv'
test_data_file = './k-means/test.csv'
outputfile = './k-means/results.csv'
X = 0                   # Coordinate for matplot
Y = 1                   # Coordinate for matplot
NOSTROKE = 0            # Data class
STROKE = 1              # Data class
DIMENSIONS = 17         # Set data dimensions  
CLASSIFIER = DIMENSIONS # Data classifier index (Located at last index after the data)
CLASSES = 2             # Number of classes in the data
CLUSTERS = 15
RUNS = 5  
MAX_ITERATIONS = 100    # Max iterations of the K-means before terminating
SCALE = 1               # matplot scalar
EQUILIBRIUM = 0.02      # Cluster centroid change in position % considered as 'in equillibrium' 
CLASSIFICATION_THRESHOLD = 0.0625 # Threshold for declaring a cluster to represent STROKE
OUTPUT = False
DISPLAY = False
COLORS = {0 : 'blue', 1 : 'orange', 2 : 'green', 3 : 'yellow', 4 : 'red', 5 : 'purple'}

COLUMN_HEADERS = []
for dim in range(DIMENSIONS):
    COLUMN_HEADERS.append(dim)

def main():
        clusters = CLUSTERS
   # for clusters in range(25,CLUSTERS+1,5):
        for run in range(RUNS):
            print(f'Clusters: {clusters}  Run: {run}')
            CENTROIDS = 0
            CLASSIFICATIONS = 1
            MSE = 2
            MSS = 3
            data, data_embedded, data_original = load_data(datafile)
            plot_tSNE_data(data_embedded, data_original, clusters, run)

            classifier = process_data(data, data_embedded, data_original, clusters, run)        
            test_data, test_data_embedded, test_data_original = load_data(test_data_file)
            results = test_classifier(classifier[CENTROIDS], classifier[CLASSIFICATIONS], test_data, test_data_original)
            with open(outputfile, "a", newline='') as csvfile:
                csvfile.write(f'CLUSTERS,{clusters},RUN,{run}, MSE,{round(classifier[MSE],2)},MSS,{round(classifier[MSS],2)}\n')
                for predicted in range(CLASSES):
                    for actual in range(CLASSES):
                        csvfile.write(f'{results[predicted][actual]},')
                    csvfile.write('\n')
                csvfile.write('\n')
            csvfile.close()

def test_classifier(centroids, cluster_class, test_data, test_data_original):
    N = len(test_data)
    clusters = len(centroids)
    results = [[0] * CLASSES for _ in range(CLASSES)]
    closest_centroid = find_closest_centroids(test_data, centroids, clusters)
    for index in range(N):
        cluster = closest_centroid[index]
        data_class = test_data_original[CLASSIFIER][index]
        results[data_class][cluster_class[cluster]] += 1
    return results

def plot_scatter(x, y, classs):
     # Plot cluster points
     plt.scatter(x, y, s=(10 * SCALE), color=COLORS[classs])

def display_plot(cluster_data, cluster_classification, iteration, clusters):
    # Create scatter plot
    plt.clf()
    for cluster in range(clusters):
        x = cluster_data[cluster]['X']
        y = cluster_data[cluster]['Y']
        # Plot cluster points
        plot_scatter(x,y,cluster_classification[cluster])
    if (DISPLAY == True):
        plt.pause(0.01)
    if(OUTPUT == True):
        plt.savefig(f'./k-means/output-c{clusters}-i{iteration}.jpg')
        
def plot_tSNE_data(data_embedded, data_original, clusters, run):
        classified_data = []
        stroke = []
        nostroke = []
        classified_data.append(nostroke)
        classified_data.append(stroke)
        for index in range(len(data_embedded)):
            if data_original[CLASSIFIER][index] == 1:
                classified_data[STROKE].append(data_embedded[index])
            else:
                classified_data[NOSTROKE].append(data_embedded[index])

        classified_data[0] = np.transpose(classified_data[0])
        classified_data[1] = np.transpose(classified_data[1])
        plt.clf()
        for classs in range(CLASSES):
            x = classified_data[classs][X]
            y = classified_data[classs][Y]
            # Plot cluster points
            plot_scatter(x, y, classs)
        plt.savefig(f'./k-means/output-tSNE-c{clusters}-r{run}.jpg')
        if (DISPLAY == True):
            plt.pause(5)
            plt.show

def calc_mse(clusters, centroid):
    mse = []
    # Calculate Mean Squares Error for each cluster
    for c in range(len(centroid)):
        # Subtract X and Y values of the centroid from every X and Y value in the cluster
        mse.append(np.subtract(clusters[c], centroid[c]))
        # Square all the X and Y values 
        mse[c] = mse[c].pow(2)
        # Sum the X and Y values for each data point in the cluster (distance)
        mse[c] = mse[c].sum(axis=1)
        # Average the distances from the centroid (MSE)
        if len(mse[c]) != 0:
            mse[c] = np.sum(mse[c]) / len(mse[c])
        else:
            mse[c] = 0
    return mse

def calc_avg_mse(mse):
    # Find the average MSE
    return sum(mse)/len(mse)

def calc_mss(centroid):
    matrix = []
    clusters = len(centroid)
    # Create a data frame with the centroid data
    df = pd.DataFrame(data = centroid)
    for dim in range(DIMENSIONS):
        matrix.append(np.array(pd.DataFrame(np.array(df[dim]))))
        vector = np.transpose(matrix[dim])
        matrix[dim] = np.tile(matrix[dim], (1, clusters))
        matrix[dim] = np.subtract(matrix[dim], vector)
        matrix[dim] = np.power(matrix[dim], 2)
    sum = np.sum(matrix) / 2
    mss = sum / (clusters * (clusters-1) / 2)
    return mss

def load_data(datafile):
    # Load data 
    with open(datafile, newline='') as csvfile:
        data_original = pd.read_csv(csvfile, sep=',', engine='python', header=None)
        data = data_original.drop([CLASSIFIER,CLASSIFIER+1], axis=1)

    data_embedded = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=3).fit_transform(data)
    return data, data_embedded, data_original

def process_data(data, data_embedded, data_original, clusters, run):
    N = len(data)
    # Randomly select starting centroids 
    centroid = [[0]*DIMENSIONS for i in range(clusters)]
    for c in range(clusters):
        index = random.randint(0,N-1)
        for dim in range(DIMENSIONS):
            centroid[c][dim] = data[dim][index]

    done = False
    iteration = 1    
    while (done == False and iteration <= MAX_ITERATIONS):
        done = True
        closest_centroid = find_closest_centroids(data, centroid, clusters)

        cluster_assignment = []             # Stores data points by assigned cluster
        cluster_assignment_embedded = []    # Stores tSNE data points by assigned cluster(mirrors cluster assignment)
        class_count = []                    # Count of each class values stored in each cluster
        cluster_size = []                   # Size of each cluster
        # Create cluster dataframes
        for cluster in range (clusters):
            # Append empty data frame to store data points
            cluster_assignment.append(pd.DataFrame(columns = COLUMN_HEADERS))
            # Append empty data frame to store tSNE data points
            cluster_assignment_embedded.append(pd.DataFrame(columns = ['X', 'Y']))
            # Start size counter for this cluster at 0
            cluster_size.append(0)
            # Create and array of of 0's, one for each data class
            cluster_class_count = []
            for classs in range(CLASSES):
                cluster_class_count.append(0)
            # Append the class counting array for the current cluster to class_count
            class_count.append(cluster_class_count)

        # Sort data into their clusters
        for index in range(N-1):
            # Get the cluster closest to the current data index
            cluster = closest_centroid[index]
            # Assign the current data to it's cluster
            cluster_assignment[cluster].loc[cluster_size[cluster]] = data.iloc[index]
            cluster_assignment_embedded[cluster].loc[cluster_size[cluster]] = data_embedded[index]
            for classs in range(CLASSES):
                if (data_original[CLASSIFIER][index] == classs):
                    class_count[cluster][classs] += 1
            # Increment the size of the current cluster
            cluster_size[cluster] += 1

        cluster_classification = []
        for cluster in range(clusters):
            cluster_classification.append(classify_cluster(class_count, cluster))

        # Display data on a scatter plot
        if (DISPLAY == True or OUTPUT == True):            
            mse = calc_mse(cluster_assignment, centroid)
            avg_mse = calc_avg_mse(mse)
            mss = calc_mss(centroid)
            display_plot(cluster_assignment_embedded, cluster_classification, iteration, clusters)

        # Calculate new centroids
        for cluster in range(clusters):
            for dim in range(DIMENSIONS):
                vals = cluster_assignment[cluster][dim]
                # Calc new centroids as mean of all X and Y values in the cluster
                if (len(vals.index > 0)):
                    dim_new_centroid = float(vals.sum())/len(vals.index)
                    # Check - Change in X and Y value < EQUILIBRIUM? 'done' when all centroids X & Y DELTA < EQUILIBRIUM
                    if (dim_new_centroid != 0 and np.abs((centroid[cluster][dim] / dim_new_centroid) - 1) > EQUILIBRIUM): 
                        done = False
                    # Store new centroids
                    centroid[cluster][dim]= dim_new_centroid
        print(f'Iteration: {iteration}')
        iteration += 1
    # Calculate Mean Square Error,Average MSE,Mean Square Separation, and Entropy
    mse = calc_mse(cluster_assignment, centroid)
    avg_mse = calc_avg_mse(mse)
    mss = calc_mss(centroid)
    display_plot(cluster_assignment_embedded, cluster_classification, iteration, clusters)  
    plt.savefig(f'./k-means/output-Final-c{clusters}-r{run}.jpg')
    print("FIN")
    return [centroid, cluster_classification, avg_mse, mss]

def find_closest_centroids(data, centroid, clusters):
    distances = pd.DataFrame()
    # Calculate distances from each point to every centroid and store in 'distances'
    for c in range(clusters):
        distances_tmp = pd.DataFrame(np.subtract(data, centroid[c]))
        distances_tmp = distances_tmp.pow(2)
        distances[c] = distances_tmp.sum(axis=1)
    # Find the closest centroid for each point, store result in 'closest_centroid'
    closest_centroid = distances.idxmin(axis=1)
    return closest_centroid

def classify_cluster(class_count, cluster):
    classs = STROKE
    if class_count[cluster][NOSTROKE] > 0:
        ratio = class_count[cluster][STROKE] / class_count[cluster][NOSTROKE]
        if ratio < CLASSIFICATION_THRESHOLD:
            classs = NOSTROKE               
    return classs

if __name__ == '__main__':
  main()
