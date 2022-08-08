import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

datafile = './k-means/training.csv'
X = 0                   # Coordinate for matplot
Y = 1                   # Coordinate for matplot
NOSTROKE = 0            # Data class
STROKE = 1              # Data class
DIMENSIONS = 17         # Number of data dimensions  
CLASSIFIER = 17         # Data classifier index
CLASSES = 2             # Number of classes in the data
CLUSTERS = 20           # Number of clusters to use
RUNS = 1                # How many training runs to complete
MAX_ITERATIONS = 100    # Max iterations of the K-means before terminating
SCALE = 1               # matplot scalar
EQUILIBRIUM = 0.02      # Cluster centroid change in position % considered as 'in equillibrium' 
OUTPUT = False          # Output each iteration to a file
DISPLAY = True          # Display each iteration on screen
# Colors to use in scatter plot
COLORS = {0 : 'blue', 1 : 'orange', 2 : 'green', 3 : 'yellow', 4 : 'red', 5 : 'purple'}

# matplot font size, decreased when # of clusters is > 12
FONTSIZE = 9
if CLUSTERS > 12:
    FONTSIZE -= int((CLUSTERS - 9)/ 3)
    if FONTSIZE < 5:
        FONTSIZE = 5

# Pandas dataframe column header list
COLUMN_HEADERS = []
for dim in range(DIMENSIONS):
    COLUMN_HEADERS.append(dim)

def plot_scatter(x, y, classs):
     # Plot cluster points
     plt.scatter(x, y, s=(10 * SCALE), color=COLORS[classs])

def display_plot(cluster_assignment_embedded, class_count, iteration, run, mse, avg_mse, mss):
    # Create scatter plot
    if (DISPLAY == True or OUTPUT == True):
        plt.clf()
        for cluster in range(CLUSTERS):
            x = cluster_assignment_embedded[cluster]['X']
            y = cluster_assignment_embedded[cluster]['Y']
            classs = STROKE
            if class_count[cluster][NOSTROKE] > 0:
                ratio = class_count[cluster][STROKE] / class_count[cluster][NOSTROKE]
                if ratio < 0.0625:
                    classs = NOSTROKE               
            # Plot cluster points
            plot_scatter(x,y,classs)
        # Display average MSE 
        plt.text(-145,-145, f'Iteration: {iteration}   AVG MSE: {round(avg_mse, 3)}   MSS: {round(mss, 3)}', fontsize=9+SCALE)
        # Display cluster MSEs
        for cluster in range(CLUSTERS):
            x_pos = -155 + (7 * FONTSIZE * int(cluster / 3))
            y_pos = 140 - (8 * (cluster % 3))
        plt.text(x_pos, y_pos,f'MSE {cluster}: {round(mse[cluster],SCALE+1)}', fontsize=FONTSIZE+SCALE)
        if (DISPLAY == True):
            plt.pause(0.01)
        if(OUTPUT == True):
            plt.savefig(f'./k-means.output-c{CLUSTERS}-r{run}-i{iteration}.jpg')

def calc_mse(clusters, centroid):
    mse = []
    # Calculate Mean Squares Error for each cluster
    for c in range(CLUSTERS):
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
    # Create a data frame with the centroid data
    df = pd.DataFrame(data = centroid)
    for dim in range(DIMENSIONS):
        matrix.append(np.array(pd.DataFrame(np.array(df[dim]))))
        vector = np.transpose(matrix[dim])
        matrix[dim] = np.tile(matrix[dim], (1, CLUSTERS))
        matrix[dim] = np.subtract(matrix[dim], vector)
        matrix[dim] = np.power(matrix[dim], 2)
    sum = np.sum(matrix) / 2
    mss = (sum) / (CLUSTERS * (CLUSTERS-1) / 2)
    return mss

def load_data(datafile):
    # Load data 
    with open(datafile, newline='') as csvfile:
        data_original = pd.read_csv(csvfile, sep=',', engine='python', header=None)
        data = data_original.drop([17,18], axis=1)
    data_embedded = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=3).fit_transform(data)

    classified_data = []
    stroke = []
    nostroke = []
    classified_data.append(nostroke)
    classified_data.append(stroke)
    for index in range(len(data)):
        if data_original[CLASSIFIER][index] == 1:
            classified_data[1].append(data_embedded[index])
        else:
            classified_data[0].append(data_embedded[index])

    classified_data[0] = np.transpose(classified_data[0])
    classified_data[1] = np.transpose(classified_data[1])
    for classs in range(CLASSES):
        x = classified_data[classs][NOSTROKE]
        y = classified_data[classs][STROKE]
        # Plot cluster points
        plot_scatter(x, y, classs)
    plt.savefig(f'output-tSNE.jpg')
    plt.pause(5)
    plt.show
    return data, data_embedded, data_original

# # Display full screen
# if (FULLSCREEN == True):
#     manager = plt.get_current_fig_manager()
#     manager.full_screen_toggle()
#     SCALE = 5

def main():
    data, data_embedded, data_original = load_data(datafile)
    N = len(data)

    for run in range(RUNS):
        # Randomly select starting centroids 
        centroid = [[0]*DIMENSIONS for i in range(CLUSTERS)]
        for c in range(CLUSTERS):
            index = random.randint(0,N-1)
            for dim in range(DIMENSIONS):
                centroid[c][dim] = data[dim][index]

        done = False
        iteration = 1    
        while (done == False and iteration <= MAX_ITERATIONS):
            done = True
            distances = pd.DataFrame()
            # Calculate distances from each point to every centroid and store in 'distances'
            for c in range(CLUSTERS):
                distances_tmp = pd.DataFrame(np.subtract(data, centroid[c]))
                distances_tmp = distances_tmp.pow(2)
                distances[c] = distances_tmp.sum(axis=1)
            # Find the closest centroid for each point, store result in 'closest_centroid'
            closest_centroid = distances.idxmin(axis=1)

            cluster_assignment = []             # Stores data points by assigned cluster
            cluster_assignment_embedded = []    # Stores tSNE data points by assigned cluster(mirrors cluster assignment)
            class_count = []                    # Count of each class values stored in each cluster
            cluster_size = []                   # Size of each cluster

            # Create cluster dataframes
            for cluster in range (CLUSTERS):
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

            # Calculate Mean Square Error,Average MSE,Mean Square Separation, and Entropy
            mse = calc_mse(cluster_assignment, centroid)
            avg_mse = calc_avg_mse(mse)
            mss = calc_mss(centroid)
            # Display data on a scatter plot
            display_plot(cluster_assignment_embedded, class_count, iteration, run, mse, avg_mse, mss)

            # Calculate new centroids
            for cluster in range(CLUSTERS):
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

            print(f'Iteration: {iteration}   AVG MSE: {round(avg_mse, 4)}   MSS: {round(mss, 4)}')
            iteration += 1
        plt.savefig(f'./k-means/output-Final.jpg')
        print("FIN")
        if (DISPLAY == True):
            plt.show()

if __name__ == '__main__':
  main()
