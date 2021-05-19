import pandas as pds
# pandas : used for working with and manipulating databases
# sklearn : machine learning library for python
from sklearn.cluster import KMeans
# matplotlib : python low level graph plotting library
import matplotlib.pyplot as plot_graph


def customer_seg_AI_and_SS():
    # Function for Segmentation on the basis of Annual Income and Spending Score
    train = customer_info.iloc[:, [3, 4]]
    train = train.values
    wcss = []
    # wcss : Within Cell Sum of Square
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(train)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow graph
    plot_graph.plot(range(1, 15), wcss)
    plot_graph.title('Elbow Method\n')
    plot_graph.xlabel('Number of Clusters')
    plot_graph.ylabel('WCSS')
    plot_graph.show()

    # Training the model
    kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
    y_kmeans = kmeansmodel.fit_predict(train)

    # Scatter Graph Plot
    plot_graph.scatter(train[y_kmeans == 0, 0], train[y_kmeans == 0, 1], s=60, c='red', label='Cluster1')
    plot_graph.scatter(train[y_kmeans == 1, 0], train[y_kmeans == 1, 1], s=60, c='blue', label='Cluster2')
    plot_graph.scatter(train[y_kmeans == 2, 0], train[y_kmeans == 2, 1], s=60, c='green', label='Cluster3')
    plot_graph.scatter(train[y_kmeans == 3, 0], train[y_kmeans == 3, 1], s=60, c='violet', label='Cluster4')
    plot_graph.scatter(train[y_kmeans == 4, 0], train[y_kmeans == 4, 1], s=60, c='yellow', label='Cluster5')
    # plot_graph.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plot_graph.xlabel('Annual Income (k$)')
    plot_graph.ylabel('Spending Score (1-100)')
    plot_graph.legend()
    plot_graph.figtext(1, 0.5,
                       "Cluster1 : Average Annual Income with Average Spending Score\nCluster2 : Low Annual Income with High Spending Score "
                       "\nCluster3 : High Annual Income with High Spending Score\nCluster4 : Low Annual Income with Low Spending Score\nCluster5 : High Annual Income with Low Spending Score")
    plot_graph.show()

    # Bar Graph Plot
    plot_graph.bar(train[y_kmeans == 0, 0], train[y_kmeans == 0, 1], color='red', label='Cluster1')
    plot_graph.bar(train[y_kmeans == 1, 0], train[y_kmeans == 1, 1], color='blue', label='Cluster2', alpha=0.6)
    plot_graph.bar(train[y_kmeans == 2, 0], train[y_kmeans == 2, 1], color='green', label='Cluster3', alpha=0.3)
    plot_graph.bar(train[y_kmeans == 3, 0], train[y_kmeans == 3, 1], color='violet', label='Cluster4', alpha=0.5)
    plot_graph.bar(train[y_kmeans == 4, 0], train[y_kmeans == 4, 1], color='yellow', label='Cluster5')
    plot_graph.xlabel('Annual Income (k$)')
    plot_graph.ylabel('Spending Score(1 - 100)')
    plot_graph.legend()
    plot_graph.figtext(1, 0.5,
                       "Cluster1 : Average Annual Income with Average Spending Score\nCluster2 : Low Annual Income with High Spending Score "
                       "\nCluster3 : High Annual Income with High Spending Score\nCluster4 : Low Annual Income with Low Spending Score\nCluster5 : High Annual Income with Low Spending Score")
    plot_graph.show()
    start()


def customer_seg_Age_and_AI():
    # Function for Segmentation on the basis of Age and Annual Income
    train1 = customer_info.iloc[:, [2, 3]]
    train1 = train1.values
    from sklearn.cluster import KMeans
    WCSS1 = []  # WCSS = Within Cell Sum of Square
    for i in range(1, 15):
        kmeans1 = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans1.fit(train1)
        WCSS1.append(kmeans1.inertia_)

    # Plotting the elbow graph
    plot_graph.plot(range(1, 15), WCSS1)
    plot_graph.title('Elbow Method\n')
    plot_graph.xlabel('Number of Clusters')
    plot_graph.ylabel('WCSS1')
    plot_graph.show()

    # Training the model
    kmeansmodel1 = KMeans(n_clusters=4, init='k-means++', random_state=0)
    y_kmeans1 = kmeansmodel1.fit_predict(train1)

    # Scatter Graph Plot
    plot_graph.scatter(train1[y_kmeans1 == 0, 0], train1[y_kmeans1 == 0, 1], s=60, c='red', label='Cluster1')
    plot_graph.scatter(train1[y_kmeans1 == 1, 0], train1[y_kmeans1 == 1, 1], s=60, c='blue', label='Cluster2')
    plot_graph.scatter(train1[y_kmeans1 == 2, 0], train1[y_kmeans1 == 2, 1], s=60, c='green', label='Cluster3')
    plot_graph.scatter(train1[y_kmeans1 == 3, 0], train1[y_kmeans1 == 3, 1], s=60, c='violet', label='Cluster4')
    # plot_graph.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plot_graph.xlabel('Age')
    plot_graph.ylabel('Annual Income (k$)')
    plot_graph.figtext(1, 0.5,
                       "Cluster1 : Middle aged People with high income\nCluster2 : Old Aged People with Average Income"
                       "\nCluster3 : Teenagers and Middle aged people with Low Income\nCluster4 : Teens and Middle aged people with Average Income")
    plot_graph.legend()
    plot_graph.show()

    # Bar Graph Plot
    plot_graph.bar(train1[y_kmeans1 == 0, 0], train1[y_kmeans1 == 0, 1], color='red', label='Cluster1')
    plot_graph.bar(train1[y_kmeans1 == 1, 0], train1[y_kmeans1 == 1, 1], color='blue', label='Cluster2', alpha=0.6)
    plot_graph.bar(train1[y_kmeans1 == 2, 0], train1[y_kmeans1 == 2, 1], color='black', label='Cluster3')
    plot_graph.bar(train1[y_kmeans1 == 3, 0], train1[y_kmeans1 == 3, 1], color='violet', label='Cluster4', alpha=0.5)
    plot_graph.xlabel('Age')
    plot_graph.ylabel('Annual Income (k$)')
    plot_graph.legend()
    plot_graph.figtext(1, 0.5,
                       "Cluster1 : Middle aged People with high income\nCluster2 : Old Aged People with Average Income"
                       "\nCluster3 : Teenagers and Middle aged people with Low Income\nCluster4 : Teens and Middle aged people with Average Income")
    plot_graph.show()
    start()


def customer_seg_Age_and_SS():
    # Function for Segmentation on the basis of Age and Spending Score
    train2 = customer_info.iloc[:, [2, 4]]
    train2 = train2.values
    from sklearn.cluster import KMeans  # Age and SC
    WCSS2 = []  # WCSS = Within Cell Sum of Square
    for i in range(1, 15):
        kmeans2 = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans2.fit(train2)
        WCSS2.append(kmeans2.inertia_)

    # Plotting Elbow Graph
    plot_graph.plot(range(1, 15), WCSS2)
    plot_graph.title('Elbow Method\n')
    plot_graph.xlabel('Number of Clusters')
    plot_graph.ylabel('WCSS2')
    plot_graph.show()

    # Training the model
    kmeansmodel2 = KMeans(n_clusters=6, init='k-means++', random_state=0)
    y_kmeans2 = kmeansmodel2.fit_predict(train2)

    # Scatter Graph Plot
    plot_graph.scatter(train2[y_kmeans2 == 0, 0], train2[y_kmeans2 == 0, 1], s=60, c='red', label='Cluster1')
    plot_graph.scatter(train2[y_kmeans2 == 1, 0], train2[y_kmeans2 == 1, 1], s=60, c='blue', label='Cluster2')
    plot_graph.scatter(train2[y_kmeans2 == 2, 0], train2[y_kmeans2 == 2, 1], s=60, c='green', label='Cluster3')
    plot_graph.scatter(train2[y_kmeans2 == 3, 0], train2[y_kmeans2 == 3, 1], s=60, c='violet', label='Cluster4')
    # plot_graph.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plot_graph.xlabel('Age')
    plot_graph.ylabel('Spending Score(1 - 100)')
    plot_graph.figtext(1, 0.5, "Cluster1 : Teens to Middle aged with Average Spending Score\nCluster2 : Teens with High Spending Score"
                       "\nCluster3 : People with Low Spending Score\nCluster4 : Middle aged to Old aged People with Average Spending Score")
    plot_graph.legend()
    plot_graph.show()

    # Bar Graph Plot
    plot_graph.bar(train2[y_kmeans2 == 0, 0], train2[y_kmeans2 == 0, 1], color='red', label='Cluster1')
    plot_graph.bar(train2[y_kmeans2 == 1, 0], train2[y_kmeans2 == 1, 1], color='blue', label='Cluster2', alpha=0.6)
    plot_graph.bar(train2[y_kmeans2 == 2, 0], train2[y_kmeans2 == 2, 1], color='black', label='Cluster3', alpha=0.3)
    plot_graph.bar(train2[y_kmeans2 == 3, 0], train2[y_kmeans2 == 3, 1], color='violet', label='Cluster4', alpha=0.5)
    plot_graph.xlabel('Age')
    plot_graph.ylabel('Spending Score(1 - 100)')
    plot_graph.legend()
    plot_graph.figtext(1, 0.5,
                       "Cluster1 : Teens to Middle aged with Average Spending Score\nCluster2 : Teens with High Spending Score"
                       "\nCluster3 : People with Low Spending Score\nCluster4 : Middle aged to Old aged People with Average Spending Score")
    plot_graph.show()
    start()


def start():
    print(
        "Menu : \n1. Segmentation on the basis of Annual Income and Spending Score.\n2.Segmentation on the basis of Age and Annual Income."
        "\n3. Segmentation on the basis of Age and Spending Score\n4. Exit.")
    o = int(input("Enter Choice -> "))
    if o == 1:
        print("------------------- SEGMENTATION ON THE BASIS OF ANNUAL INCOME AND SPENDING SCORE -------------------")
        customer_seg_AI_and_SS()
    elif o == 2:
        print("------------------- SEGMENTATION ON THE BASIS OF AGE AND ANNUAL INCOME -------------------")
        customer_seg_Age_and_AI()
    elif o == 3:
        print("------------------- SEGMENTATION ON THE BASIS OF AGE AND SPENDING SCORE -------------------")
        customer_seg_Age_and_SS()
    elif o == 4:
        exit()


if __name__ == "__main__":
    # Main Function
    customer_info = pds.read_csv("C:/Users/Siddhant Bohra/PycharmProjects/Mini Project/Mall_Customers.csv")
    # Storing Database into customer_info variable
    print("\n->\tTo take an idea about what type of Database we are dealing with : ")
    print(customer_info.head())
    print("\n->\tTo check 6 sample data from the given Database : ")
    print(customer_info.sample(6))
    print("\n->\tTo check the dimensions of the Database : ", end=" ")
    print(customer_info.shape)
    print("\n->\tTo get the information about the Database : ")
    print(customer_info.info())
    print("\n->\tTo check whether if the Database contains any Empty value : ")
    print(customer_info.isnull().sum())
    print("\n->\tLets start Segmenting data : \n")
    start()
