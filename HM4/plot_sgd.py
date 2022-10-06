def plotSGDPath (trainX, trainY, trajectory):
    pca = PCA(n_components=2)
    traj = np.array(trajectory)
    transW_B = pca.fit_transform(traj)

    def toyFunction (x1, x2):
        invW_B = pca.inverse_transform([x1, x2])
        return forward_prop(trainX, trainY, invW_B)[0] #First return value of forward_prop is loss.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-10, +10, 1)  
    axis2 = np.arange(-10, +10, 1)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  
    Yaxis = 2*np.pi*np.random.random(8) - np.pi
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()