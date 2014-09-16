from my_pcl import *



if __name__ == '__main__':
    pc1 = read_pcd('../../videos/rgbd/scenes/desk/desk_1/desk_1_1.pcd')
    pc2 = read_pcd('../../videos/rgbd/scenes/desk/desk_1/desk_1_2.pcd')

    print "La primer nube tiene", points(pc1), "puntos y la segunda tiene", points(pc2)

    pc1 = filter_cloud(pc1, "x", -0.2, 0.2)
    pc2 = filter_cloud(pc2, "x", -0.3, 0.3)

    print "Ahora la primer nube tiene", points(pc1), "y la segunda tiene", points(pc2)

    icp_result = icp(pc1, pc2)

    print "ICP convergio?", icp_result.has_converged
    print "ICP score =", icp_result.score

    pc = icp_result.cloud
    print "La nube resultante tiene", points(pc), "puntos"

