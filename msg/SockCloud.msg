# One detected sock's 3D point-cloud blob, expressed in the action's target_frame.
sensor_msgs/PointCloud2 cloud         # points belonging to the sock
geometry_msgs/PointStamped centroid   # blob centroid (target_frame)
geometry_msgs/Vector3 dimensions      # axis-aligned bounding-box size (m)
string label                          # detection class label
float32 score                         # detection confidence
