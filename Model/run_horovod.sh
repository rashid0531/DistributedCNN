#/home/rashid/mpi3/bin/mpirun -np 1 \
#    -H 127.0.0.1:1 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO \
#    -mca pml ob1 -mca btl ^openib \
#    /home/rashid/.conda/envs/dnn/bin/python /home/rashid/DistributedCNN/Model/distributed_train_horovod.py -i "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset" -gt "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset_GT/density_map/xavier" -ckpt_path "/discus/P2IRC/rashid/tf_ckpt"
#
#


HOROVOD_TIMELINE=/home/rashid/DistributedCNN/Model/horovod_timeline_2workers.json /home/rashid/mpi3/bin/mpirun -np 2 \
    -H 192.168.1.29:1,192.168.1.28:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    --mca oob_tcp_if_include br0 --mca btl_tcp_if_include br0 -x NCCL_SOCKET_IFNAME=br0 \
    /home/rashid/.conda/envs/dnn/bin/python /home/rashid/DistributedCNN/Model/distributed_train_horovod.py -i "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset" -gt "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset_GT/density_map/xavier" -ckpt_path "/discus/P2IRC/rashid/tf_ckpt" -nw 2


#/home/rashid/mpi3/bin/mpirun -np 4 \
#    -H 192.168.1.29:1,192.168.1.28:1,192.168.1.27:1,192.168.1.26:1 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    --mca oob_tcp_if_include br0 --mca btl_tcp_if_include br0 -x NCCL_SOCKET_IFNAME=br0 \
#    /home/rashid/.conda/envs/dnn/bin/python /home/rashid/DistributedCNN/Model/distributed_train_horovod.py -i "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset" -gt "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset_GT/density_map/xavier" -ckpt_path "/discus/P2IRC/rashid/tf_ckpt" -nw 4




#HOROVOD_TIMELINE=/home/rashid/DistributedCNN/Model/horovod_timeline_8workers.json /home/rashid/mpi3/bin/mpirun -np 8 \
#    -H 192.168.1.29:1,192.168.1.28:1,192.168.1.27:1,192.168.1.26:1,192.168.1.25:1,192.168.1.24:1,192.168.1.23:1,192.168.1.22:1 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    --mca oob_tcp_if_include br0 --mca btl_tcp_if_include br0 -x NCCL_SOCKET_IFNAME=br0 \
#    /home/rashid/.conda/envs/dnn/bin/python /home/rashid/DistributedCNN/Model/distributed_train_horovod.py -i "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset" -gt "/data/mounted_hdfs_path/user/hduser/mrc689/Sampled_Dataset_GT/density_map/xavier" -ckpt_path "/discus/P2IRC/rashid/tf_ckpt" -nw 8
