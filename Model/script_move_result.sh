#!/bin/bash

#mv out.txt exm_per_sec.txt timeline nohup.out /discus/P2IRC/rashid/thesis_results/2_paramserver/8_workers/worker_1/

for var in {0..8}
do
    rsync -rvP /home/rashid/DistributedCNN/Model/distributed_train_horovod.py /home/rashid/DistributedCNN/Model/prepare.py rashid@discus-spark$var:/home/rashid/DistributedCNN/Model/

done



#mv /discus/P2IRC/rashid/tf_ckpt /discus/P2IRC/rashid/thesis_results/1_paramserver/4_workers/ 
