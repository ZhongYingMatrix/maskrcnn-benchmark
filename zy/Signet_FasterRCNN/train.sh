cd /home/zhongying/reference/segment/maskrcnn-benchmark
#conda activate pytorch
#  
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train5",)' DATASETS.TEST '("Signet_ring_cell_test5",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/set5"
