cd /home/zhongying/reference/segment/maskrcnn-benchmark
#conda activate pytorch
#  
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --skip-test --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train1",)' DATASETS.TEST '("Signet_ring_cell_test1",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/set1"

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --skip-test --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train2",)' DATASETS.TEST '("Signet_ring_cell_test2",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/set2"

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --skip-test --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train3",)' DATASETS.TEST '("Signet_ring_cell_test3",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/set3"

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --skip-test --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train4",)' DATASETS.TEST '("Signet_ring_cell_test4",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/set4"

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --skip-test --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
DATASETS.TRAIN '("Signet_ring_cell_train5",)' DATASETS.TEST '("Signet_ring_cell_test5",)'  \
OUTPUT_DIR "/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FRCNN_Res50/set5"
