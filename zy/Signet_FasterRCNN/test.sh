cd /home/zhongying/reference/segment/maskrcnn-benchmark
#conda activate pytorch
CUDA_VISIBLE_DEVICES='0' python tools/test_net.py --config-file \
"/home/zhongying/reference/segment/maskrcnn-benchmark/zy/Signet_FasterRCNN/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"
