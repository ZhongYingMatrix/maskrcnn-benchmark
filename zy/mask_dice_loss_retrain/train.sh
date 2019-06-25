cd /home/zhongying/reference/segment/maskrcnn-benchmark
export NGPUS=2
CUDA_VISIBLE_DEVICES='3' python tools/train_net.py \
--config-file '/home/zhongying/reference/segment/maskrcnn-benchmark/zy/mask_dice_loss_retrain/e2e_mask_rcnn_R_50_FPN_1x_caffe2_mask_dice_loss_retrain.yaml'

#-m torch.distributed.launch --nproc_per_node=$NGPUS \