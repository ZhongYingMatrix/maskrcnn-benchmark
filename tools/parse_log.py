import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_log(args):
    itr, loss, loss_c, loss_br, loss_m, loss_o, loss_rbr=[], [], [], [], [], [], []
    def extract_value(line_lst,index):
        if args.smooth:
            return float(line_lst[index+1][1:-1])
        else:
            return float(line_lst[index])
    #smooth = 1 if args.smooth else 0
    with open(args.log_file,'r') as f:
        for line in f.readlines():
            line_lst = line.split()
            #if len(line_lst)==39 and line_lst[2] == 'maskrcnn_benchmark.trainer':
            if len(line_lst)>3 and line_lst[2] == 'maskrcnn_benchmark.trainer' and 'iter:' in line_lst:
                base_index =  line_lst.index('iter:') + 1
                itr.append(float(line_lst[base_index]))
                loss.append(extract_value(line_lst,base_index + 2))
                loss_c.append(extract_value(line_lst,base_index + 5))
                loss_br.append(extract_value(line_lst,base_index + 8))
                loss_m.append(extract_value(line_lst,base_index + 11))
                loss_o.append(extract_value(line_lst,base_index + 14))
                loss_rbr.append(extract_value(line_lst,base_index + 17))
    return np.array(itr), np.array(loss), np.array(loss_c), \
        np.array(loss_br), np.array(loss_m), np.array(loss_o), np.array(loss_rbr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parsing log")
    parser.add_argument(
        "--log-file",
        default="./log.txt",
        metavar="FILE",
        help="path to log file",
        type=str,
    )
    parser.add_argument(
      "--smooth",
      dest="smooth",
      help="use smooth data",
      action="store_true",
    )
    args = parser.parse_args()
    #print(args)

    itr, loss, loss_c, loss_br, loss_m, loss_o, loss_rbr = get_log(args)

    #print(itr, loss)#, loss_c, loss_br, loss_m, loss_o, loss_rbr)
    plt.subplot(321)
    plt.plot(itr,loss,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.xlim(xmin=100)
    plt.subplot(322)
    plt.plot(itr,loss_c,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss_classifier')
    plt.xlim(xmin=100)
    plt.subplot(323)
    plt.plot(itr,loss_br,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss_box_reg')
    plt.xlim(xmin=100)
    plt.subplot(324)
    plt.plot(itr,loss_m,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss_mask')
    plt.xlim(xmin=100)
    plt.subplot(325)
    plt.plot(itr,loss_o,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss_objectness')
    plt.xlim(xmin=100)
    plt.subplot(326)
    plt.plot(itr,loss_rbr,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('loss_rpn_box_reg')
    plt.xlim(xmin=100)
    plt.show()

    
    
