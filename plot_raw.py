import os
import csv
import struct
import numpy as np
import matplotlib.pyplot as plt



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_FLOAT = 'Resnet_optimized.raw'   # 浮点raw路径
PATH_QUANT = 'Resnet_quantized.raw'   # 定点raw路径
PATH_CSV   = 'Resnet_raws.csv'        # csv路径
PATH_LOG   = 'logs/'                  # 图片保存路径
BIT        = 12                       # 8或12
DPI        = 200                      # 图片质量
BIN_NUM    = 60                       # 直方图格子数
IS_LOG     = False                    # 是否取对数



#-------------------------------------#
#       创建日志文件夹
#-------------------------------------#
path_distribution = PATH_LOG + 'distribution/'
path_error = PATH_LOG + 'error/'

if not os.path.exists(PATH_LOG):
    os.mkdir(PATH_LOG)

if not os.path.exists(path_distribution):
    os.mkdir(path_distribution)

if not os.path.exists(path_error):
    os.mkdir(path_error)



#-------------------------------------#
#       解析饱和点、Scale
#-------------------------------------#
dic = {}
with open(PATH_CSV, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        try:
            raw_id = int(row[0])
            sat_point = float(row[3])
            scale     = float(row[4])
            dic[raw_id] = [sat_point, scale]
        except:
            continue



#-------------------------------------#
#       解析权重
#-------------------------------------#
raw_addr_float = {}
raw_addr_quant = {}

raw_size_float = {}
raw_size_quant = {}

weights_float  = {}
weights_quant  = {}

raw_id   = []


whole_length_float = len(open(PATH_FLOAT, 'rb').read())
whole_length_quant = len(open(PATH_QUANT, 'rb').read())

binfile_float = open(PATH_FLOAT, 'rb')
binfile_quant = open(PATH_QUANT, 'rb')

layer_num = struct.unpack('<i', binfile_float.read(4))[0]
layer_num = struct.unpack('<i', binfile_quant.read(4))[0]


for i in range(layer_num):
    layer_id   = struct.unpack('<i', binfile_float.read(4))[0]
    layer_id   = struct.unpack('<i', binfile_quant.read(4))[0]
    start_addr_float = struct.unpack('<i', binfile_float.read(4))[0]
    start_addr_quant = struct.unpack('<i', binfile_quant.read(4))[0]
    raw_addr_float[layer_id] = start_addr_float
    raw_addr_quant[layer_id] = start_addr_quant
    raw_id.append(layer_id)


for i in range(len(raw_id)-1):
    raw_size_float[raw_id[i]] = raw_addr_float[raw_id[i+1]] - raw_addr_float[raw_id[i]]
    raw_size_quant[raw_id[i]] = raw_addr_quant[raw_id[i+1]] - raw_addr_quant[raw_id[i]]


raw_size_float[raw_id[i+1]] = whole_length_float - raw_addr_float[raw_id[i+1]]
raw_size_quant[raw_id[i+1]] = whole_length_quant - raw_addr_quant[raw_id[i+1]]


for i in raw_id:
    weights_float[i] = []
    weights_quant[i] = []
    for j in range(0, raw_size_float[i], 4):
        data = struct.unpack('<f', binfile_float.read(4))[0]
        weights_float[i].append(data)

    if BIT == 8:
        for j in range(0, raw_size_quant[i], 1):
            data = struct.unpack('<b', binfile_quant.read(1))[0]
            weights_quant[i].append(data)
    else:
        for j in range(0, raw_size_quant[i], 2):
            data = struct.unpack('<h', binfile_quant.read(2))[0]
            weights_quant[i].append(data)



#-------------------------------------#
#       画图
#-------------------------------------#
for k in weights_float.keys():

    sat_point = dic[k][0]
    scale     = dic[k][1]

    raw_float_all = np.array(weights_float[k])                                # 浮点权重
    raw_fixed_all = np.array(weights_quant[k]) * scale                        # 定点权重

    index        = (raw_float_all > sat_point) + (raw_float_all < -sat_point) # 记录被饱和的位置
    raw_float_qnt = raw_float_all[~index]                                     # 需要被量化的浮点权重
    raw_fixed_qnt = raw_fixed_all[~index]                                     # 被量化后的定点权重
    raw_float_sat = raw_float_all[index]                                      # 需要被饱和的浮点权重
    raw_fixed_sat = raw_fixed_all[index]                                      # 被饱和后的定点权重
    raw_float_all_mean = np.mean(np.abs(raw_float_all))                       # 浮点权重绝对平均值
    raw_float_all_sum  = np.sum(np.abs(raw_float_all))                        # 浮点权重绝对和
    diff_all  = np.abs(raw_float_all - raw_fixed_all)                         # 两权重的总误差
    diff_qnt  = np.abs(raw_float_qnt - raw_fixed_qnt)                         # 两权重的量化误差
    diff_sat  = np.abs(raw_float_sat - raw_fixed_sat)                         # 两权重的饱和误差

    error_all = np.sum(diff_all)                                              # 总误差的和
    error_qnt = np.sum(diff_qnt)                                              # 量化误差的和
    error_sat = np.sum(diff_sat)                                              # 饱和误差的和

    error_all_mean = error_all / raw_float_all_sum                            # 平均总误差
    error_qnt_mean = error_qnt / raw_float_all_sum                            # 平均量化误差
    error_sat_mean = error_sat / raw_float_all_sum                            # 平均饱和误差

    plt.hist(diff_qnt/raw_float_all_mean, bins = BIN_NUM, fill = None, log = IS_LOG)
    plt.xlabel('value of error')
    plt.ylabel('frequency of error')
    plt.savefig(path_error + str(k) + ".jpg", dpi = DPI)
    plt.cla()               # 清空画布
    # plt.show()
    raw_float_counts, raw_float_bins = np.histogram(raw_float_all, range = (min(raw_float_all+raw_fixed_all), max(raw_float_all+raw_fixed_all)), bins = BIN_NUM)
    raw_fixed_counts, raw_fixed_bins = np.histogram(raw_fixed_all, range = (min(raw_float_all+raw_fixed_all), max(raw_float_all+raw_fixed_all)), bins = BIN_NUM)
    raw_cover_counts = np.minimum(raw_float_counts, raw_fixed_counts)
    plt.hist(raw_float_bins[:-1], raw_float_bins, weights = raw_cover_counts, facecolor = 'greenyellow', edgecolor = 'black',  log = IS_LOG, label = 'overlap')
    plt.hist(raw_float_bins[:-1], raw_float_bins, weights = raw_float_counts, fill      = None         , edgecolor = 'black',  log = IS_LOG, label = 'float')
    plt.hist(raw_fixed_bins[:-1], raw_fixed_bins, weights = raw_fixed_counts, fill      = None         , edgecolor = 'red'  ,  log = IS_LOG, label = 'quant')
    plt.axvline(x = -sat_point, ls = "--", c = "green", label = 'sat_point')   # 负饱和点
    plt.axvline(x =  sat_point, ls = "--", c = "green")                        # 正饱和点

    error_info = 'all_error:{:.4f} \n quant_error: {:.4f} sat_error: {:.4e}'.format(error_all_mean, error_qnt_mean, error_sat_mean)
    plt.title(error_info)
    plt.xlabel('value of elements')
    plt.ylabel('frequency of elements')
    plt.legend()
    plt.savefig(path_distribution + str(k) + ".jpg", dpi = DPI)
    plt.cla()               # 清空画布
    # plt.show()