import os
import csv
import numpy as np
import matplotlib.pyplot as plt



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_FLOAT = 'o/'                # 浮点ftmp路径
PATH_QUANT = 'q/'                # 量化后ftmp路径
PATH_CSV   = 'YoloV5_ftmps.csv'  # csv路径
PATH_LOG   = 'logs/'             # 图片保存路径
DPI        = 200                 # 图片质量
BIN_NUM    = 60                  # 直方图格子数
IS_LOG     = False               # 是否取对数



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
#       解析饱和点
#-------------------------------------#
dic = {}
with open(PATH_CSV, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        try:
            ftmp_id = int(row[0])
            sat_point = float(row[3])
            dic[ftmp_id] = sat_point
        except:
            continue



#-------------------------------------#
#       画图
#-------------------------------------#
for file_name in os.listdir(PATH_FLOAT):
    file_id      = int(file_name.split('.')[0])                             # 特征图文件id
    sat_point    = dic[file_id]                                             # 该特征图的饱和点
    fm_float_all = np.fromfile(PATH_FLOAT + file_name, np.float32)          # 浮点特征图
    fm_fixed_all = np.fromfile(PATH_QUANT + file_name, np.float32)          # 定点特征图
    index        = (fm_float_all > sat_point) + (fm_float_all < -sat_point) # 记录被饱和的位置
    fm_float_qnt = fm_float_all[~index]                                     # 需要被量化的浮点特征图
    fm_fixed_qnt = fm_fixed_all[~index]                                     # 被量化后的定点特征图
    fm_float_sat = fm_float_all[index]                                      # 需要被饱和的浮点特征图
    fm_fixed_sat = fm_fixed_all[index]                                      # 被饱和后的定点特征图
    fm_float_all_mean = np.mean(np.abs(fm_float_all))                       # 浮点特征图绝对平均值
    fm_float_all_sum  = np.sum(np.abs(fm_float_all))                        # 浮点特征图绝对和
    diff_all  = np.abs(fm_float_all - fm_fixed_all)                         # 两特征图的总误差
    diff_qnt  = np.abs(fm_float_qnt - fm_fixed_qnt)                         # 两特征图的量化误差
    diff_sat  = np.abs(fm_float_sat - fm_fixed_sat)                         # 两特征图的饱和误差

    error_all = np.sum(diff_all)                                            # 总误差的和
    error_qnt = np.sum(diff_qnt)                                            # 量化误差的和
    error_sat = np.sum(diff_sat)                                            # 饱和误差的和

    error_all_mean = error_all / fm_float_all_sum                           # 平均总误差
    error_qnt_mean = error_qnt / fm_float_all_sum                           # 平均量化误差
    error_sat_mean = error_sat / fm_float_all_sum                           # 平均饱和误差

    plt.hist(diff_qnt/fm_float_all_mean, bins = BIN_NUM, fill = None, log = IS_LOG)
    plt.xlabel('value of error')
    plt.ylabel('frequency of error')
    plt.savefig(path_error + str(file_id) + ".jpg", dpi = DPI)
    plt.cla()               # 清空画布
    # plt.show()
    fm_float_counts, fm_float_bins = np.histogram(fm_float_all, range = (min(fm_float_all+fm_fixed_all), max(fm_float_all+fm_fixed_all)), bins = BIN_NUM)
    fm_fixed_counts, fm_fixed_bins = np.histogram(fm_fixed_all, range = (min(fm_float_all+fm_fixed_all), max(fm_float_all+fm_fixed_all)), bins = BIN_NUM)
    fm_cover_counts = np.minimum(fm_float_counts, fm_fixed_counts)
    plt.hist(fm_float_bins[:-1], fm_float_bins, weights = fm_cover_counts, facecolor = 'greenyellow', edgecolor = 'black',  log = IS_LOG, label = 'overlap')
    plt.hist(fm_float_bins[:-1], fm_float_bins, weights = fm_float_counts, fill      = None         , edgecolor = 'black',  log = IS_LOG, label = 'float')
    plt.hist(fm_fixed_bins[:-1], fm_fixed_bins, weights = fm_fixed_counts, fill      = None         , edgecolor = 'red'  ,  log = IS_LOG, label = 'quant')
    plt.axvline(x = -sat_point, ls = "--", c = "green", label = 'sat_point')   # 负饱和点
    plt.axvline(x =  sat_point, ls = "--", c = "green")                        # 正饱和点

    error_info = 'all_error:{:.4f} \n quant_error: {:.4f} sat_error: {:.4e}'.format(error_all_mean, error_qnt_mean, error_sat_mean)
    plt.title(error_info)
    plt.xlabel('value of elements')
    plt.ylabel('frequency of elements')
    plt.legend()
    plt.savefig(path_distribution + str(file_id) + ".jpg", dpi = DPI)
    plt.cla()               # 清空画布
    # plt.show()