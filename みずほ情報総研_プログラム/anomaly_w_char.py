# coding: utf-8
import numpy as np
import sys
import csv
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

######## should be modified ########
width_wave = 5.12
delta_t    = 0.50

x_ticks_d1 = 50 #図の横軸の目盛間隔
x_ticks_d2 = 25 #図の横軸の目盛間隔
# x_ticks_d1 = 50 #図の横軸の目盛間隔
# x_ticks_d2 = 5 #図の横軸の目盛間隔

x_convt1_ax = 430
x_convt2_ax = 195
# x_convt1_ax = 461
# x_convt2_ax = 45

result_fig1 = 'result_trial_Adum_00_hon_axis_v2.png'
result_fig2 = 'result_trial_Adum_03_yo_axis_v2.png'

Adum_train = 'all_train_data_01_miharu_data_axis_5.12_5.12_on.csv'
Adum_test1 = 'test_data_00_hon_T1_axis_5.12_0.5_off.csv'
Adum_test2 = 'test_data_03_yo_T1_axis_5.12_0.5_off.csv'

# result_fig1 = 'result_trial_Bdum_05_hon_vertical_v2.png'
# result_fig2 = 'result_trial_Bdum_07_yo_vertical_v2.png'

# Adum_train = 'all_train_data_02_aratozawa_data_vertical_5.12_5.12_on.csv'
# Adum_test1 = 'test_data_40412-20080614-084345-T1_vertical_5.12_0.5_off.csv'
# Adum_test2 = 'test_data_40412-20080614-090143-T1_vertical_5.12_0.5_off.csv'


# args        = sys.argv
# width_wave  = float(args[1])
# delta_t     = float(args[2])
# x_ticks_d1  = int(args[3])
# x_ticks_d2  = int(args[4])
# iflg_d_kind = int(args[5])
# result_fig1 = args[6]
# result_fig2 = args[7]
# Adum_train  = args[8]
# Adum_test1  = args[9]
# Adum_test2  = args[10]
####################################

# Faeture00 :平均
# Faeture01 :分散
# Faeture02 :歪度
# Faeture03 :尖度
# Faeture04 :最小値
# Faeture05 :最大値
# Faeture06 :中央値
# Faeture07 :RMS
# Faeture08 :動的特徴量
# Faeture09 :パワーバンド（5~15Hz）
# Faeture10 :周波数領域エントロピー
# Faeture11 :エネルギー

def Calc_RMS(signal):
    a = signal * signal     # 二乗
    sum_a = np.sum(a)       # 総和
    sqrt_a = np.sqrt(sum_a) # 平方根
    RMS = np.mean(sqrt_a)   # 平均値
    return RMS

def Calc_LinearCoef(signal):
    X=np.arange(len(signal))
    X=np.reshape(X, (len(X), 1))

    Y=signal

    ModelLR = LinearRegression() # 線形回帰モデルのセット
    ModelLR.fit(X, Y) # パラメータ獲得（Xは時間、Yは信号値）
    
    return ModelLR.coef_[0]

def Calc_PowerBand(signal, sf, min_f, max_f):
    
    L = len(signal) # 信号長
    freq = np.linspace(0, sf, L) # 周波数
    yf = np.abs(fft(signal)) # 振幅スペクトル
    yf = yf * yf # パワースペクトル
    
    n1, n2 = [], []
    for i in range(1, int(L/2)): # <- 直流成分は探索範囲から除外
        n1.append(np.abs(freq[i] - min_f))
        n2.append(np.abs(freq[i] - max_f))
    min_id = np.argmin(n1) # min_fと最も近い周波数が格納された配列番号取得
    max_id = np.argmin(n2)# max_fと最も近い周波数が格納された配列番号取得
    
    PB = np.sum(yf[min_id:max_id])
                
    return PB

def Calc_FreqEnt(signal, sf):
    
    L = len(signal) # 信号長
    yf = np.abs(fft(signal)) # 振幅スペクトル
    yf = yf * yf # パワースペクトル

    a = 0
    for i in range(1, int(L/2)): # <- 直流成分除去 & ナイキスト周波数まで
        a = a + yf[i]

    E = 0
    for i in range(1, int(L/2)): # <- 直流成分除去 & ナイキスト周波数まで
        if yf[i] == 0.0:
            yf[i] = 1.0e-15
        E = E + (yf[i]/a) * np.log2(yf[i]/a)
    E = -E

    return E

def Calc_Energy(signal, sf):
    
    L = len(signal) # 信号長
    yf = np.abs(fft(signal)) # 振幅スペクトル
    yf = yf * yf # パワースペクトル

    En = np.mean(yf[1:int(L/2)]) # <- 直流成分除去 & ナイキスト周波数まで
    
    return En

def FeatureVector(signal, sf):
    FV = []
    FV.append(np.mean(signal))                      # j=0 平均
    FV.append(np.var(signal))                       # j=1 分散
    FV.append(st.skew(signal))                      # j=2 歪度
    FV.append(st.kurtosis(signal))                  # j=3 尖度
    FV.append(np.min(signal))                       # j=4 最小値
    FV.append(np.max(signal))                       # j=5 最大値
    FV.append(np.median(signal))                    # j=6 中央値
    FV.append(Calc_RMS(signal))                     # j=7 RMS
    FV.append(Calc_LinearCoef(signal))              # j=8 動的特徴量
    FV.append(Calc_PowerBand(signal, sf,   5, 15))  # j=9 PB1
    FV.append(Calc_FreqEnt(signal, sf))              # j=10 周波数領域エントロピー
    FV.append(Calc_Energy(signal, sf))               # j=11 エネルギー
    return FV

def calc_FV_waves_norm(input_csv, outcsv1, outcsv2):
    with open(outcsv1, 'w') as f1:
        writer1 = csv.writer(f1, lineterminator='\n')
        sig_all  = np.loadtxt(input_csv, delimiter=',')
        num_wavelet = len(sig_all)
        num_feature = 12
        FV_all_wave = []
        for i in range(0, num_wavelet):
            sig_list = []
            sig_list = sig_all[i]
            writer1.writerow(FeatureVector(sig_list,100))
            FV_all_wave.append(FeatureVector(sig_list,100))

        arr_FV_waves = np.array(FV_all_wave)

        Zscore = []
        for j in range(0, num_feature):
            feature_ave = np.mean(arr_FV_waves[:,j])
            feature_std = np.std(arr_FV_waves[:,j])
            Zscore.append([feature_ave, feature_std])

        arr_Zscore = np.array(Zscore)

        for i in range(0, num_wavelet):
            for j in range(0, num_feature):
                arr_FV_waves[i,j] = (arr_FV_waves[i,j] - arr_Zscore[j,0]) / arr_Zscore[j,1]
    
        np.savetxt(outcsv2, arr_FV_waves, delimiter=',', fmt='%.5f')

    return arr_FV_waves

out_csv_file1f = 'feature_vector_' + Adum_train
out_csv_file1n = 'normalization_'  + Adum_train
out_csv_file2f = 'feature_vector_' + Adum_test1
out_csv_file2n = 'normalization_'  + Adum_test1
out_csv_file3f = 'feature_vector_' + Adum_test2
out_csv_file3n = 'normalization_'  + Adum_test2

arr_FV_waves_A_train = calc_FV_waves_norm(Adum_train, out_csv_file1f, out_csv_file1n)
arr_FV_waves_A_test1 = calc_FV_waves_norm(Adum_test1, out_csv_file2f, out_csv_file2n)
arr_FV_waves_A_test2 = calc_FV_waves_norm(Adum_test2, out_csv_file3f, out_csv_file3n)

# One Class SVM(original :nu=0.05)
#clf_OCSVM = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma='scale') # OCSVMのパラメータ設定
clf_OCSVM = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma='scale') # OCSVMのパラメータ設定
clf_OCSVM.fit(arr_FV_waves_A_train) # フィット（識別平面の計算）
y_pred_test1_OCSVM = clf_OCSVM.predict(arr_FV_waves_A_test1) # 分類結果1
y_pred_test2_OCSVM = clf_OCSVM.predict(arr_FV_waves_A_test2) # 分類結果2
#print(y_pred_train_OCSVM)

# Isolation Forest(original :contamination=0.05, max_features=default)
#clf_ISFOR = IsolationForest(n_estimators=1000, max_samples=100, contamination=0.05, random_state=200) # Isolation Forestのパラメータ設定
clf_ISFOR = IsolationForest(n_estimators=1000, max_samples=100, contamination=0.01, max_features=3 ,random_state=200) # Isolation Forestのパラメータ設定
clf_ISFOR.fit(arr_FV_waves_A_train)
y_pred_test1_ISFOR = clf_ISFOR.predict(arr_FV_waves_A_test1)
y_pred_test2_ISFOR = clf_ISFOR.predict(arr_FV_waves_A_test2)
#print(y_pred_train_ISFOR)

# k-Nearest Neighnors
clf_KNENE = NearestNeighbors(n_neighbors=1) # k-Nearest Neighnorsのパラメータ設定
clf_KNENE.fit(arr_FV_waves_A_train)
d_knn_test1 = clf_KNENE.kneighbors(arr_FV_waves_A_test1)[0]
d_knn_maxt1 = np.max(d_knn_test1)
d_knn_test1 = d_knn_test1 / d_knn_maxt1
d_knn_test2 = clf_KNENE.kneighbors(arr_FV_waves_A_test2)[0]
d_knn_maxt2 = np.max(d_knn_test2)
d_knn_test2 = d_knn_test2 / d_knn_maxt2
#print(d_knn)

# plot1
x_ID1    = np.arange(0, len(y_pred_test1_OCSVM), 1)
x_convt1 = [(delta_t*i)+(width_wave/2.0) for i in x_ID1] #各部分波形を時間に変換

fig1 = plt.figure()
#plt.figure(figsize=(30, 30)) # 描画領域の横幅, 縦幅

# OCSVM result
ax1  = fig1.add_subplot(311)
ax1.scatter(x_convt1, y_pred_test1_OCSVM, s=10, c='b', marker="o")
#ax1.set_xticks(np.arange(0, max(x_convt1), x_ticks_d1))
#ax1.set_xticklabels(np.arange(0, max(x_convt1), x_ticks_d1),fontsize=6)    
ax1.set_xticks(np.arange(0, x_convt1_ax, x_ticks_d1))
ax1.set_xticklabels(np.arange(0, x_convt1_ax, x_ticks_d1),fontsize=6)   
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels([-1, 0, 1], fontsize=6)

# ISFOR result
ax1  = fig1.add_subplot(312)
ax1.scatter(x_convt1, y_pred_test1_ISFOR, s=10, c='g', marker="o")
#ax1.set_xticks(np.arange(0, max(x_convt1), x_ticks_d1))
#ax1.set_xticklabels(np.arange(0, max(x_convt1), x_ticks_d1),fontsize=6)    
ax1.set_xticks(np.arange(0, x_convt1_ax, x_ticks_d1))
ax1.set_xticklabels(np.arange(0, x_convt1_ax, x_ticks_d1),fontsize=6)   
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels([-1, 0, 1], fontsize=6)

# k-NN result
ax1  = fig1.add_subplot(313)
ax1.scatter(x_convt1, d_knn_test1, s=10, c='orange', marker="o")
#ax1.set_xticks(np.arange(0, max(x_convt1), x_ticks_d1))
#ax1.set_xticklabels(np.arange(0, max(x_convt1), x_ticks_d1),fontsize=6)    
ax1.set_xticks(np.arange(0, x_convt1_ax, x_ticks_d1))
ax1.set_xticklabels(np.arange(0, x_convt1_ax, x_ticks_d1),fontsize=6)   
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels([0, 0.5, 1], fontsize=6)

fig1.savefig(result_fig1, bbox_inches='tight', pad_inches=0.1)

# plot2
x_ID2    = np.arange(0, len(y_pred_test2_OCSVM), 1)
x_convt2 = [(delta_t*i)+(width_wave/2.0) for i in x_ID2] #各部分波形を時間に変換

fig2 = plt.figure()
#plt.figure(figsize=(50, 50)) # 描画領域の横幅, 縦幅

# OCSVM result
ax2  = fig2.add_subplot(311)
ax2.scatter(x_convt2, y_pred_test2_OCSVM, s=10, c='b', marker="o")
#ax2.set_xticks(np.arange(0, max(x_convt2), x_ticks_d2))
#ax2.set_xticklabels(np.arange(0, max(x_convt2), x_ticks_d2),fontsize=6)
ax2.set_xticks(np.arange(0, x_convt2_ax, x_ticks_d2))
ax2.set_xticklabels(np.arange(0, x_convt2_ax, x_ticks_d2),fontsize=6)   
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels([-1, 0, 1], fontsize=6)

# ISFOR result
ax2  = fig2.add_subplot(312)
ax2.scatter(x_convt2, y_pred_test2_ISFOR, s=10, c='g', marker="o")
#ax2.set_xticks(np.arange(0, max(x_convt2), x_ticks_d2))
#ax2.set_xticklabels(np.arange(0, max(x_convt2), x_ticks_d2),fontsize=6)
ax2.set_xticks(np.arange(0, x_convt2_ax, x_ticks_d2))
ax2.set_xticklabels(np.arange(0, x_convt2_ax, x_ticks_d2),fontsize=6)   
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels([-1, 0, 1], fontsize=6)

# k-NN result
ax2  = fig2.add_subplot(313)
ax2.scatter(x_convt2, d_knn_test2, s=10, c='orange', marker="o")
#ax2.set_xticks(np.arange(0, max(x_convt2), x_ticks_d2))
#ax2.set_xticklabels(np.arange(0, max(x_convt2), x_ticks_d2),fontsize=6)
ax2.set_xticks(np.arange(0, x_convt2_ax, x_ticks_d2))
ax2.set_xticklabels(np.arange(0, x_convt2_ax, x_ticks_d2),fontsize=6)   
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticklabels([0, 0.5, 1], fontsize=6)

fig2.savefig(result_fig2, bbox_inches='tight', pad_inches=0.1)

# output
out_OCSVM_t1   = 'OCSVM_' + Adum_test1
out_OCSVM_t2   = 'OCSVM_' + Adum_test2
out_ISFOR_t1   = 'ISFOR_' + Adum_test1
out_ISFOR_t2   = 'ISFOR_' + Adum_test2
out_KNENE_t1   = 'KNENE_' + Adum_test1
out_KNENE_t2   = 'KNENE_' + Adum_test2

arr1 = np.array([x_convt1, y_pred_test1_OCSVM]).T
np.savetxt(out_OCSVM_t1, arr1, delimiter=',', fmt='%.5f')
arr2 = np.array([x_convt2, y_pred_test2_OCSVM]).T
np.savetxt(out_OCSVM_t2, arr2, delimiter=',', fmt='%.5f')

arr3 = np.array([x_convt1, y_pred_test1_ISFOR]).T
np.savetxt(out_ISFOR_t1, arr3, delimiter=',', fmt='%.5f')
arr4 = np.array([x_convt2, y_pred_test2_ISFOR]).T
np.savetxt(out_ISFOR_t2, arr4, delimiter=',', fmt='%.5f')

arr5 = np.array([x_convt1, d_knn_test1.reshape(len(x_convt1))]).T
np.savetxt(out_KNENE_t1, arr5, delimiter=',', fmt='%.5f')
arr6 = np.array([x_convt2, d_knn_test2.reshape(len(x_convt2))]).T
np.savetxt(out_KNENE_t2, arr6, delimiter=',', fmt='%.5f')

