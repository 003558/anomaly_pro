# =====================================================================================================
# ダム異常検知AI　応用ケース  異常検知
#   同一ダム同種データ、同一ダム異種データの両方（観測時系列）に対応
#      method: (1)KNN, (2)Isolation Forest, (3)kmeans 
#　2022.01.14 ver01 N.Tagashira ツール用に再構成
#  2022.05.10 ver02 N.Tagashira 同種、異種データの両方に対応,　学習期間設定に対応
#  2022.05.14 ver03 N.Tagashira 前後にlookback_timestep分の平均をとっていたのを、lookback_timesteps/2ずらすに変更
# =====================================================================================================

# ライブラリの読み込み
import configparser  # iniファイル形式のconfigを読み込む
# import copy
import datetime      # 時間計測に必要なライブラリ
# import glob          # ファイル名取得や削除に必要なライブラリ
import json          # iniファイルからリスト形式を読む場合にjson.loadsを使用する
import math
import os            # pathの連結等に必要なライブラリ
# import pickle        # オブジェクトをバイナリで保存、ロードするのに必要なライブラリ
import random        # random.seedを固定用
import sys           # コマンドライン引数を読み込むのに必要なライブラリ
import shutil        # ファイルコピーに必要なライブラリ
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

# 図の描画に必要なライブラリ
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.ioff()

# # 図の描画に必要なライブラリ（GoogleColabo用）
# import japanize_matplotlib #日本語化matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# japanize_matplotlib.japanize()

# 多次元配列、データフレームの操作に必要なライブラリ
import numpy as np
import pandas as pd
from scipy.spatial import distance

# # 状態空間モデル用のライブラリ
# import statsmodels.api as sm

# 機械学習用ライブラリ
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    # cofigファイルを読み込む
    configfile = sys.argv[1]  # コマンドライン引数の1つ目
    cf = Set_config(f=configfile)
    
    '''共通設定'''  
    # output, modelフォルダの作成
    Create_Update_folder(cf.output_path)
    # configfileの保存
    shutil.copy2(configfile, f'{cf.output_path}/')
    # random性を固定する
    Set_seed(cf.flg_random)

    num_cols_x1 = len(cf.cols_x1)
    num_cols_x2 = len(cf.cols_x2)

    '''データの入力'''
    csv_path = os.path.join(cf.input_path, cf.infile)
    df = pd.read_csv(csv_path , header=[0], index_col=0, parse_dates=True, encoding='shift-jis')
    
    '''データの前処理'''
    # 説明変数と目的変数の抽出
    if cf.flg_monthlydata == 1:
        cols_x2_intp = [f'{cf.cols_x2[i]}_LI_月平均' for i in range(num_cols_x2)]
        cols_x1_intp = [f'{cf.cols_x1[i]}_LI_月平均' for i in range(num_cols_x1)]    
    else:
        cols_x2_intp = [f'{cf.cols_x2[i]}_LI' for i in range(num_cols_x2)]
        cols_x1_intp = [f'{cf.cols_x1[i]}_LI' for i in range(num_cols_x1)]
    
    if num_cols_x2 == 0:
        dftgt = df[cols_x1_intp].copy()
    else:
        dftgt_x = df[cols_x2_intp].copy()
        dftgt_y = df[cols_x1_intp].copy()
        dftgt = pd.concat([dftgt_x, dftgt_y], axis=1)
        del dftgt_x, dftgt_y
    
    # 欠損値の削除
    dftgt = dftgt.dropna(how='any')  # TimeSeriesの場合は、NAの観測孔があってはダメ

    num_variables = dftgt.shape[1]
    
    # 訓練データ、allデータ作成、時間区切りデータ作成
    if cf.flg_trainterm==1:
        dftrain = dftgt[(dftgt.index >= cf.train_term[0]) & (dftgt.index <= cf.train_term[1])].copy()
    else:
        dftrain = dftgt.copy()
    dfall = dftgt.copy()

    # num_samples_train = dftgt_train.shape[0]
    # num_variables = dftgt_train.shape[1]
    # # tm = [dfterm_mean_train.index[0], dfterm_mean_train.index[num_samples_train-1]]

    # dftrain = dftgt.copy()
    # # 時間区切りデータ作成
    # tm = [dfterm_mean.index[0], dfterm_mean.index[num_samples-1]]

    # 正規化
    if cf.normalization_type != 'none':
        # インスタンス作成
        if cf.normalization_type == 'standard':
            norm = StandardScaler()
        elif cf.normalization_type == 'minmax':
            norm = MinMaxScaler()
        # fitting
        norm.fit(dftrain)
        # 正規化（train）
        _nptrainnm = norm.transform(dftrain)
        dftrainnm  = pd.DataFrame(_nptrainnm, index=dftrain.index, columns=dftrain.columns)
        # 正規化（all）
        _npallnm = norm.transform(dfall)
        dfallnm  = pd.DataFrame(_npallnm, index=dfall.index, columns=dfall.columns)
        del _nptrainnm, _npallnm 

    else:
        dftrainnm = dftrain.copy()  # 正規化しない場合は、元データを正規化後データにする
        dfallnm   = dfall.copy()
    
    # データシフト
    # trainデータ
    _nptrainnm = dftrainnm.to_numpy().reshape(-1, num_variables)
    nptrainnm_ts = Shift_Lookforward_fromCurrent(_nptrainnm, cf.lookback_timesteps)
    # allデータ
    _npallnm = dfallnm.to_numpy().reshape(-1, num_variables)
    npallnm_ts = Shift_Lookforward_fromCurrent(_npallnm, cf.lookback_timesteps)
    del _nptrainnm, _npallnm
       
    '''異常検知分析 （異常度スコアを計算）'''
    # kNNの場合
    if cf.method == 'kNN':
        # インスタンス作成・fitting
        nn = NearestNeighbors(n_neighbors=cf.kNN_k)
        nn.fit(nptrainnm_ts)
        # 距離計算
        dist_train = nn.kneighbors(nptrainnm_ts)[0]
        dist_train = np.max(dist_train, axis=1)
        dist_all = nn.kneighbors(npallnm_ts)[0]
        dist_all = np.max(dist_all, axis=1)

        del nn, nptrainnm_ts, npallnm_ts
    
    # Isolation Forestの場合
    elif cf.method == 'IsolationForest':
        # インスタンス作成・fitting
        # rs = np.random.RandomState(123)
        clf = IsolationForest(n_estimators=100, max_samples='auto', random_state=123)    #random_state以外はデフォルト採用  random_state=rs
        clf.fit(nptrainnm_ts)
        # 距離計算
        dist_train = clf.decision_function(nptrainnm_ts) * -1 + 0.5  # https://stats.stackexchange.com/questions/335274/scikit-learn-isolationforest-anomaly-score
        dist_all   = clf.decision_function(npallnm_ts) * -1 + 0.5  # https://stats.stackexchange.com/questions/335274/scikit-learn-isolationforest-anomaly-score
        # dist_train[dist_train<0] = 0

        del clf, nptrainnm_ts, npallnm_ts  # rs
        
    # k-meansの場合
    elif cf.method == 'kmeans':
        # インスタンス作成・fitting
        km = KMeans(n_clusters=cf.kmeans_n)  # 2 -> cf.kmeans_nに変更
        cls = km.fit(nptrainnm_ts)           # プログラム最新版はnptrainnm_tsを使っている（最新版は、正規化しないパターンもあり）
        # 所属グループ予測
        pred_train = cls.fit_predict(nptrainnm_ts)
        pred_all = cls.predict(npallnm_ts)
        # 正常グループと異常グループの設定
        num_cls = []
        for i in range(cf.kmeans_n):
            num_cls.append(len(pred_train[pred_train==i]))
        id_normal = num_cls.index(max(num_cls))
        id_abnormal = num_cls.index(min(num_cls))
        # 正常グループ、異常グループの重心設定
        centers_normal   = cls.cluster_centers_[id_normal]
        centers_abnormal = cls.cluster_centers_[id_abnormal]
        # 距離計算
        dist_train = calc_kmeans_dist(centers_normal, centers_abnormal, nptrainnm_ts)
        dist_all   = calc_kmeans_dist(centers_normal, centers_abnormal, npallnm_ts)
        # dist_base = distance.euclidean(centers_normal.tolist(), centers_abnormal.tolist())
        # dist_all = []
        # for i in range(len(npallnm_ts)):
        #     _dist_normal   = distance.euclidean(npallnm_ts[i].tolist(), centers_normal.tolist())
        #     _dist_abnormal = distance.euclidean(npallnm_ts[i].tolist(), centers_abnormal.tolist())

        #     # ベン図のAかつBの部分
        #     if _dist_normal <= dist_base and _dist_abnormal <= dist_base:
        #         _dist = _dist_normal - _dist_abnormal
        #     else:
        #         # AかつBの部分を除いた範囲のうち、正常からの距離が遠い（異常の重心付近）データ群
        #         if _dist_normal >= _dist_abnormal:
        #             _dist = _dist_normal - 0
        #         # AかつBの部分を除いた範囲のうち、異常からの距離が遠い（正常の重心付近）データ群
        #         else:
        #             _dist = 0 - _dist_abnormal

        #     # if _dist_abnormal < 0:
        #     #     _dist_abnormal = 0
        #     dist_all.append(_dist/dist_base)

        del km, cls, pred_train, pred_all, num_cls, id_normal, id_abnormal, centers_normal, centers_abnormal, nptrainnm_ts, npallnm_ts

    '''出力'''
    # 出力_csv
    
    # if cf.lookback_timesteps >= 3:
    #     # 偶数の場合
    #     if cf.lookback_timesteps % 2 == 0:
    #         dist_all_ext = np.append(np.full(int(my_round(cf.lookback_timesteps/2-1)), np.nan), dist_all)           # 前方にnanを(時間窓数/2-1)個つける
    #         dist_all_ext = np.append(dist_all_ext, np.full(int(my_round(cf.lookback_timesteps/2)), np.nan))         # 前方にnanを(時間窓数/2)個つける
    #     # 奇数の場合
    #     else:
    #         dist_all_ext = np.append(np.full(int(my_round((cf.lookback_timesteps-1)/2)), np.nan), dist_all)         # 前方にnanを(時間窓数-1)/2個つける
    #         dist_all_ext = np.append(dist_all_ext, np.full(int(my_round((cf.lookback_timesteps-1)/2)), np.nan))     # 後方にnanを(時間窓数-1)/2個つける
    
    dist_all_ext = np.append(np.full(cf.lookback_timesteps-1, np.nan), dist_all)         # 前方にnanを時間窓数-1分つける
    dist_all_ext = np.append(dist_all_ext, np.full(cf.lookback_timesteps-1, np.nan))     # 後方にnanを時間窓数-1分つける
    dist_all_ext = dist_all_ext.reshape(-1, 1)
    dist_all_ext  = Shift_Lookforward_fromCurrent(dist_all_ext, cf.lookback_timesteps)   # 時間窓分スライドさせる
    dist_all_ext  = np.nanmean(dist_all_ext, axis=1)                                      # 横方向に平均値をとる

    dftgt['異常度'] = dist_all_ext
    # 閾値設定
    if cf.flg_trainterm == 1:
        dftrain = dftgt[(dftgt.index >= cf.train_term[0]) & (dftgt.index <= cf.train_term[1])].copy()
        cf.threshold = max(dftrain['異常度'])
    dftgt['閾値']   = cf.threshold
    dist_all_ext_NG = np.where(dist_all_ext >  cf.threshold, dist_all_ext, np.nan) # 閾値超過データ
    dist_all_ext_OK = np.where(dist_all_ext <= cf.threshold, dist_all_ext, np.nan) # 閾値以下データ
    dftgt['異常度_閾値超過'] = dist_all_ext_NG
    dftgt['異常度_閾値以下'] = dist_all_ext_OK
    # dfterm = pd.merge(dfterm, dfterm_mean['anomaly_dist'], left_index=True, right_index=True, how='left')
    # dfterm.to_csv(f'{cf.output_path}/{cf.headname}_01term_dfprd.csv', encoding='shift-jis')
    dftgt.to_csv(f'{cf.output_path}/{cf.title}_01dfprd.csv', encoding='shift-jis')
    del dist_all, dist_all_ext, dist_all_ext_NG, dist_all_ext_OK

    # 出力_時系列グラフ_all＋異常度距離
    if cf.flg_monthlydata == 1:
        unit_name = '月'
    else:
        unit_name = '日'

    if cf.flg_samedata == 1:
        nrows = num_cols_x2 + 2
    else:
        nrows = num_cols_x1 + 1

    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 3*nrows), sharex=True, squeeze=False)

    if cf.flg_samedata == 1:
        # 説明変数の描画
        for i in range(num_cols_x2):
            ax[i, 0].plot(dftgt.index.to_numpy(), dftgt[cols_x2_intp[i]].to_numpy(), linestyle='solid', label=f'{cf.cols_x2[i]}_{unit_name}平均値') # 平均値
            ax[i, 0].set_ylabel(cf.cols_x2[i])
            ax[i, 0].legend()

        # 目的変数の描画
        # flg = 0; handles = []; labels = []
        for i in range(num_cols_x1):
            ax[num_cols_x2, 0].plot(dftgt.index.to_numpy(), dftgt[cols_x1_intp[i]].to_numpy(), linestyle='solid', label=f'{cf.cols_x1[i]}_{unit_name}平均値', color='gray') # 平均値
        #     dftgt_y = dftgt[[cols_x1_intp[i], 'anomaly_dist']].copy()
        #     dftgt_y = dftgt_y[dftgt_y['anomaly_dist']>cf.threshold]    
        #     if len(dftgt_y) > 0:
        #         point_abnormal, = ax[num_cols_x2, 0].plot(dftgt_y.index.to_numpy(), dftgt_y[cols_x1_intp[i]].to_numpy(), linestyle='None', label='異常疑いがあるデータ', marker='o', markersize=2, color='red')
        #         if flg == 0:
        #             handles.append(point_abnormal)
        #             labels.append('異常疑いがある区間')
        #             flg += 1
        # if len(handles) > 0:
        #     ax[num_cols_x2, 0].legend(handles, labels)
        ax[num_cols_x2, 0].set_ylabel(cf.x1label)
    else:
        # 目的変数の描画
        for i in range(num_cols_x1):
            ax[i, 0].plot(dftgt.index.to_numpy(), dftgt[cols_x1_intp[i]].to_numpy(), linestyle='solid', label=f'{cf.cols_x1[i]}_{unit_name}平均値', color='gray')
            ax[i, 0].set_ylabel(cf.cols_x1[i])
    
    # 異常度距離の描画
    ax[nrows-1, 0].plot(dftgt.index.to_numpy(), dftgt[['異常度_閾値以下']].to_numpy(), linestyle='None', marker='o', markersize=4, color='none', markeredgecolor='gray', label='閾値以下')
    ax[nrows-1, 0].plot(dftgt.index.to_numpy(), dftgt[['異常度_閾値超過']].to_numpy(), linestyle='None', marker='o', markersize=4, color='none', markeredgecolor='red', label='閾値超過')
    ax[nrows-1, 0].set_ylabel('異常度')
    ax[nrows-1, 0].legend(loc='upper left')
    ax[nrows-1, 0].axhline(y=cf.threshold, color='red', linestyle='dashed')

    # x軸、タイトル等の描画
    for i in range(nrows):
        ax[i, 0].grid(which='both', axis='both')
        if cf.flg_trainterm==1:
            ax[i, 0].axvline(x=cf.train_term[0], color='magenta', linestyle='dotted')
            ax[i, 0].axvline(x=cf.train_term[1], color='magenta', linestyle='dotted')
    ax[nrows-1, 0].xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
    ax[nrows-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[nrows-1, 0].xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))
    fig.suptitle(f'{unit_name}別観測値と異常度（{cf.method}  時間窓:{cf.lookback_timesteps} 閾値:{my_round(cf.threshold, 3)}）')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{cf.output_path}/{cf.title}_01gprd.png', bbox_inches="tight", pad_inches=0.05)  # グラフの保存
    # if cf.method == 'IsolationForest':
    #     ax[nrows-1, 0].set_ylim([0, 1])
    #     fig.savefig(f'{cf.output_path}/{cf.title}_01gprd_絶対値.png', bbox_inches="tight", pad_inches=0.05)  # グラフの保存
    plt.clf()
    plt.close()  # plt.clf() -> plt.close()でメモリ解放


def my_round(x, d=0):
    p = 10 ** d
    return float(math.floor((x * p) + math.copysign(0.5, x)))/p

def Set_config(f):  
    '''
    configファイル '*.ini' の値を読み込む（デフォルトはconfig.ini）
        args
            f:configファイル名
        return
            cf:configファイルからの読み取り値を設定したオブジェクト
    '''
    cf = configparser.ConfigParser()
    try:
        cf.read(f, encoding='utf-8')
    except:
        cf.read(f, encoding='shift-jis')

    # [General]
    cf.flg_random  = cf.getint('General', 'flg_random')
    cf.title       = cf.get('General', 'title')

    # [Path]
    cf.input_path   = cf.get('Path', 'input_path')
    cf.output_path  = cf.get('Path', 'output_path')
    cf.infile       = cf.get('Path', 'infile')
    
    # [Data]
    cf.cols_x1        = json.loads(cf.get('Data', 'colnames_x1'))   # colnames_y -> colnames_x1    

    cf.flg_samedata    = cf.getint('Data', 'flg_samedata')
    if cf.flg_samedata == 1:
        cf.x1label = cf.get('Data', 'x1label')
        cf.cols_x2 = json.loads(cf.get('Data', 'colnames_x2'))   # colnames_x -> colnames_x2
    else:
        cf.x1label = None
        cf.cols_x2 = []
    
    cf.flg_trainterm  = cf.getint('Data', 'flg_trainterm')
    if cf.flg_trainterm==1:
        cf.train_term = json.loads(cf.get('Data', 'train_term'))
        cf.train_term = [datetime.datetime.strptime(cf.train_term[i], '%Y-%m-%d') for i in range(len(cf.train_term))]

    cf.flg_monthlydata = cf.getint('Data', 'flg_monthlydata')

    # [Model]
    cf.normalization_type = cf.get('Model', 'normalization_type')
    cf.lookback_timesteps = cf.getint('Model', 'lookback_timesteps')
    # cf.ts_width       = cf.getint('Model', 'ts_width')

    cf.method     = cf.get('Model', 'method')
    if cf.method == 'kmeans':
        cf.kmeans_n = cf.getint('Model', 'kmeans_n')
    elif cf.method == 'kNN':
        cf.kNN_k = cf.getint('Model', 'kNN_k')
    
    cf.threshold  = cf.getfloat('Model', 'threshold')

    ## 入力データの設定（自動・固定）          
    cf.timestep_origin = 'D'
    # 出力ファイルの接頭語
    # cf.headname      = os.path.splitext(os.path.basename(f))[0]
        
    return cf


def Create_Update_folder(path_folder):
    '''
    新しいフォルダを作成する。既にある場合は既存フォルダを削除して改めて作成する
        args
            path:作成・更新対象のフォルダ
    '''
    try:
        os.makedirs(path_folder)
    except:
        shutil.rmtree(path_folder)
        os.makedirs(path_folder)    


def Set_seed(flg_random):
    '''
    プログラムのモデルのランダム性を固定する
    '''
    if flg_random == 1:
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        # tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)


def Shift_Lookforward_fromCurrent(nprawdata, prd_time):
    '''
    将来時系列を作成するメソッド（現時刻から）
        args
            nprawdata:元データ(2次元numpy.ndarray)
            prd_time:入力する将来時系列の期間  2の場合は、t, t+1
        return
            nptsdata_fwd :prd_time分を将来へシフトした時系列テンソル(num_samples, prd_time)
    '''
    num_samples = nprawdata.shape[0]
    num_variables = nprawdata.shape[1]
    # 変数ごとに作成
    nptsdata_fwd_list  = [[] for _ in range(num_variables)]
    for n in range(num_variables):
        _ts_fwds = []
        for i in range(num_samples-prd_time+1):
            _ts_fwds.append(list(nprawdata[i : i + prd_time, n]))
        nptsdata_fwd_list[n]  = np.array(_ts_fwds)
    # 変数分作成したものをここで統合
    nptsdata_fwd  = np.concatenate(nptsdata_fwd_list, axis=1)

    return nptsdata_fwd

def calc_kmeans_dist(centers_normal, centers_abnormal, np_ts):
    '''
    k-means法の際の距離を計算する
        args
            centers_normal:正常群の重心座標, centers_abnormal:異常群の重心座標, np_ts:距離を計算する入力する部分時系列データ
        return
            dist:正常群からの距離（異常度）
    '''
    dist_base = distance.euclidean(centers_normal.tolist(), centers_abnormal.tolist())
    dist = []
    for i in range(len(np_ts)):
        _dist_normal   = distance.euclidean(np_ts[i].tolist(), centers_normal.tolist())
        _dist_abnormal = distance.euclidean(np_ts[i].tolist(), centers_abnormal.tolist())

        _dist = _dist_normal - _dist_abnormal

        # # ベン図のAかつBの部分
        # if _dist_normal <= dist_base and _dist_abnormal <= dist_base:
        #     _dist = _dist_normal - _dist_abnormal
        # else:
        #     # AかつBの部分を除いた範囲のうち、正常からの距離が遠い（異常の重心付近）データ群
        #     if _dist_normal > _dist_abnormal:
        #         # _dist = _dist_normal - 0
        #         _dist = dist_base
        #     # AかつBの部分を除いた範囲のうち、異常からの距離が遠い（正常の重心付近）データ群
        #     elif _dist_normal < _dist_abnormal:
        #         # _dist = 0 - _dist_abnormal
        #         _dist = -dist_base
        #     # AかつBの部分を除いた範囲のうち、正常と異常との距離が同じ場合
        #     elif _dist_normal == _dist_abnormal:
        #         _dist = 0

        dist.append(_dist/dist_base)
    
    return dist

# main関数の実行
if __name__ == '__main__':
    main()
