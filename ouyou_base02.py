# =====================================================================================================
# ダム異常検知AI　応用ケース  データの整形とグラフの描画
#   同一ダム同種データ、同一ダム異種データの両方に対応
#　2022.01.14 ver01 N.Tagashira ツール用に再構成
#  2022.05.10 ver02 N.Tagashira 同種、異種データの両方に対応,　学習期間設定に対応
# =====================================================================================================

# ライブラリの読み込み
import configparser  # iniファイル形式のconfigを読み込む
# import copy
import datetime      # 時間計測に必要なライブラリ
# import glob          # ファイル名取得や削除に必要なライブラリ
import json          # iniファイルからリスト形式を読む場合にjson.loadsを使用する
import os            # pathの連結等に必要なライブラリ
# import pickle        # オブジェクトをバイナリで保存、ロードするのに必要なライブラリ
# import random        # random.seedを固定用
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
pydevd_warn_slow_resolve_timeout=2 

# # 状態空間モデル用のライブラリ
# import statsmodels.api as sm

# 機械学習用ライブラリ
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    # cofigファイルを読み込む
    configfile = sys.argv[1]  # コマンドライン引数の1つ目
    cf = Set_config(f=configfile)
    
    '''共通設定'''
    timestep_origin = 'D'
    x1intp_sfx = ['LI', '線形補間']       # ['SSM', '状態空間モデル補間']

    num_cols_x1  = len(cf.cols_x1)  # cols_y -> cols_x1
    num_cols_x2  = len(cf.cols_x2)  # cols_x -> cols_x2
    cols_x1_intp = [f'{cf.cols_x1[i]}_{x1intp_sfx[0]}' for i in range(num_cols_x1)]   # 目的変数はyintp_sfxに従う ⇒　現在の設定は線形補間に固定
    cols_x2_intp = [f'{cf.cols_x2[i]}_LI' for i in range(num_cols_x2)]               # 説明変数は線形補間で固定
    
    if cf.flg_monthlydata == 1:
        timestep_analysis = 'M'
        month_sfx = '月平均'  
        cols_x1_month = [f'{cols_x1_intp[i]}_{month_sfx}' for i in range(num_cols_x1)]
        cols_x2_month = [f'{cols_x2_intp[i]}_{month_sfx}' for i in range(num_cols_x2)]
        
    gsize = [15, 3, 3]  # 1グラフの大きさ[グラフの幅, グラフの高さ, 凡例1列の幅]

    # output, フォルダの作成
    Create_Update_folder(cf.output_path)
    # configfileの保存
    shutil.copy2(configfile, f'{cf.output_path}/')

    '''日データ（オリジナル）'''
    # データ入力とcsv出力
    df = Input_Interpolation_data(cf.input_path, cf.infile, timestep_origin, cf.cols_x1, cf.cols_x2)
    df.to_csv(f'{cf.output_path}/{cf.title}_01df.csv', encoding='shift-jis')

    # グラフ出力 観測時系列ごと、全時系列
    Plot_TimeSeries_Graph_Daily_All(gsize, cf.flg_samedata, cf.output_path, cf.title, df, cf.cols_x1, cf.cols_x2, cols_x1_intp, cols_x2_intp, x1intp_sfx, cf.flg_trainterm, cf.train_term, cf.x1label)
    Plot_TimeSeries_Graph_Daily_Individual(gsize, cf.flg_samedata, cf.output_path, cf.title, df, cf.cols_x1, cf.cols_x2, cols_x1_intp, cols_x2_intp, x1intp_sfx, cf.flg_trainterm, cf.train_term, cf.x1label)

    '''月データ'''
    # 日⇒月データのリサンプリング、出力_csv
    if cf.flg_monthlydata == 1:
        # dfterm = Resampling_Term(df, cf.timestep_analysis, cols_x1_intp, cols_x_intp)
        dfterm = Resampling_Term(df, timestep_analysis, cols_x1_intp, cols_x2_intp, month_sfx)
        dfterm.to_csv(f'{cf.output_path}/{cf.title}_11df_monthly.csv', encoding='shift-jis')

        # グラフ出力 観測時系列ごと、全時系列
        Plot_TimeSeries_Graph_Monthly_All(gsize, cf.flg_samedata, cf.output_path, cf.title, dfterm, cf.cols_x1, cf.cols_x2, cols_x1_month, cols_x2_month, month_sfx, cf.flg_trainterm, cf.train_term, cf.x1label)
        Plot_TimeSeries_Graph_Monthly_Individual(gsize, cf.flg_samedata, cf.output_path, cf.title, dfterm, cf.cols_x1, cf.cols_x2, cols_x1_month, cols_x2_month, month_sfx, cf.flg_trainterm, cf.train_term, cf.x1label)


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
    cf.title       = cf.get('General', 'title')

    # [Path]
    cf.input_path  = cf.get('Path', 'input_path')
    cf.infile      = cf.get('Path', 'infile')
    cf.output_path = cf.get('Path', 'output_path')
    
    # [Data]
    cf.cols_x1      = json.loads(cf.get('Data', 'colnames_x1'))   # colnames_y -> colnames_x1
    
    cf.flg_samedata    = cf.getint('Data', 'flg_samedata')
    if cf.flg_samedata == 1:
        cf.x1label = cf.get('Data', 'x1label')
        cf.cols_x2 = json.loads(cf.get('Data', 'colnames_x2'))   # colnames_x -> colnames_x2
    else:
        cf.x1label = None
        cf.cols_x2 = []
    
    cf.flg_trainterm = cf.getint('Data', 'flg_trainterm')
    if cf.flg_trainterm==1:
        cf.train_term = json.loads(cf.get('Data', 'train_term'))
        cf.train_term = [datetime.datetime.strptime(cf.train_term[i], '%Y-%m-%d') for i in range(len(cf.train_term))]
    else:
        cf.train_term = None

    ## 入力データの設定（自動・固定）
    cf.flg_monthlydata = 1
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

def Input_Interpolation_data(input_path, infile, timestep, cols_x1, cols_x2):
    '''
    訓練＋検証期間のデータを抽出し、目的変数と説明変数の列のみのデータフレームを返す
    その際に、説明変数は線形補完し、目的変数がnanで始まらないように、開始行はnanではないデータまで削除する
        args
            iunput_path:入力フォルダのパス, infile:対象とするcsvファイル（train,val,testデータが連結されたデータ）
            timestep_unit:時系列データの単位（分:T, 時間:H, 日:D, 月:M, 年:Y）
            cols_x1:目的変数のフィールド名,  cols_x2:説明変数のフィールド名
            interpolation:補間方法（線形補間, SSM補間）
        return
            df:補間処理したデータフレーム
    '''
    # csvファイルの読み込み
    csv_path = os.path.join(input_path, infile)
    df = pd.read_csv(csv_path , header=[0], index_col=0, parse_dates=True, encoding='shift-jis')

    # 日付欠損の修正, 必要なフィールドの抽出, 数値変換
    df = df.asfreq(timestep)
    df = df[(cols_x2 + cols_x1)]
    df = df.astype(float)

    # 線形補間を行い、1行目と最終行は、目的変数に欠損値がない状態にする
    num_cols_x1 = len(cols_x1)
    num_cols_x2 = len(cols_x2)
    
    # 線形補間
    for i in range(num_cols_x2):
        df[f'{cols_x2[i]}_LI'] = df[cols_x2[i]].interpolate()
    for i in range(num_cols_x1):
        df[f'{cols_x1[i]}_LI'] = df[cols_x1[i]].interpolate()
    
    # # 状態空間モデル補間（目的変数）を実施　（ローカルレベル＋固定回帰モデルによる補間）
    # for i in range(num_cols_x1):
    #     # 説明変数がある場合
    #     if num_cols_x2 > 0:
    #         cols_x2_LI = [col_x + '_LI' for col_x in cols_x2]
    #         model = sm.tsa.UnobservedComponents(endog=df[cols_x1[i]], level='local level', exog=df[cols_x2_LI])
    #     # 説明変数がない場合
    #     else:
    #         model = sm.tsa.UnobservedComponents(endog=df[cols_x1[i]], level='local level')
    #     res = model.fit(start_parms=model.fit(method='nm').params, method='bfgs')
    #     df[f'{cols_x1[i]}_SSM'] = res.smoother_results.smoothed_forecasts.flatten()
        
    # 上から目的変数x1がnanではない行のindex(startindex)を検索する
    for index, row in df.iterrows():
        if not all(row[cols_x1].isnull()):  # kihonはallの部分がany
            startindex = index
            break
    # データフレームの行方向の並びを反転して、上から目的変数x1がnanではない行のindex(endindex)を検索する
    for index, row in df.iloc[::-1].iterrows():
        if not all(row[cols_x1].isnull()):  # kihonはallの部分がany
            endindex = index
            break
    df = df[(df.index >= startindex) & (df.index <= endindex)]
    
    return df

def Resampling_Term(df, term, cols_x1_intp, cols_x2_intp, month_sfx):
    '''
    期間別に集計する
        args:
            df:集計対象とする元dataframe, term:集計期間（M:月, Q:四半期, Y:年）,
            cols_x1_inpt:目的変数補間値の列名, cols_x2_intp:説明変数補間値の列名, month_sfx:月平均値データの接尾語
        return:
            dfterm:期間で平均したdataframe（月平均）
    '''
    # 平均値を計算して列名をrename
    dfterm = df[cols_x2_intp+cols_x1_intp].resample(rule=term).mean().copy()
    dfterm = dfterm.add_suffix(f'_{month_sfx}')
    # # 月最大値を計算して、dftermに追加
    # dfmax = df.resample(rule=term).max().copy()
    # for i in range(num_cols_x2):
    #     dfterm[f'{cols_x2_intp[i]}_max'] = dfmax[cols_x2_intp[i]]
    # for i in range(num_cols_x1):
    #     dfterm[f'{cols_x1_intp[i]}_max'] = dfmax[cols_x1_intp[i]]
    # # 月最小値を計算して、dftermに追加
    # dfmin = df.resample(rule=term).min().copy()
    # for i in range(len(cols_x2_intp)):
    #     dfterm[f'{cols_x2_intp[i]}_min'] = dfmin[cols_x2_intp[i]]
    # for i in range(len(cols_x1_intp)):
    #     dfterm[f'{cols_x1_intp[i]}_min'] = dfmin[cols_x1_intp[i]]

    return dfterm


def Plot_TimeSeries_Graph_Daily_All(gsize, flg_samedata, output_path, title, df, cols_x1, cols_x2, cols_x1_intp, cols_x2_intp, x1intp_sfx, flg_trainterm, train_term, x1label=None):
    '''日データの全目的変数時系列を一緒にしたグラフを作成
        args:
            gsize(リスト):1グラフの大きさ[グラフの幅, グラフの高さ, 凡例1列の幅],
            flg_samedata:同種データ判定（0:異種, 1:同種）, output_path:出力フォルダ, title:グラフタイトル, df:対象とするdataframe,
            cols_x1:目的変数の列名, cols_x2:説明変数の列名, cols_x1_intp:目的変数補間値の列名, cols_x2_intp:説明変数補間値の列名, 
            x1intp_sfx:目的変数の補間方法の接尾語（記号）, flg_trainterm:学習期間のライン（0:追加しない, 1:追加する）, train_term:学習期間,
            x1label:目的変数名（同種データの場合のみ）
    '''
    num_cols_x1 = len(cols_x1)
    num_cols_x2 = len(cols_x2)

    if flg_samedata == 1:
        nrows = num_cols_x2 + 1
        ncols_legend = int(num_cols_x1/13)+1  # 最大1列12段
    else:
        nrows = num_cols_x2 + num_cols_x1
        ncols_legend = 1
    
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(gsize[0]+gsize[2]*ncols_legend, gsize[1]*nrows), sharex=True, squeeze=False)
    
    # 説明変数の描画
    for i in range(num_cols_x2):
        # ax[i, 0].plot(df.index.to_numpy(), df[cols_x2[i]].to_numpy(),      linestyle='None',  label=f'{cols_x2[i]}_観測値', marker='o', markersize=2) # 観測値
        ax[i, 0].plot(df.index.to_numpy(), df[cols_x2_intp[i]].to_numpy(), linestyle='solid', label=f'{cols_x2[i]}_線形補間')                         # 線形補間値
        ax[i, 0].set_ylabel(cols_x2[i])
        ax[i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        ax[i, 0].grid(which='both', axis='both')
    
    # 目的変数の描画
    if flg_samedata == 1:  # 同種データ:目的変数を1つのグラフに集約
        for i in range(num_cols_x1):
            ax[num_cols_x2, 0].plot(df.index.to_numpy(), df[cols_x1_intp[i]].to_numpy(), linestyle='solid', label=f'{cols_x1[i]}_{x1intp_sfx[1]}')  # 補間値（線形orSSM）
        ax[num_cols_x2, 0].set_ylabel(x1label)
        ax[num_cols_x2, 0].grid(which='both', axis='both')
        # if num_cols_x1 < 11: ax[num_cols_x2, 0].legend()
        ax[num_cols_x2, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', ncol=ncols_legend)
    else:                  # 異種データ:計測項目ごとにグラフを描画
        for i in range(num_cols_x1):
            ax[num_cols_x2+i, 0].plot(df.index.to_numpy(), df[cols_x1_intp[i]].to_numpy(), linestyle='solid', label=f'{cols_x1[i]}_{x1intp_sfx[1]}')  # 補間値（線形orSSM）
            ax[num_cols_x2+i, 0].set_ylabel(cols_x1[i])
            ax[num_cols_x2+i, 0].grid(which='both', axis='both')
            ax[num_cols_x2+i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # x軸、タイトル等の描画
    if flg_trainterm==1:
        for i in range(nrows):
            ax[i, 0].axvline(x=train_term[0], color='magenta', linestyle='dotted')#, ymin=0.5, ymax=0.95)
            ax[i, 0].axvline(x=train_term[1], color='magenta', linestyle='dotted')
    ax[nrows-1, 0].xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
    ax[nrows-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[nrows-1, 0].xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))
    fig.suptitle(f'{title} 観測値および補間値')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{output_path}/{title}_02g_all.png', bbox_inches="tight", pad_inches=0.05)  # グラフの保存
    plt.clf()
    plt.close()  # plt.clf() -> plt.close()でメモリ解放


def Plot_TimeSeries_Graph_Daily_Individual(gsize, flg_samedata, output_path, title, df, cols_x1, cols_x2, cols_x1_intp, cols_x2_intp, x1intp_sfx, flg_trainterm, train_term, x1label=None):
    '''日データの観測時系列ごとのグラフを作成
        args:
            gsize(リスト):1グラフの大きさ[グラフの幅, グラフの高さ, 凡例1列の幅],
            flg_samedata:同種データ判定（0:異種, 1:同種）, output_path:出力フォルダ, title:グラフタイトル, df:対象とするdataframe,
            cols_x1:目的変数の列名, cols_x2:説明変数の列名, cols_x1_intp:目的変数補間値の列名, cols_x2_intp:説明変数補間値の列名, 
            x1intp_sfx:目的変数の補間方法の接尾語（記号）, flg_trainterm:学習期間のライン（0:追加しない, 1:追加する）, train_term:学習期間,
            x1label:目的変数名（同種データの場合のみ）
    '''
    num_cols_x1 = len(cols_x1)
    num_cols_x2 = len(cols_x2)

    if flg_samedata == 1:
        idf = df[cols_x1].copy()
        min_y = min(idf.min())
        max_y = max(idf.max())    
        range_y = max_y - min_y

    for n in range(num_cols_x1):
        nrows = num_cols_x2 + 1
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(gsize[0]+gsize[2], gsize[1]*nrows), sharex=True, squeeze=False)
        # 説明変数の描画
        for i in range(num_cols_x2):
            ax[i, 0].plot(df.index.to_numpy(), df[cols_x2[i]].to_numpy(),      linestyle='None',  label=f'{cols_x2[i]}_観測値', marker='o', markersize=2)  # 観測値
            ax[i, 0].plot(df.index.to_numpy(), df[cols_x2_intp[i]].to_numpy(), linestyle='solid', label=f'{cols_x2[i]}_線形補間')                          # 線形補間値
            ax[i, 0].set_ylabel(cols_x2[i])
            ax[i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            ax[i, 0].grid(which='both', axis='both')
            
        # 目的変数の描画
        ax[num_cols_x2, 0].plot(df.index.to_numpy(), df[cols_x1[n]].to_numpy(),      linestyle='None',  label=f'{cols_x1[n]}_観測値', marker='o', markersize=2)  # 観測値
        ax[num_cols_x2, 0].plot(df.index.to_numpy(), df[cols_x1_intp[n]].to_numpy(), linestyle='solid', label=f'{cols_x1[n]}_{x1intp_sfx[1]}')                   # 補間値（線形orSSM）
        if flg_samedata == 1:
            ax[num_cols_x2, 0].set_ylabel(x1label)
            ax[num_cols_x2, 0].set_ylim([min_y-range_y*0.05, max_y+range_y*0.05])
        else:
            ax[num_cols_x2, 0].set_ylabel(cols_x1[n])
        ax[num_cols_x2, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        ax[num_cols_x2, 0].grid(which='both', axis='both')
      
        # x軸、タイトル等の描画
        if flg_trainterm==1:
            for i in range(nrows):
                ax[i, 0].axvline(x=train_term[0], color='magenta', linestyle='dotted')#, ymin=0.5, ymax=0.95)
                ax[i, 0].axvline(x=train_term[1], color='magenta', linestyle='dotted')
    
        ax[num_cols_x2, 0].xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
        ax[num_cols_x2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax[num_cols_x2, 0].xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))
        fig.suptitle(f'{title} {cols_x1[n]} 観測値および補間値')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(f'{output_path}/{title}_03g_{n+1:02}.png', bbox_inches="tight", pad_inches=0.05)          # グラフの保存
        plt.clf()
        plt.close()  # plt.clf() -> plt.close()でメモリ解放

def Plot_TimeSeries_Graph_Monthly_All(gsize, flg_samedata, output_path, title, df, cols_x1, cols_x2, cols_x1_month, cols_x2_month, month_sfx, flg_trainterm, train_term, x1label=None):
    '''日データの全目的変数時系列を一緒にしたグラフを作成
        args:
            gsize(リスト):1グラフの大きさ[グラフの幅, グラフの高さ, 凡例1列の幅],
            flg_samedata:同種データ判定（0:異種, 1:同種）, output_path:出力フォルダ, title:グラフタイトル, df:対象とするdataframe,
            cols_x1:目的変数の列名, cols_x2:説明変数の列名, cols_x1_intp:目的変数補間値の列名, cols_x2_intp:説明変数補間値の列名, 
            month_sfx:目的変数の補間方法の接尾語（記号）, flg_trainterm:学習期間のライン（0:追加しない, 1:追加する）, train_term:学習期間,
            x1label:目的変数名（同種データの場合のみ）
    '''
    num_cols_x1 = len(cols_x1)
    num_cols_x2 = len(cols_x2)

    if flg_samedata == 1:
        nrows = num_cols_x2 + 1
        ncols_legend = int(num_cols_x1/13)+1  # 最大1列12段
    else:
        nrows = num_cols_x2 + num_cols_x1
        ncols_legend = 1
    
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(gsize[0]+gsize[2]*ncols_legend, gsize[1]*nrows), sharex=True, squeeze=False)
    
    # 説明変数の描画
    for i in range(num_cols_x2):
        ax[i, 0].plot(df.index.to_numpy(), df[cols_x2_month[i]].to_numpy(), linestyle='solid', label=f'{cols_x2[i]}_月平均値')
        ax[i, 0].grid(which='both', axis='both')
        ax[i, 0].set_ylabel(cols_x2[i])
        ax[i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        
    # 目的変数の描画
    if flg_samedata == 1:  # 同種データ:目的変数を1つのグラフに集約
        for i in range(num_cols_x1):
            ax[num_cols_x2, 0].plot(df.index.to_numpy(), df[cols_x1_month[i]].to_numpy(), linestyle='solid', label=f'{cols_x1[i]}_月平均値') 
        ax[num_cols_x2, 0].grid(which='both', axis='both')
        ax[num_cols_x2, 0].set_ylabel(x1label)
        # if num_cols_x1 < 11: ax[num_cols_x2, 0].legend() 
        ax[num_cols_x2, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', ncol=ncols_legend)
        
    else:                  # 異種データ:計測項目ごとにグラフを描画
        for i in range(num_cols_x1):
            ax[num_cols_x2+i, 0].plot(df.index.to_numpy(), df[cols_x1_month[i]].to_numpy(), linestyle='solid', label=f'{cols_x1[i]}_{month_sfx}')
            ax[num_cols_x2+i, 0].grid(which='both', axis='both')
            ax[num_cols_x2+i, 0].set_ylabel(cols_x1[i])
            ax[num_cols_x2+i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        
    # x軸、タイトル等の描画
    if flg_trainterm==1:
        for i in range(nrows):
            ax[i, 0].axvline(x=train_term[0], color='magenta', linestyle='dotted')#, ymin=0.5, ymax=0.95)
            ax[i, 0].axvline(x=train_term[1], color='magenta', linestyle='dotted')
    ax[nrows-1, 0].xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
    ax[nrows-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[nrows-1, 0].xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))
    fig.suptitle(f'{title} 月平均値')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{output_path}/{title}_12g_{month_sfx}_all.png', bbox_inches="tight", pad_inches=0.05)  # グラフの保存
    plt.clf()
    plt.close()  # plt.clf() -> plt.close()でメモリ解放


def Plot_TimeSeries_Graph_Monthly_Individual(gsize, flg_samedata, output_path, title, df, cols_x1, cols_x2, cols_x1_month, cols_x2_month, month_sfx, flg_trainterm, train_term, x1label=None):
    '''月平均値データの観測時系列ごとのグラフを作成 
        args:
            gsize(リスト):1グラフの大きさ[グラフの幅, グラフの高さ, 凡例1列の幅],
            flg_samedata:同種データ判定（0:異種, 1:同種）, output_path:出力フォルダ, title:グラフタイトル, df:対象とするdataframe,
            cols_x1:目的変数の列名, cols_x2:説明変数の列名, cols_x1_month:目的変数月平均の列名, cols_x2_month:説明変数月平均の列名, 
            month_sfx:目的変数の月平均の接尾語（記号）, flg_trainterm:学習期間のライン（0:追加しない, 1:追加する）, train_term:学習期間,
            x1label:目的変数名（同種データの場合のみ）
    '''
    num_cols_x1 = len(cols_x1)
    num_cols_x2 = len(cols_x2)

    if flg_samedata == 1:
        idf = df[cols_x1_month].copy()
        min_y = min(idf.min())
        max_y = max(idf.max())    
        range_y = max_y - min_y
    
    for n in range(num_cols_x1):
        nrows = num_cols_x2 + 1
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(gsize[0]+gsize[2], gsize[1]*nrows), sharex=True, squeeze=False)
        
        # 説明変数の描画
        for i in range(num_cols_x2):
            ax[i, 0].plot(df.index.to_numpy(), df[cols_x2_month[i]].to_numpy(), linestyle='solid', label=f'{cols_x2[i]}_月平均値')
            ax[i, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            ax[i, 0].grid(which='both', axis='both')
            ax[i, 0].set_ylabel(cols_x2[i])
        
        # 目的変数の描画
        ax[num_cols_x2, 0].plot(df.index.to_numpy(), df[cols_x1_month[n]].to_numpy(), linestyle='solid', label=f'{cols_x1[n]}_月平均値')
        ax[num_cols_x2, 0].grid(which='both', axis='both')
        ax[num_cols_x2, 0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        if flg_samedata == 1:
            ax[num_cols_x2, 0].set_ylabel(x1label)
            ax[num_cols_x2, 0].set_ylim([min_y-range_y*0.05, max_y+range_y*0.05])
        else:
            ax[num_cols_x2, 0].set_ylabel(cols_x1[n])
        
        # x軸、タイトル等の描画
        if flg_trainterm==1:
            for i in range(nrows):
                ax[i, 0].axvline(x=train_term[0], color='magenta', linestyle='dotted')#, ymin=0.5, ymax=0.95)
                ax[i, 0].axvline(x=train_term[1], color='magenta', linestyle='dotted')    
        ax[num_cols_x2, 0].xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
        ax[num_cols_x2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax[num_cols_x2, 0].xaxis.set_minor_locator(mdates.YearLocator(base=1, month=1, day=1))
        fig.suptitle(f'{title} {cols_x1[n]} 月平均値')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(f'{output_path}/{title}_13g_{month_sfx}_{n+1:02}.png', bbox_inches="tight", pad_inches=0.05)          # グラフの保存
        plt.clf()
        plt.close()  # plt.clf() -> plt.close()でメモリ解放


# main関数の実行
if __name__ == '__main__':
    main()

