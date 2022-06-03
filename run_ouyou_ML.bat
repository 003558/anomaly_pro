@echo off
rem Ouyou 同一ダム　CTI kokubunken ISP

rem 下の2行を自分のインストールした環境に合わせる　←ここから3行はexeの場合は不要（削除する）せる
call C:\Users\nt001764\anaconda3\Scripts\activate.bat
call activate t21stats

echo プログラムを開始しました

rem ここに自分が設定したconfigファイル「*.ini」を記述する
rem 複数のconfigファイルを連続して回す場合は、行を追加する

call python ouyou_ML03.py setting_ML.ini

echo プログラムが終了しました

pause
