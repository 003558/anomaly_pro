@echo off
rem Ouyou ����_���@CTI kokubunken ISP

rem ����2�s�������̃C���X�g�[���������ɍ��킹��@����������3�s��exe�̏ꍇ�͕s�v�i�폜����j����
call C:\Users\nt001764\anaconda3\Scripts\activate.bat
call activate t21stats

echo �v���O�������J�n���܂���

rem �����Ɏ������ݒ肵��config�t�@�C���u*.ini�v���L�q����
rem ������config�t�@�C����A�����ĉ񂷏ꍇ�́A�s��ǉ�����

call python ouyou_ML03.py setting_ML.ini

echo �v���O�������I�����܂���

pause
