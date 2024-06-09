在完成代码部分开发后，执行如下指令检验所编写代码是否符合精度要求以及运算结果是否合理。
需要分别执行CPU侧调试以及NPU侧调试
虚拟机仿真建议环境版本：Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64
完整工程结构如下，需要补充完整的部分有标注：
SinhCustom
|-- input
|-- output
|-- CMakeLists.txt 
|-- SinhCustom.cpp    需要补充完整
|-- SinhCustom.py
|-- cmake 
|-- data_utils.h 
|-- main.cpp    需要补充完整
|-- run.sh 
|-- readme.md
|-- verify_result.py 

## CPU调试
进入root用户权限
bash run.sh SinhCustom ascend910B1 VectorCore cpu
python3 verify_result.py ./output/output_z.bin ./output/golden.bin

## NPU调试
进入root用户权限
bash run.sh SinhCustom ascend910B1 VectorCore npu
python3 verify_result.py ./output/output_z.bin ./output/golden.bin

## 验证结果正确会显示如下反馈（示例）
----------your result is :----------
[1.628e+01 4.535e+00 1.181e+02 ... 6.508e+00 3.139e+00 3.412e+03]
--------The real result is :--------
[1.628e+01 4.535e+00 1.181e+02 ... 6.508e+00 3.139e+00 3.412e+03]
test pass

## 验证结果错误会显示如下反馈（示例）
----------your result is :----------
[0. 0. 0. ... 0. 0. 0.]
--------The real result is :--------
[6.3828e+00 5.8750e+00 6.7320e+03 ... 1.2675e+02 1.8300e+02 5.8945e+00]
[ERROR] result error

