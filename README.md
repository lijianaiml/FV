# FV
face verification project(python+keras+tensorflow)

基本思路：

使用预训练的图片分类网络作为特征提取器，对自己的照片提取特征，与输入图片做差值后转变成一个二分类问题

具体做法：
使用预训练的inceptionV3（去掉最后一层后加上flatten层）提取正负样本（自己与他人的人脸图片）特征后做差值处理，将差值进行logistics回归，输出0/1


数据集：
Olivetti Faces图片及自拍图片
229*229 灰度图