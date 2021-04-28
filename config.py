train_title = 'ResNet50_v1_'

model = 'ResNet50_v1'
# Vgg: Vgg
# ResNet: ResNet50_v1 ResNet50_v1_Pretrain ResNet50_v2 ResNet50_v2_Pretrain
# Inception_ResNetV2: Inception_ResNetV2 Inception_ResNetV2_Pretrain
# EfficientNet: EfficientNetB4 EfficientNetB4_Pretrain EfficientNetB5 EfficientNetB5_Pretrain

# Training config
category_num = 40
batch_size = 32
epoch_num = 50
input_shape = (224, 224, 3)

log_file = './result/log/' + train_title + '.txt'
plt_loss_file = './result/log/' + train_title + '_loss.jpg'
plt_accuracy_file = './result/log/' + train_title + '_accuracy.jpg'
save_weight_file = './result/weight/' + train_title + '.h5'

# Dataset config
train_list_path = './data/train_label.txt'
test_list_path = './data/test_label.txt'
train_data_path = './data/MyData/TrainData'
test_data_path = './data/MyData/TestData'

# Augmentation config
# From 'Bag of tricks for image classification with convolutional neural networks'
# Or https://github.com/dmlc/gluon-cv
data_augment_tricks = False

weight_decay = 1e-5
label_smoothing = 0

initial_learning_rate = 0.05
minimum_learning_rate = 0.0001

short_side_scale = (256, 384)
aspect_ratio_scale = (0.8, 1.25)
hue_delta = (-36, 36)
saturation_scale = (0.6, 1.4)
brightness_scale = (0.6, 1.4)
pca_std = 0.1

mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]

