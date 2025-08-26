# %% --------------------------------------- Load Packages -------------------------------------------------------------
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from skimage.transform import resize

# %% --------------------------------------- Define FID ----------------------------------------------------------------
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


# %% --------------------------------------- Calculate FID for Generator -----------------------------------------------
# scale an array of images to a new size
# Note: skimage will automatically change image range into [0, 1] after resizing
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return asarray(images_list)



# load real images from validation set

#path = './NKSID/'
path = './FLSMDD/'
path2 = './FLSMDD_raw/'

real_imgs = np.load(path + 'x_val1.npy')
real_label = np.load(path + 'y_val1.npy')

n_classes = 10
class_num = 10
channel = 1
latent_size = 128 
latent_dim = 128

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers

# 注册自定义层
@tf.keras.utils.register_keras_serializable()
class ChannelAttention(layers.Layer):
    def __init__(self, channel, reduction_ratio=16, name=None, **kwargs):
        super(ChannelAttention, self).__init__(name=name, **kwargs)
        self.channel = channel
        self.reduction_ratio = reduction_ratio
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
    def build(self, input_shape):
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(self.channel // self.reduction_ratio, activation='relu', use_bias=False),
            layers.Dense(self.channel, use_bias=False)
        ])
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        attention = tf.nn.sigmoid(avg_out + max_out)
        return tf.reshape(attention, [-1, 1, 1, x.shape[-1]])
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "channel": self.channel,
            "reduction_ratio": self.reduction_ratio
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, name=None, **kwargs):
        super(SpatialAttention, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=3)
        return self.conv(concat)
    
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable()
class CBAM(layers.Layer):
    def __init__(self, channel, reduction_ratio=16, kernel_size=7, name=None, **kwargs):
        super(CBAM, self).__init__(name=name, **kwargs)
        self.channel = channel
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.channel_attention = ChannelAttention(self.channel, self.reduction_ratio)
        self.spatial_attention = SpatialAttention(self.kernel_size)
        super(CBAM, self).build(input_shape)
        
    def call(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x
    
    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            "channel": self.channel,
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ConvBlock(layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.filters = filters
        
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        super(ConvBlock, self).build(input_shape)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "filters": self.filters
        })
        return config


@tf.keras.utils.register_keras_serializable()
class AttentionResidualEncoderBlock(layers.Layer):
    def __init__(self, filters, strides=1, name=None, **kwargs):
        # 确保正确传递 name 参数
        super(AttentionResidualEncoderBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        self.conv_block = tf.keras.Sequential([
            layers.Conv2D(self.filters, 3, strides=self.strides, padding='same'),
            #layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 3, padding='same'),
            #layers.BatchNormalization()
        ])
        self.cbam = CBAM(self.filters)
        self.downsample = None
        
        if input_shape[-1] != self.filters or self.strides != 1:
            self.downsample = layers.Conv2D(self.filters, 1, strides=self.strides, padding='same')
        
        super(AttentionResidualEncoderBlock, self).build(input_shape)
        
    def call(self, x):
        # 原始输入
        identity = x
        
        # 主路径
        out = self.conv_block(x)
        out = self.cbam(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = layers.Add()([out, identity])
        return layers.ReLU()(out)
    
    def get_config(self):
        config = super(AttentionResidualEncoderBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides
        })
        return config


# 同样为解码器块添加注册装饰器
@tf.keras.utils.register_keras_serializable()
class AttentionResidualDecoderBlock(layers.Layer):
    def __init__(self, filters,name=None, **kwargs):
        super(AttentionResidualDecoderBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.upsample = layers.Conv2DTranspose(self.filters, 3, strides=2, padding='same')
        self.conv_block = ConvBlock(self.filters)
        self.cbam = CBAM(self.filters)
        self.conv1x1 = layers.Conv2D(self.filters, 1, padding='same')
        
        super(AttentionResidualDecoderBlock, self).build(input_shape)
        
    def call(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
            x = self.conv1x1(x)
            
        identity = x
        
        # 主路径
        out = self.conv_block(x)
        out = self.cbam(out)
        
        # 残差连接
        out = layers.Add()([out, identity])
        return layers.ReLU()(out)
    
    def get_config(self):
        config = super(AttentionResidualDecoderBlock, self).get_config()
        config.update({
            "filters": self.filters
        })
        return config


from tensorflow.keras.utils import register_keras_serializable
@register_keras_serializable()
def combined_loss(y_true, y_pred):
    # 将图像从[-1,1]映射到[0,1]
    y_true = (y_true + 1) / 2.0
    y_pred = (y_pred + 1) / 2.0
    
    # MSE损失
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # MAE损失
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

   #    # 确保数据类型为 float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 确保数据在 [0, 1] 范围
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    # 计算多通道 MS-SSIM
    ms_ssim = tf.image.ssim_multiscale(
        y_true, 
        y_pred, 
        max_val=1.0,
        power_factors=[0.33, 0.33, 0.33],  # 使用3个尺度而不是默认的5个
        filter_size=11,  # 减小滤波器大小
        filter_sigma=1.5,  # 减小标准差
        k1=0.01,
        k2=0.03
    )
    
    # 计算平均 MS-SSIM
    ms_ssim = 1.0 - tf.reduce_mean(ms_ssim)
    
    # 感知损失
    # def perceptual_loss(y_true_normalized, y_pred_normalized):
    #     # 预处理
    #     y_true_processed = preprocess_input((y_true_normalized * 255.0) )
    #     y_pred_processed = preprocess_input((y_pred_normalized * 255.0) )
        
    #     # 提取特征
    #     y_true_features = feature_extractor(y_true_processed)
    #     y_pred_features = feature_extractor(y_pred_processed)
        
    #     # 计算特征层面的损失
    #     return tf.reduce_mean(tf.square(y_true_features - y_pred_features))
    
    # 计算感知损失
    #perc_loss = perceptual_loss(y_true_normalized, y_pred_normalized)
    
    # 组合损失（可以根据需要调整权重）
    total_loss = mse_loss + mae_loss + 0.1 * ms_ssim #+ 0.01 * perc_loss
    
    return total_loss



# 注册自定义对象
custom_objects = {
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
    'CBAM': CBAM,
    'ConvBlock': ConvBlock,
    'AttentionResidualEncoderBlock': AttentionResidualEncoderBlock,
    'AttentionResidualDecoderBlock': AttentionResidualDecoderBlock,
    'combined_loss': combined_loss,
}

tf.keras.utils.get_custom_objects().update(custom_objects)

with tf.keras.utils.custom_object_scope(custom_objects):
    #generator = load_model(path+'g_model_115.h5')
    generator = load_model(path2+'g_model_BAGAN_199.h5')


from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential, Model


def build_generator(latent_size):
    
    #init = RandomNormal(stddev=0.02)
    image_class = Input(shape=(1,), dtype='int32')
    noise_le = Input((latent_size,))
    cls = Flatten()(Embedding(class_num, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([noise_le, cls])

    x = Dense(4*4*256)(h)
    x = LeakyReLU(alpha=0.2)(x)

    ## Size: 4 x 4 x 256
    x = Reshape((4, 4, 256))(x)
    ## Size: 8 x 8 x 128
    x = AttentionResidualDecoderBlock(128)(x)
    
    ## Size: 16 x 16 x 128
    x = AttentionResidualDecoderBlock(64)(x)
 
    ## Size: 32 x 32 x 64
    x = AttentionResidualDecoderBlock(32)(x)

    ## Size: 64 x 64 x 3
    fake_image = Conv2DTranspose(channel, 3, strides=2, padding='same', activation='tanh')(x)
    

    return Model([noise_le , image_class], fake_image)



# load generator
generator = build_generator(latent_size)
generator.load_weights('acgan_FLS/params_generator_epoch_449.hdf5')
#generator.load_weights('cWGAN_FLS/params_generator_epoch_550.hdf5')

#generator.load_weights('acgan_NKSID/params_generator_epoch_699.hdf5')
#generator.load_weights('cWGAN_NKSID/params_generator_epoch_440.hdf5')


# calculate FID for each class
fids = []
nums = []
n_classes = len(np.unique(real_label))
sample_size = 1000
for c in range(n_classes):
    ########### get generated samples by class ###########
    label = np.ones(sample_size) * c
    noise = np.random.normal(0, 1, (sample_size, generator.input_shape[0][1]))
    print('Latent dimension:', generator.input_shape[0][1])
    gen_samples = generator.predict([noise, label])
    gen_samples = gen_samples*0.5 + 0.5

    ########### load real samples from training set ###########
    # gen_samples = np.load('x_train.npy')
    # gen_label = np.load('y_train.npy')
    # gen_samples = gen_samples[gen_label.reshape(-1) == c]
    # shuffle(gen_samples)
    # gen_samples = gen_samples[:1000]

    ########### get real samples by class ###########
    real_samples = real_imgs[real_label.reshape(-1) == c]
    # shuffle(real_imgs)  # shuffle it or not
    # real_samples = real_samples[:1000]  # less calculation
    real_samples = real_samples.astype('float32') / 255.

    # resize images
    gen_samples = scale_images(gen_samples, (299,299,3))
    real_samples = scale_images(real_samples, (299,299,3))


    print('Scaled', gen_samples.shape, real_samples.shape)
    print('Scaled range for generated', np.min(gen_samples[0]), np.max(gen_samples[0]))
    print('Scaled range for real', np.min(real_samples[0]), np.max(real_samples[0]))

    # preprocess images
    gen_samples = preprocess_input(gen_samples)
    real_samples = preprocess_input(real_samples)
    print('Scaled range for generated', np.min(gen_samples[0]), np.max(gen_samples[0]))
    print('Scaled range for real', np.min(real_samples[0]), np.max(real_samples[0]))

    # calculate fid
    fid = calculate_fid(model, gen_samples, real_samples)
    fids.append(fid)
    nums.append(real_samples.shape[0])
    print('>>FID(%d): %.3f' % (c, fid))
    print('-'*50)

print(nums)
print(fids)