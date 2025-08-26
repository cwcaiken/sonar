# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
tf.config.experimental_run_functions_eagerly(True)

# %% --------------------------------------- Fix Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)


# %% ---------------------------------- Data Preparation ---------------------------------------------------------------
def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images


path = './NKSID/'
#path = './FLSMDD/'
images = np.load(path + 'x_train.npy')
labels = np.load(path + 'y_train.npy')
images = change_image_shape(images)

######################## Preprocessing ##########################
# Set channel
channel = images.shape[-1]
img_size = (64,64, 3)
# to 64 x 64 x channel
real = np.ndarray(shape=(images.shape[0], img_size[0], img_size[1], channel))
for i in range(images.shape[0]):
    real[i] = cv2.resize(images[i], img_size[0:2]).reshape((img_size[0], img_size[1], channel))

# Train test split, for autoencoder (actually, this step is redundant if we already have test set)
x_train, x_test, y_train, y_test = train_test_split(real, labels, test_size=0.2, shuffle=True, random_state=42)

# It is suggested to use [-1, 1] input for GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train[0].shape
# Get number of classes
n_classes = len(np.unique(y_train))

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------


latent_dim = 128
# trainRatio === times(Train D) / times(Train G)
trainRatio = 3

# %% ---------------------------------- Models Setup -------------------------------------------------------------------
# Build Generator/Decoder

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


def decoder():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    noise_le = Input((latent_dim,))

    x = Dense(4*4*256)(noise_le)
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
    generated = Conv2DTranspose(channel, 3, strides=2, padding='same', activation='tanh', kernel_initializer=init)(x)

    generator = Model(inputs=noise_le, outputs=generated)
    return generator

# Build Encoder
def encoder():
    # weight initialization
    #init = RandomNormal(stddev=0.02)

    img = Input(img_size)

    x = AttentionResidualEncoderBlock(32, strides=2)(img)
   
    x = AttentionResidualEncoderBlock(64, strides=2)(x)

    x = AttentionResidualEncoderBlock(128, strides=2)(x)

    x = AttentionResidualEncoderBlock(256, strides=2)(x)

    # 4 x 4 x 256
    feature = Flatten()(x)
    #feature = layers.GlobalAveragePooling2D(x)
    out = Dense(latent_dim, kernel_initializer='he_normal')(feature)
    
    model = Model(inputs=img, outputs=out)
    return model

# Build Embedding model
def embedding_labeled_latent():
    # # weight initialization
    # init = RandomNormal(stddev=0.02)

    label = Input((1,), dtype='int32')
    noise = Input((latent_dim,))
    # ne = Dense(256)(noise)
    # ne = LeakyReLU(0.2)(ne)

    le = Flatten()(Embedding(n_classes, latent_dim)(label))
    # le = Dense(256)(le)
    # le = LeakyReLU(0.2)(le)

    noise_le = multiply([noise, le])
    # noise_le = Dense(latent_dim)(noise_le)

    model = Model([noise, label], noise_le)

    return model



# from tensorflow.keras.experimental import CosineDecayRestarts

# initial_learning_rate = 1e-3
# steps_per_epoch = 200
# first_decay_steps = 20 * steps_per_epoch

# lr_decayed_fn = CosineDecayRestarts(
#     initial_learning_rate, 
#     first_decay_steps,
#     t_mul=2.0,  # 每个周期的长度翻倍
#     m_mul=1.0,  # 保持最大学习率不变
#     alpha=0.1   # 最小学习率为初始学习率的10%
# )

# optimizer = Adam(learning_rate=lr_decayed_fn)

optimizer = Adam(lr=1e-3, beta_1=0.5, beta_2=0.9)

from tensorflow.keras.callbacks import LearningRateScheduler
def lr_schedule(epoch):
    initial_lr = 1e-3
    if epoch < 10:
        return initial_lr * (epoch + 1) / 10
    else:
        return initial_lr * np.exp(0.1 * (10 - epoch))

lr_scheduler = LearningRateScheduler(lr_schedule)


from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

# 全局定义VGG模型，避免重复创建
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)

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


# Build Autoencoder
def autoencoder_trainer(encoder, decoder, embedding):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    latent = encoder(img)
    labeled_latent = embedding([latent, label])
    rec_img = decoder(labeled_latent)
    model = Model([img, label], rec_img)

    model.compile(optimizer=optimizer, loss=combined_loss)
    return model

# Train Autoencoder
en = encoder()
de = decoder()
em = embedding_labeled_latent()
ae = autoencoder_trainer(en, de, em)

# ae.fit([x_train, y_train], x_train,
#        epochs=80,
#        batch_size=128,
#        shuffle=True,
#        validation_data=([x_test, y_test], x_test),
#        callbacks=[lr_scheduler])

# en.save(path +'en.h5', save_format='h5')
# em.save(path +'em.h5', save_format='h5')
# de.save(path +'de.h5', save_format='h5')
# ae.save(path +'AE.h5', save_format='h5')

from tensorflow.keras.models import load_model
with tf.keras.utils.custom_object_scope(custom_objects):
    en = load_model(path +'en.h5')
    em = load_model(path +'em.h5')
    de = load_model(path +'de.h5')
    ae = load_model(path +'AE.h5')

# Show results of reconstructed images
decoded_imgs = ae.predict([x_test, y_test])
n = n_classes
plt.figure(figsize=(2*n, 4))
decoded_imgs = decoded_imgs*0.5 + 0.5
x_real = x_test*0.5 + 0.5

for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    if channel == 3:
        plt.imshow(x_real[y_test.reshape(-1)==i][0].reshape(img_size[0], img_size[0], channel))
    else:
        plt.imshow(x_real[y_test.reshape(-1)==i][0].reshape( img_size[0],img_size[0]))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    if channel == 3:
        plt.imshow(decoded_imgs[y_test.reshape(-1)==i][0].reshape(img_size[0], img_size[0], channel))
    else:
        plt.imshow(decoded_imgs[y_test.reshape(-1)==i][0].reshape(img_size[0], img_size[0]))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.show()

####################### Use the pre-trained Autoencoder #########################


# %% ----------------------------------- BAGAN-GP Part -----------------------------------------------------------------
# Refer to the WGAN-GP Architecture. https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan_gp.py
# Build our BAGAN-GP

class BAGAN_GP(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        gp_weight=10.0,
    ):
        super(BAGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.train_ratio = trainRatio
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(BAGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            #real_images = augment_image(data[0])
            real_images = data[0]
            labels = data[1]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        #print(batch_size)
        ########################### Train the Discriminator ###########################
        # For each batch, we are going to perform cwgan-like process
        for i in range(self.train_ratio):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
            wrong_labels = tf.random.uniform((batch_size,), 0, n_classes)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, fake_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, fake_labels], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([real_images, labels], training=True)
                # Get the logits for wrong label classification
                wrong_label_logits = self.discriminator([real_images, wrong_labels], training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits,
                                        wrong_label_logits=wrong_label_logits
                                        )

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        ########################### Train the Generator ###########################
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, fake_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, fake_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5, beta_2=0.9 are recommended


generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

discriminator_optimizer = Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9
)


# We refer to the DRAGAN loss function. https://github.com/kodalinaveen3/DRAGAN
# Define the loss functions to be used for discrimiator
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_logits, fake_logits, wrong_label_logits):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    wrong_label_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_label_logits, labels=tf.zeros_like(fake_logits)))

    return wrong_label_loss + fake_loss + real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss

# build generator with pretrained decoder and embedding
def generator_label(embedding, decoder):
    # # Embedding model needs to be trained along with GAN training
    # embedding.trainable = False

    label = Input((1,), dtype='int32')
    latent = Input((latent_dim,))

    labeled_latent = embedding([latent, label])
    gen_img = decoder(labeled_latent)
    model = Model([latent, label], gen_img)
    model.summary()
    return model

# Build discriminator with pre-trained Encoder
def build_discriminator(encoder):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    inter_output_model = Model(inputs=encoder.input, outputs=encoder.layers[-2].output)
    x = inter_output_model(img)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(4 * 4 * 256)(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[img, label], outputs=out)

    return model

# Build Discriminator without inheriting the pre-trained Encoder
# Similar to cWGAN
def discriminator_cwgan():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(img_size, name='input_img')
    label = Input((1,), dtype='int32')

    x = AttentionResidualEncoderBlock(32, strides=2)(img)

    x = AttentionResidualEncoderBlock(64, strides=2)(x)
    
    x = AttentionResidualEncoderBlock(128, strides=2)(x)

    x = AttentionResidualEncoderBlock(256, strides=2)(x)

    x = Flatten()(x)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(4 * 4 * 256,name='dense_1')(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512, name='dense_2')(x_y)

    out = Dense(1, name='dense_3')(x_y)

    model = Model(inputs=[img, label], outputs=out)
    model.summary()
    return model

# %% ----------------------------------- Compile Models ----------------------------------------------------------------
d_model = build_discriminator(en)  # initialized with Encoder
#d_model = discriminator_cwgan()  # without initialization
d_model.summary()
g_model = generator_label(em, de)  # initialized with Decoder and Embedding
g_model.summary()

with tf.keras.utils.custom_object_scope(custom_objects):
    d_model = load_model(path+'d_model_103.h5')
    g_model = load_model(path+'g_model_103.h5')

bagan_gp = BAGAN_GP(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
)

# Compile the model
bagan_gp.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)


# %% ----------------------------------- Start Training ----------------------------------------------------------------
# Plot/save generated images through training
def plt_img(generator, epoch):
    np.random.seed(42)
    latent_gen = np.random.normal(size=(n_classes, latent_dim))

    x_real = x_test * 0.5 + 0.5
    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test.reshape(-1)==i][4].reshape(img_size[0], img_size[0], channel))
        else:
            plt.imshow(x_real[y_test.reshape(-1) == i][4].reshape(img_size[0], img_size[0]))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
            decoded_imgs = decoded_imgs * 0.5 + 0.5
            # display generation
            ax = plt.subplot(n+1, n, (i+1)*n + 1 + c)
            if channel == 3:
                plt.imshow(decoded_imgs[i].reshape(img_size[0], img_size[0], channel))
            else:
                plt.imshow(decoded_imgs[i].reshape(img_size[0], img_size[0]))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig(path + 'results/generated_plot_%d.png' % epoch)
    #plt.show()
    return

# make directory to store results
os.system('mkdir results')

# Record the loss
d_loss_history = []
g_loss_history = []

############################# Start training #############################
LEARNING_STEPS = 200
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 50)
    bagan_gp.fit(x_train, y_train, batch_size=128, epochs=10)
    d_loss_history += bagan_gp.history.history['d_loss']
    g_loss_history += bagan_gp.history.history['g_loss']
    if (learning_step+1) % 4 == 0:
        plt_img(bagan_gp.generator, learning_step)
        bagan_gp.discriminator.save(path + 'd_model_%d.h5' %learning_step)
        bagan_gp.generator.save(path + 'g_model_%d.h5' %learning_step)

############################# Display performance #############################
# plot loss of G and D
plt.plot(d_loss_history, label='D')
plt.plot(g_loss_history, label='G')
plt.legend()
plt.show()

# save gif
import imageio
ims = []
for i in range(LEARNING_STEPS):
    fname = 'generated_plot_%d.png' % i
    dir = 'results/'
    if fname in os.listdir(dir):
        print('loading png...', i)
        im = imageio.imread(dir + fname, 'png')
        ims.append(im)
print('saving as gif...')
imageio.mimsave(dir + 'training_demo.gif', ims, duration=20)