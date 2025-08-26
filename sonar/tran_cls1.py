import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback,ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


print(device_lib.list_local_devices())

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


# 注册自定义对象
custom_objects = {
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
    'CBAM': CBAM,
    'ConvBlock': ConvBlock,
    'AttentionResidualEncoderBlock': AttentionResidualEncoderBlock,
    'AttentionResidualDecoderBlock': AttentionResidualDecoderBlock,
}

tf.keras.utils.get_custom_objects().update(custom_objects)

def build_model(path, input_shape, n_classes):
    with tf.keras.utils.custom_object_scope(custom_objects):
        encoder = load_model(path +'en.h5')
    img = Input( input_shape)
    inter_output_model = Model(inputs=encoder.input, outputs=encoder.layers[-1].output)
    x = inter_output_model(img)
    x = Dense(n_classes, kernel_initializer='he_normal')(x)
    out = Softmax()(x)
    model = Model(inputs=img, outputs=out)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
   # model.summary()

    return model


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir):
        super().__init__()
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        self.first_epoch = True

    def on_train_begin(self, logs=None):
        plt.ion()
        if logs is None:
            logs = {}
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.f, self.axs = plt.subplots(1, 3, figsize=(13, 4))

    def on_train_end(self, logs=None):
        self.f.savefig(f"{self.i_dir}/metrics")

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        acc = max(self.max_acc, round(logs.get("accuracy"), 4))
        val_acc = max(self.max_val_acc, round(logs.get("val_accuracy"), 4))
        loss = min(self.min_loss, round(logs.get("loss"), 4))
        val_loss = min(self.min_val_loss, round(logs.get("val_loss"), 4))

        if acc == self.max_acc:
            self.acc_ep += 1
        else:
            self.acc_ep = 0
        if val_acc == self.max_val_acc:
            self.val_acc_ep += 1
        else:
            self.val_acc_ep = 0

        if loss == self.min_loss:
            self.loss_ep += 1
        else:
            self.loss_ep = 0

        if val_loss == self.min_val_loss:
            self.val_loss_ep += 1
        else:
            self.val_loss_ep = 0

        self.max_acc = acc
        self.max_val_acc = val_acc
        self.min_loss = loss
        self.min_val_loss = val_loss

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <25}: {self.max_acc:.4f}, epoch {self.acc_ep}\n{'Max val_accuracy': <25}: {self.max_val_acc:.4f}, epoch {self.val_acc_ep}")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <25}: {self.min_loss:.4f}, epoch {self.loss_ep}\n{'Min val_loss': <25}: {self.min_val_loss:.4f}, epoch {self.val_loss_ep}")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()



def main():

    path = 'C:/Users/70916/Desktop/bagangp/NKSID/'
    labels = ['big_propeller', 'cylinder', 'fishing_net','floats', 
       'iron_pipeline', 'small_propeller', 'soft_pipeline','tire']

    # path = 'C:/Users/70916/Desktop/bagangp/FLSMDD/'
    # labels = ['bottle', 'can', 'chain','drink-carton', 'hook', 
    #         'propeller', 'shampoo-bottle', 'standing-bottle','tire', 'valve']

    print(labels)

   
    x_train_all = np.load(path + 'x_train1.npy')
    y_train_all = np.load(path + 'y_train1.npy')
    x_test_all = np.load(path + 'x_val1.npy')
    y_test_all = np.load(path + 'y_val1.npy')
    key = np.unique(y_train_all)
    results = {k: np.sum(y_train_all == k) for k in key}
    print('y_train:', results)
    key = np.unique(y_test_all)
    results = {k: np.sum(y_test_all == k) for k in key}
    print('y_test:', results)

    with tf.keras.utils.custom_object_scope(custom_objects):
            #generator = load_model(path+'g_model_103.h5')
            generator = load_model(path+'g_model_103.h5')

    n = len(labels)
    key = np.unique(y_train_all)
    results = {k: np.sum(y_train_all == k) for k in key}
    M = max(results.values())

    gen_labels = []
    for c in range(n):
        sampled_size =  M - results[c]
        sampled_labels = [c] * sampled_size
        gen_labels += sampled_labels
    gen_labels = np.array(gen_labels)

    noise = np.random.normal(0, 1, (gen_labels.shape[0], generator.input_shape[0][1]))
    gen_samples = generator.predict([noise, gen_labels])
    gen_samples = gen_samples * 122.5 + 122.5
    #print(gen_samples.shape)

    x_train_all = np.concatenate((x_train_all, gen_samples))
    y_train_all = np.concatenate((y_train_all, gen_labels))
    print(x_train_all.shape)
    print(y_train_all.shape)
    key = np.unique(y_train_all)
    results = {k: np.sum(y_train_all == k) for k in key}
    print('y_train:', results)
    key = np.unique(y_test_all)
    results = {k: np.sum(y_test_all == k) for k in key}
    print('y_test:', results)

    x_train = x_train_all
    y_train = y_train_all
    x_test = x_test_all
    y_test = y_test_all

    classes = np.unique(y_train)
    K = len(classes)
    y_train = to_categorical(y_train, K)
    y_test = to_categorical(y_test, K)
    print(y_test.shape)
    print("number of classes:", K)
    print("input shape", x_train[0].shape)

    
    model = build_model(path, input_shape=(64,64,3), n_classes=K)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, shuffle=True)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
       # zoom_range=0.05,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        vertical_flip=True,
        horizontal_flip=True,
    )  # randomly flip images

    validgen = ImageDataGenerator(rescale=1. / 255, )


    batch_size = 64
    epochs = 300

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    #valid_gen = validgen.flow(x_val, y_val, batch_size=batch_size)
    valid_gen = validgen.flow(x_test, y_test, batch_size=batch_size)

    checkpoint = ModelCheckpoint(filepath = 'best_model.h5',
                                 monitor = 'val_loss',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='min',
                                 verbose=1)
    #lr_scheduler = LearningRateScheduler(lr_schedule)
    #print(x_test.shape[0]//batch_size)

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=x_test.shape[0] // batch_size,
        callbacks=[
            #EarlyStopping(monitor="val_loss", verbose=1, patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6, verbose=1, min_lr=0.00001),
            PlotProgress("data"),
            checkpoint
            #lr_scheduler
        ],
    )
    
    model.evaluate(x_train/255, y_train)
    model.evaluate(x_test/255, y_test)
    #model.save("NKSID-" + str(a[0]) + str(a[1]) + "_balanced")
    #model.save("NKSID-" + str(a[0])  + "_balanced")
    model.save("NKSID_attention_transfer_balanced")

    plt.plot(history.history['accuracy'], label='acc', color='red')
    plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
