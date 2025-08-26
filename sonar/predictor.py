import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.layers import Conv2D, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
# Get the feature output from the pre-trained model ResNet50
#pretrained_model = load_model('./transfer_resnet50model.h5')
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
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

#path = 'C:/Users/70916/Desktop/bagangp/FLSMDD/'
path = 'C:/Users/70916/Desktop/bagangp/NKSID/'

model = build_model(path, input_shape=(64,64,3), n_classes=8)

with tf.keras.utils.custom_object_scope(custom_objects):
    
    #model = load_model('FLSMDD_attention')
    #model = load_model('FLSMDD_attention_transfer')
    #model = load_model('FLSMDD_attention_transfer_balanced')
    #model.load_weights('best_model.h5')

    #model = load_model('NKSID_attention')
    #model = load_model('NKSID_attention_transfer')
    model = load_model('NKSID_attention_transfer_balanced')
    model = load_model('NKSID_attention_map')

# 加载保存的模型
#model = tf.keras.models.load_model('path_to_your_model')  # 替换为你的模型路径

path = 'C:/Users/70916/Desktop/bagangp/NKSID/'
labels = ['big_propeller', 'cylinder', 'fishing_net','floats', 
       'iron_pipeline', 'small_propeller', 'soft_pipeline','tire']

#path = 'C:/Users/70916/Desktop/bagangp/FLSMDD/'
# labels = ['bottle', 'can', 'chain','drink-carton', 'hook', 
#             'propeller', 'shampoo-bottle', 'standing-bottle','tire', 'valve']

# labels = ['big_propeller', 'cylinder', 'fishing_net','floats', 
#        'iron_pipeline', 'small_propeller', 'soft_pipeline','tire']


print(labels)

x_train_all = np.load(path + 'x_train1.npy')/255
y_train_all = np.load(path + 'y_train1.npy')
X_test = np.load(path + 'x_val1.npy')/255
y_test = np.load(path + 'y_val1.npy')
# 使用模型进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test
# 计算性能指标


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
average_type = 'macro'  # 可选'macro'(平均)/'micro'(全局)/'weighted'(加权平均)
auc_prob = y_pred
#print(auc_prob.shape)
# 计算各项指标
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred_classes),
    "Precision": precision_score(y_true, y_pred_classes, average=average_type),
    "Recall": recall_score(y_true, y_pred_classes, average=average_type),
    "F1 Score": f1_score(y_true, y_pred_classes, average=average_type),
    "AUC": roc_auc_score(y_true, auc_prob, multi_class='ovr')  # 多分类使用'ovr'策略
}

# 打印结果（保留4位小数）
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


#accuracy = accuracy_score(y_true, y_pred_classes)
#f1 = f1_score(y_true, y_pred_classes, average='macro')  # 宏平均F1

#print(f'准确率 (Accuracy): {accuracy:.4f}')
#print(f'F1 分数 (Macro): {f1:.4f}\n')

# 输出详细分类报告（包含精确率、召回率等）
#print("分类报告:")
#print(classification_report(y_true, y_pred_classes))