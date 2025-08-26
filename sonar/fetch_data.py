# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% --------------------------------------- Data Prep -----------------------------------------------------------------

# folder = '../NKSID/'
# path = './NKSID/'
# dict = {'big_propeller': 0, 'cylinder': 1, 'fishing_net': 2,'floats':3, 
#         'iron_pipeline':4, 'small_propeller':5, 'soft_pipeline':6,'tire':7}

folder = '../FLSMDD/'
path = './FLSMDD/'
dict = {'bottle': 0, 'can': 1, 'chain': 2,'drink-carton':3, 'hook':4, 
        'propeller':5, 'shampoo-bottle':6, 'standing-bottle':7,'tire':8, 'valve':9}

img_data = []
img_label = []

def letterbox(image, new_size=(640, 640), color=(0,0,0)):
    """
    精确缩放图像到指定尺寸，保持宽高比
    
    参数:
        image: 输入图像 (numpy array)
        new_size: 目标尺寸 (width, height)
        color: 填充颜色 (B,G,R)
    
    返回:
        resized_image: 调整大小后的图像
        ratio: 缩放比例
        (dw, dh): 填充像素值
    """
    
    # 获取原始图像尺寸
    old_height, old_width = image.shape[:2]
    
    # 目标尺寸
    new_width, new_height = new_size
    
    # 计算缩放比例
    scale = min(new_width / old_width, new_height / old_height)
    
    # 计算新的图像尺寸
    new_w = int(old_width * scale)
    new_h = int(old_height * scale)
    
    # 调整图像大小
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建全填充图像
    output_image = np.full((new_height, new_width, 3), color, dtype=np.uint8)
    #output_image = np.full((new_height, new_width), color, dtype=np.uint8)
    
    # 计算居中位置
    x_start = (new_width - new_w) // 2
    y_start = (new_height - new_h) // 2
    
    # 将调整大小的图像放置在中心
    output_image[y_start:y_start+new_h, x_start:x_start+new_w] = resized_image
    
    return output_image, scale, (x_start, y_start)


# if __name__ == "__main__":
#     img = cv2.imread("../NKSID/fishing_net/img_1.jpg")
#     resized_img, ratio, (dw, dh) = letterbox(img, new_size=(128, 128))
#     cv2.imshow("Original", img)
#     cv2.imshow("Resized", resized_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


for class_name in os.listdir(folder):
    for image_name in os.listdir(os.path.join(folder, class_name)):
        imgpath = os.path.join(folder, class_name, image_name)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img,_,_ = letterbox(img, new_size=(64, 64))
        #img = np.expand_dims(img,-1)
        print(img.shape)
        img_data.append(img)
        img_label.append(dict[class_name])

image_data = np.array(img_data)
image_label = np.array(img_label)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataAugmentationSaver:
    def __init__(self, 
                 rotation_range=20,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 brightness_range=(0.8, 1.2),
                 horizontal_flip=True,
                 vertical_flip=True):
        """
        数据增强器初始化
        """
        self.datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            fill_mode='nearest'
        )
    
    def augment(self, X, y, n_augmented_per_image=10):
    
        augmented_images = []
        augmented_labels = []
        
        # 保存原始数据
        augmented_images.extend(X)
        #if y is not None:
        augmented_labels.extend(y)
        
        # 遍历原始图像进行增强
        for i, img in enumerate(X):
            aug_count = 0
            for batch in self.datagen.flow(np.expand_dims(img, 0), batch_size=1):
                if aug_count >= n_augmented_per_image:
                    break
                
                aug_img = batch[0]
                augmented_images.append(aug_img)
                
                #if y is not None:
                augmented_labels.append(y[i])
                
                aug_count += 1
        
        # 转换为numpy数组
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        return augmented_images, augmented_labels
       

# 创建数据增强器
augmenter = DataAugmentationSaver(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    vertical_flip=True
)



#images, labels = augmenter.augment(image_data, image_label, n_augmented_per_image=6)
images, labels = image_data, image_label
print(images.shape)
print(labels.shape)

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
from collections import Counter
print(Counter(y_val))
print(Counter(y_train))

#image_label = labels[:, np.newaxis]
print(x_train.shape)
print(y_val.shape)

# %% --------------------------------------- Save as .npy --------------------------------------------------------------
# Save

np.save(path + "x_train1.npy", x_train)
np.save(path + "y_train1.npy", y_train)
np.save(path + "x_val1.npy", x_val)
np.save(path + "y_val1.npy", y_val)