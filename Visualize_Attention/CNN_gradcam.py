import os
import numpy as np
import cv2
import keras.backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import array_to_img, img_to_array, load_img

# load image
imgs=[]
img_path='img'
for im in os.listdir(img_path):
  img = cv2.imread(img_path+'/'+im)
  if img is not None:
    img = cv2.resize(img, (224,224))
    imgs.append(img)

imgs = np.array(imgs)


from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import array_to_img, img_to_array, load_img

layer_name ="block5_conv3"
x= imgs[0]
preprocessed_input = x.reshape(1, 224,224,3) / 255.0


# 予測クラスの算出
model = VGG16(weights='imagenet')
predictions = model.predict(preprocessed_input)
class_idx = np.argmax(predictions[0])
class_output = model.output[:, class_idx]  # (?, )   model.output==(1000, )

conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット (?, 14, 14, 512)
grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す  (?, 14, 14, 512)
gradient_function = K.function([model.input], [conv_output, grads])

output, grads_val = gradient_function([preprocessed_input])
output, grads_val = output[0], grads_val[0] # both (14, 14, 512)

# 重みを平均化して、レイヤーのアウトプットに乗じる
weights = np.mean(grads_val, axis=(0, 1))
cam = np.dot(output, weights)

cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
cam = np.maximum(cam, 0)
cam = cam / cam.max() # (224, 224) grayscale

jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成
array_to_img(jetcam)
