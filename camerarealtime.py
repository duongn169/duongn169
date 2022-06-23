from tkinter import Frame
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
#############################################
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

threshold = 0.75 #THRESHOLD của Xác Suất
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Chiều rộng cửa sổ
cap.set(4, 480) # Chiều dài cửa sổ
cap.set(10, 180) # Độ sáng
# IMPORT TRAINED MODEL
model = load_model('model1.h5')

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Asagi'
    elif classNo == 1:
        return 'Bekko'
    elif classNo == 2:
        return 'Doitsu koi'
    elif classNo == 3:
        return 'Ghosiki'
    elif classNo == 4:
        return 'Goromo'
    elif classNo == 5:
        return 'Hikarimoyo'
    elif classNo == 6:
        return 'Hikarimuji mono'
    elif classNo == 7:
        return 'Hikariutsuri'
    elif classNo == 8:
        return 'Kanoko koi'
    elif classNo == 9:
        return 'Yamato Nishiki'
    elif classNo == 10:
        return 'Kawarimono'
    elif classNo == 11:
        return 'Kin Ginrin'
    elif classNo == 12:
        return 'Kohaku'
    elif classNo == 13:
        return 'Showa Sanshoku'
    elif classNo == 14:
        return 'Shusui'
    elif classNo == 15:
        return 'Taischo Sanke'
    elif classNo == 16:
        return 'Tancho'
    elif classNo == 17:
        return 'Utsuri'
    
while True:
    # Đọc ảnh từ Webcame
    success, imgOrignal = cap.read()

    # Xử lý ảnh
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Tiến hành dự đoán kết quả
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
                font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + " %", (180, 75),
                font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detecting Koi Fish", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('r'):
        break