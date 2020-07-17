import cv2
import numpy as np
import pytesseract as pt
from pyzbar import pyzbar

# 對圖片做文字辨識
# image = cv2.imread("image_3.png", 1)
#
# word = pt.image_to_string(image, "001", "")
# print(word)

# 對 QR Code 或是 code39、code128 做辨識
# 加上攝影功能，可以使攝影機變成條碼掃描機
# image = cv2.imread("hard.png", 1)
#
# code = pyzbar.decode(image)
# for d in code:
#     print("條碼類型", d.type)
#     try:
#         print("結果文字", d.data.decode("utf-8").encode('sjis').decode("utf-8"))
#     except:
#         print("結果文字", d.data.decode("utf-8"))
#     x, y, w, h = d.rect
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,225), 2)
#     print("======================================================")
#
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 人臉辨識
# image =cv2.imread("people.jpg", 1)
#
# control = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
#
# return_ =control.detectMultiScale(
#         image,
#         minNeighbors = 2,
#         minSize = (10,10))
# for x, y, w,h in return_:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,225), 2)
#
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



image =cv2.imread("classifier_test/Data/Image.jpg", 1)

control = cv2.CascadeClassifier("classifier_test/xml/cascade.xml")

return_ =control.detectMultiScale(
        image,
        minNeighbors = 2,
        minSize = (10,10))
for x, y, w,h in return_:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,225), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()