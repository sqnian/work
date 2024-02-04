# from PIL import Image
 
# # 打开原始彩色图像文件
# image = Image.open('test.jpg')
 
# # 将彩色图像转换为灰度图像
# grayscale_image = image.convert('L')
 
# # 保存灰度图像到新的文件
# grayscale_image.save('output_grayscale_image.jpg')
 
# # 显示灰度图像
# grayscale_image.show()


import cv2

# 读取彩色图像
image = cv2.imread('one.jpg')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为二值图像
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 保存二值图像
cv2.imwrite('binary_image.jpg', binary_image)
