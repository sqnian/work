from PIL import Image

# 打开png图片
png_image = Image.open('test.png')
png_image = png_image.convert('RGB')
# 将png图片保存为jpg格式
png_image.save('test.jpg', 'JPEG')

# 关闭图片
png_image.close()


"""
png图片是4通道RGBA图像，
具有4个通道（红色、绿色、蓝色和透明度），
用于表示彩色图像以及透明度信息。
"""