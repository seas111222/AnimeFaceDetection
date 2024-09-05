import cv2
import os


def list_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            face_detect(file_path, cascade_name,output_folder)
def face_detect(file_name, cascade_name,output_folder):
    img = cv2.imread(file_name)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
    face_cascade = cv2.CascadeClassifier(cascade_name)  # 加载级联分类器
    faces = face_cascade.detectMultiScale(img)  # 多尺度检测
    for i,(x, y, w, h) in enumerate(faces):  # 遍历所有检测到的动漫脸
        face_img = img[y:y + h, x:x + w]
        # 保存提取出的动漫脸部分
        face_file_name = os.path.join(output_folder, f'face_{i + 1}.jpg')
        cv2.imwrite(face_file_name, face_img)
    #     img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)  # 绘制矩形框
    # cv2.imshow('Face detection', img)  # 检测效果预览
    # cv2.waitKey(0)  # 保持窗口显示

# file_name = 'img/anime/test_1.jpg'
cascade_name = 'data/lbpcascades/anime/lbpcascade_animeface.xml'
output_folder = 'result/anime'
directory_path = 'img/anime'
if __name__ == "__main__":
    list_files_in_directory(directory_path)
    # face_detect(file_name, cascade_name,output_folder)