import numpy as np
import cv2
import os
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

# 用于从文件夹加载图片数据
def load_images_from_folder(folder):
    images = []
    labels = []
    label = 0
    for person_folder in os.listdir(folder):
        person_path = os.path.join(folder, person_folder)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img.flatten())
                    labels.append(label)
            label += 1
    return np.array(images), np.array(labels)

# 从FERET数据库加载数据
database_path = "face_data/feret_data"  #项目中的feret数据库相对路径
images, labels = load_images_from_folder(database_path)

# 检查数据加载是否正确
print(f"共加载 {len(images)} 张图片，共有 {len(np.unique(labels))} 类。")

# 归一化
images = normalize(images, axis=1)

# 使用PCA降维
n_components = 1260  # 增加PCA组件数量
pca = PCA(n_components=n_components)
images_pca = pca.fit_transform(images)

# 检查PCA降维结果
print(f"PCA降维后的数据形状: {images_pca.shape}")

# 划分训练和测试集
unique_labels = np.unique(labels)
test_ratio = 0.2
test_indices = []
train_indices = []

for label in unique_labels:
    label_indices = np.where(labels == label)[0]
    num_test_samples = int(len(label_indices) * test_ratio)
    test_indices.extend(random.sample(list(label_indices), num_test_samples))
    train_indices.extend(list(set(label_indices) - set(test_indices)))

train_images = images_pca[train_indices]
train_labels = labels[train_indices]
test_images = images_pca[test_indices]
test_labels = labels[test_indices]

print(f"训练集大小: {len(train_images)}, 测试集大小: {len(test_images)}")

# 定义一个Lasso模型用于稀疏表示
lasso = Lasso(alpha=0.001)

# 或者使用OMP模型
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=50)  # 减少非零系数数量

# 稀疏表示分类
def sparse_representation_classification(test_image, train_images, train_labels, model):
    model.fit(train_images.T, test_image)
    coefficients = model.coef_
    min_error = float('inf')
    predicted_label = -1
    for label in np.unique(train_labels):
        class_indices = np.where(train_labels == label)[0]
        class_coefficients = coefficients[class_indices]
        class_error = np.linalg.norm(test_image - np.dot(train_images[class_indices].T, class_coefficients))
        if class_error < min_error:
            min_error = class_error
            predicted_label = label
    return predicted_label

# 测试
predictions = []
for i in range(len(test_images)):
    predicted_label = sparse_representation_classification(test_images[i], train_images, train_labels, omp)  # 使用OMP模型
    predictions.append(predicted_label)
    if i % 10 == 0:
        print(f"测试进度: {i}/{len(test_images)}")

# 计算准确率

# 打印预测和真实标签对比
for i in range(183):
    print(f"预测: {predictions[i]}, 真实: {test_labels[i]}")

accuracy = accuracy_score(test_labels, predictions)
print(f'识别准确率: {accuracy * 100:.2f}%')
