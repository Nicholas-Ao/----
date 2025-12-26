"""
手写数字识别实验 - 卷积神经网络实现
使用MindSpore框架
"""

import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU
os.environ['GLOG_v'] = '3'  # 减少日志输出
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import context, Tensor, Model
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common.initializer import Normal
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# ==================== 子任务1：数据准备与预处理 ====================
def prepare_mnist_data(data_path="./data"):
    """
    加载并预处理MNIST数据集
    """
    # 创建数据集目录
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("请将MNIST数据集放置在 ./data 目录下")
        print("下载链接: http://yann.lecun.com/exdb/mnist/")
        print("或百度网盘: https://pan.baidu.com/s/1eKVYon0BzSRe0V6xUfaqBQ 提取码: psf5")
        return None
    
    # 定义数据集路径
    train_data_path = os.path.join(data_path, "train-images-idx3-ubyte")
    train_label_path = os.path.join(data_path, "train-labels-idx1-ubyte")
    test_data_path = os.path.join(data_path, "t10k-images-idx3-ubyte")
    test_label_path = os.path.join(data_path, "t10k-labels-idx1-ubyte")
    
    # 如果没有下载数据集，则使用MindSpore内置的MNIST数据集
    if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
        print("使用MindSpore内置的MNIST数据集...")
        
        # 加载MNIST数据集
        train_dataset = ds.MnistDataset(dataset_dir=data_path, usage='train', shuffle=True)
        test_dataset = ds.MnistDataset(dataset_dir=data_path, usage='test', shuffle=False)
        
        return train_dataset, test_dataset
    
    return None

def create_dataset(dataset, batch_size=32, usage='train'):
    """
    创建数据管道
    """
    # 定义图像预处理操作
    if usage == 'train':
        transform = [
            vision.Rescale(1.0 / 255.0, 0.0),  # 归一化到[0,1]
            vision.Normalize(mean=(0.1307,), std=(0.3081,)),  # MNIST标准化参数
            vision.HWC2CHW()  # 从(H,W,C)转换为(C,H,W)
        ]
    else:
        transform = [
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            vision.HWC2CHW()
        ]
    
    # 应用转换
    dataset = dataset.map(operations=transform, input_columns="image")
    
    # 类型转换
    type_cast_op = transforms.TypeCast(ms.int32)
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    
    # 批处理
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

# ==================== 子任务2：模型设计 ====================
class CNN(nn.Cell):
    """卷积神经网络模型"""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 卷积层1: 输入通道1, 输出通道32, 卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, 3, pad_mode='valid', has_bias=True, weight_init=Normal(0.02))
        self.relu1 = nn.ReLU()
        
        # 池化层1: 最大池化2x2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 输入通道32, 输出通道64, 卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='valid', has_bias=True, weight_init=Normal(0.02))
        self.relu2 = nn.ReLU()
        
        # 池化层2: 最大池化2x2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 全连接层1
        self.fc1 = nn.Dense(64 * 5 * 5, 128, weight_init=Normal(0.02))
        self.relu3 = nn.ReLU()
        
        # Dropout层 (用于防止过拟合)
        self.dropout = nn.Dropout(keep_prob=0.5)
        
        # 全连接层2 (输出层)
        self.fc2 = nn.Dense(128, num_classes, weight_init=Normal(0.02))
        
    def construct(self, x):
        # 卷积层1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        
        # 卷积层2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        
        # 全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== 模型训练与评估 ====================
def train_model(model, train_dataset, test_dataset, epoch_num=10):
    """
    训练模型
    """
    # 定义损失函数
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 定义优化器
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
    
    # 定义模型
    net = Model(model, loss_fn, optimizer, metrics={'accuracy'})
    
    # 定义回调函数
    config_ck = CheckpointConfig(saved_network=model)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_mnist", config=config_ck)
    loss_cb = LossMonitor(per_print_times=100)
    
    print("开始训练模型...")
    print("=" * 50)
    
    # 训练模型
    net.train(epoch_num, train_dataset, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=False)
    
    # 评估模型
    print("\n评估模型在测试集上的性能...")
    acc = net.eval(test_dataset, dataset_sink_mode=False)
    print(f"测试集准确率: {acc['accuracy']:.4f}")
    
    return net

# ==================== 子任务3：模型应用与测试 ====================
def preprocess_custom_image(image_path, img_size=(28, 28)):
    """
    预处理自定义手写数字图片
    """
    # 打开图片并转换为灰度
    img = Image.open(image_path).convert('L')
    
    # 调整大小
    img = img.resize(img_size)
    
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 反转颜色 (MNIST是黑底白字)
    img_array = 255 - img_array
    
    # 归一化
    img_array = img_array / 255.0
    
    # 标准化 (使用与训练数据相同的参数)
    img_array = (img_array - 0.1307) / 0.3081
    
    # 添加批次和通道维度
    img_array = img_array[np.newaxis, np.newaxis, :, :]
    
    # 转换为Tensor
    img_tensor = Tensor(img_array, dtype=ms.float32)
    
    return img_tensor, img

def predict_custom_image(model, image_path):
    """
    预测自定义图片
    """
    # 预处理图片
    img_tensor, original_img = preprocess_custom_image(image_path)
    
    # 预测
    model.set_train(False)
    output = model(img_tensor)
    
    # 获取预测结果
    pred = np.argmax(output.asnumpy(), axis=1)
    probabilities = nn.Softmax()(output).asnumpy()[0]
    
    return pred[0], probabilities, original_img

# ==================== 可视化工具 ====================
def visualize_predictions(model, dataset, num_samples=10):
    """
    可视化预测结果
    """
    data_iter = dataset.create_dict_iterator()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, batch in enumerate(data_iter):
        if i >= num_samples:
            break
            
        images = batch['image'].asnumpy()
        labels = batch['label'].asnumpy()
        
        # 预测
        model.set_train(False)
        output = model(Tensor(images))
        pred = np.argmax(output.asnumpy(), axis=1)
        
        # 显示图片
        img = images[0].squeeze()  # 移除通道维度
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {labels[0]}, Pred: {pred[0]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    绘制训练历史
    """
    if not history:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, dataset, num_classes=10):
    """
    绘制混淆矩阵
    """
    all_preds = []
    all_labels = []
    
    data_iter = dataset.create_dict_iterator()
    
    for batch in data_iter:
        images = batch['image']
        labels = batch['label'].asnumpy()
        
        # 预测
        model.set_train(False)
        output = model(images)
        preds = np.argmax(output.asnumpy(), axis=1)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)]))

# ==================== 高级功能：数据增强 ====================
def create_dataset_with_augmentation(dataset, batch_size=32):
    """
    创建带有数据增强的数据集
    """
    # 数据增强操作
    transform = [
        vision.RandomRotation(degrees=15),  # 随机旋转
        vision.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    
    dataset = dataset.map(operations=transform, input_columns="image")
    
    type_cast_op = transforms.TypeCast(ms.int32)
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

# ==================== 主程序 ====================
def main():
    """
    主函数
    """
    print("=" * 60)
    print("手写数字识别实验 - CNN实现")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 准备MNIST数据集...")
    train_dataset_raw, test_dataset_raw = prepare_mnist_data()
    
    if train_dataset_raw is None:
        # 如果内置数据集不可用，使用随机数据演示
        print("注意: 使用随机生成的数据进行演示")
        print("实际使用时，请下载MNIST数据集")
        return
    
    # 2. 创建数据管道
    print("\n2. 创建数据管道...")
    train_dataset = create_dataset(train_dataset_raw, batch_size=64, usage='train')
    test_dataset = create_dataset(test_dataset_raw, batch_size=64, usage='test')
    
    # 3. 创建模型
    print("\n3. 创建CNN模型...")
    model = CNN(num_classes=10)
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    trained_model = train_model(model, train_dataset, test_dataset, epoch_num=5)
    
    # 5. 可视化结果
    print("\n5. 可视化预测结果...")
    visualize_predictions(model, test_dataset, num_samples=10)
    
    # 6. 混淆矩阵
    print("\n6. 生成混淆矩阵...")
    plot_confusion_matrix(model, test_dataset)
    
    # 7. 测试自定义图片
    print("\n7. 测试自定义手写数字图片...")
    custom_images_dir = "./custom_images"
    
    if os.path.exists(custom_images_dir):
        custom_images = [f for f in os.listdir(custom_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if custom_images:
            fig, axes = plt.subplots(1, min(5, len(custom_images)), figsize=(15, 3))
            
            for i, img_file in enumerate(custom_images[:5]):
                img_path = os.path.join(custom_images_dir, img_file)
                
                try:
                    pred, probabilities, original_img = predict_custom_image(model, img_path)
                    
                    # 显示图片
                    axes[i].imshow(original_img, cmap='gray')
                    axes[i].set_title(f'Predicted: {pred}\nProb: {probabilities[pred]:.2f}')
                    axes[i].axis('off')
                    
                    print(f"图片: {img_file}, 预测数字: {pred}")
                    print(f"各类别概率: {[f'{p:.3f}' for p in probabilities]}")
                    
                except Exception as e:
                    print(f"处理图片 {img_file} 时出错: {e}")
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"目录 {custom_images_dir} 中没有找到图片")
            print("请在该目录下放置手写数字图片进行测试")
    else:
        print(f"目录 {custom_images_dir} 不存在")
        print("请创建该目录并放置手写数字图片进行测试")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

# ==================== 高阶任务：模型优化 ====================
def advanced_experiment():
    """
    高阶任务：尝试不同的网络结构和超参数
    """
    print("\n" + "=" * 60)
    print("高阶任务：模型优化实验")
    print("=" * 60)
    
    # 加载数据
    train_dataset_raw, test_dataset_raw = prepare_mnist_data()
    
    if train_dataset_raw is None:
        return
    
    # 不同的网络结构
    class AdvancedCNN(nn.Cell):
        """更深的CNN模型"""
        def __init__(self, num_classes=10):
            super(AdvancedCNN, self).__init__()
            
            # 卷积块1
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1, pad_mode='pad', has_bias=True)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            
            # 卷积块2
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, pad_mode='pad', has_bias=True)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 卷积块3
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1, pad_mode='pad', has_bias=True)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            
            # 卷积块4
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1, pad_mode='pad', has_bias=True)
            self.bn4 = nn.BatchNorm2d(256)
            self.relu4 = nn.ReLU()
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 全局平均池化
            self.avg_pool = nn.AvgPool2d(kernel_size=7)
            
            # 全连接层
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(keep_prob=0.5)
            self.fc = nn.Dense(256, num_classes)
            
        def construct(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.max_pool1(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.max_pool2(x)
            
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            
            return x
    
    # 创建数据集
    print("1. 创建带有数据增强的训练集...")
    train_dataset_aug = create_dataset_with_augmentation(train_dataset_raw, batch_size=64)
    test_dataset = create_dataset(test_dataset_raw, batch_size=64, usage='test')
    
    # 创建更深的模型
    print("2. 创建更深的CNN模型...")
    advanced_model = AdvancedCNN(num_classes=10)
    
    # 定义损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(advanced_model.trainable_params(), learning_rate=0.0005)
    
    # 训练模型
    print("3. 训练高级模型...")
    net = Model(advanced_model, loss_fn, optimizer, metrics={'accuracy'})
    
    # 训练更多轮次
    net.train(10, train_dataset_aug, callbacks=[LossMonitor(per_print_times=100)], dataset_sink_mode=False)
    
    # 评估
    acc = net.eval(test_dataset, dataset_sink_mode=False)
    print(f"高级模型测试准确率: {acc['accuracy']:.4f}")
    
    # 比较不同学习率
    print("\n4. 比较不同学习率的效果...")
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        print(f"\n使用学习率: {lr}")
        model_simple = CNN(num_classes=10)
        optimizer = nn.Adam(model_simple.trainable_params(), learning_rate=lr)
        net_simple = Model(model_simple, loss_fn, optimizer, metrics={'accuracy'})
        
        # 只训练少量批次以快速比较
        net_simple.train(1, train_dataset_aug, callbacks=[], dataset_sink_mode=False)
        acc = net_simple.eval(test_dataset, dataset_sink_mode=False)
        print(f"学习率 {lr} 的准确率: {acc['accuracy']:.4f}")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 运行主实验
    main()
    
    # 运行高阶实验（可选）
    run_advanced = input("\n是否运行高阶实验？(y/n): ").lower()
    if run_advanced == 'y':
        advanced_experiment()
    
    print("\n实验完成！请参考实验报告要求撰写报告。")