import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import struct

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



if torch.cuda.is_available():
    print("GPU可用！设备名：", torch.cuda.get_device_name(0))
    print("系统CUDA版本：", torch._C._cuda_getCompiledVersion())
else:
    print("GPU不可用，当前用CPU运行")




# ==================== 子任务1：数据准备 ====================
def load_mnist_images(filename):
    """读取MNIST图像文件"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    """读取MNIST标签文件"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def prepare_data():
    """加载并预处理数据"""
    # 加载数据
    train_images = load_mnist_images('./mnist/train/train-images-idx3-ubyte')
    train_labels = load_mnist_labels('./mnist/train/train-labels-idx1-ubyte')
    test_images = load_mnist_images('./mnist/test/t10k-images-idx3-ubyte')
    test_labels = load_mnist_labels('./mnist/test/t10k-labels-idx1-ubyte')
    
    # 转换为Tensor并归一化
    train_images = torch.FloatTensor(train_images).unsqueeze(1) / 255.0  # [60000, 1, 28, 28]
    train_labels = torch.LongTensor(train_labels)
    test_images = torch.FloatTensor(test_images).unsqueeze(1) / 255.0    # [10000, 1, 28, 28]
    test_labels = torch.LongTensor(test_labels)
    
    # 划分训练集和验证集 
    val_size = 10000
    train_dataset = TensorDataset(train_images[val_size:], train_labels[val_size:])
    val_dataset = TensorDataset(train_images[:val_size], train_labels[:val_size])
    test_dataset = TensorDataset(test_images, test_labels)
    
    return train_dataset, val_dataset, test_dataset

def visualize_samples(dataset, num_samples=32):
    """可视化数据样本"""
    images, labels = dataset.tensors
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(num_samples):
        ax = axes[i//8, i%8]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./report/data_samples.png', dpi=150)
    plt.show()

# ==================== 子任务2：模型设计 ====================
class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 尺寸减半
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 经过两次池化：28->14->7
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))  # 32@14x14
        x = self.pool(F.relu(self.conv2(x)))  # 64@7x7
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
    
    return history, model

def evaluate_model(model, test_loader):
    """在测试集上评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

# ==================== 子任务3：模型应用 ====================
def preprocess_custom_image(image_path):
    """预处理自定义手写数字图片"""
    # 读取图片并转为灰度
    img = Image.open(image_path).convert('L')
    
    # 调整大小
    img = img.resize((28, 28))
    
    # 转换为numpy数组并归一化
    img_array = np.array(img).astype(np.float32)
    
    # 反色处理（如果背景是白色）
    if img_array.mean() > 127:  # 如果平均像素值较高，说明背景较亮
        img_array = 255 - img_array
    
    # 归一化到[0, 1]
    img_array = img_array / 255.0
    
    # 转换为Tensor并调整维度
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    
    return img_tensor

def predict_custom_image(model, image_path):
    """预测单张自定义图片"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 预处理图片
    img_tensor = preprocess_custom_image(image_path).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 显示结果
    img = Image.open(image_path).convert('L')
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f'预测结果: {predicted.item()}\n置信度: {confidence.item():.2%}')
    plt.axis('off')
    
    # 保存图片
    if not os.path.exists('./report'):
        os.makedirs('./report')
    plt.savefig(f'./report/pred_{os.path.basename(image_path)}', dpi=150)
    plt.show()
    
    return predicted.item(), confidence.item()

# ==================== 可视化函数 ====================
def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='训练损失', marker='o')
    ax1.plot(history['val_loss'], label='验证损失', marker='s')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['val_acc'], label='验证准确率', color='green', marker='^')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./report/training_history.png', dpi=150)
    plt.show()

# ==================== 主程序 ====================
def main():
    print("MNIST手写数字识别实验")
    print("=" * 50)
    
    # 1. 数据准备
    print("步骤1: 准备数据...")
    train_dataset, val_dataset, test_dataset = prepare_data()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 可视化样本
    print("可视化数据样本...")
    visualize_samples(train_dataset)
    
    # 2. 模型训练
    print("\n步骤2: 训练模型...")
    model = SimpleCNN()
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    history, trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=10, 
        lr=0.001
    )
    
    # 3. 模型评估
    print("\n步骤3: 评估模型...")
    accuracy = evaluate_model(trained_model, test_loader)
    
    # 4. 可视化训练过程
    print("\n步骤4: 可视化训练过程...")
    plot_training_history(history)
    
    # 5. 保存模型
    print("\n保存模型...")
    torch.save(trained_model.state_dict(), './mnist_cnn_model.pth')
    print("模型已保存为: ./mnist_cnn_model.pth")
    
    # 6. 自定义图片测试（可选）
    custom_test = input("\n是否测试自定义图片？(y/n): ")
    if custom_test.lower() == 'y':
        # 创建自定义图片目录
        if not os.path.exists('./custom_digits'):
            os.makedirs('./custom_digits')
            print("请在 ./custom_digits 目录中放入你的手写数字图片(28x28像素)")
            input("放入图片后按Enter键继续...")
        
        # 加载模型
        model.load_state_dict(torch.load('./mnist_cnn_model.pth'))
        
        # 测试所有图片
        for filename in os.listdir('./custom_digits'):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join('./custom_digits', filename)
                print(f"\n测试图片: {filename}")
                pred, conf = predict_custom_image(model, img_path)
                print(f"  预测数字: {pred}, 置信度: {conf:.2%}")
    
    print("\n实验完成！")
    print(f"最终测试准确率: {accuracy:.2f}%")
    if accuracy >= 95:
        print("✅ 达到实验要求（≥95%）")
    else:
        print("⚠️  未达到实验要求，可尝试：")
        print("   - 增加训练轮次")
        print("   - 调整学习率")
        print("   - 增加网络深度")

if __name__ == "__main__":
    # 创建报告目录
    if not os.path.exists('./report'):
        os.makedirs('./report')
    
    main()