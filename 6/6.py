

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果显示中文有问题可以取消注释）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# Fashion-MNIST类别标签
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', 
               '凉鞋', '衬衫', '运动鞋', '包包', '踝靴']

# ============================================================================
# 辅助函数
# ============================================================================

def visualize_dataset_samples():
    """
    可视化数据集样本 - 类似您图片中的2行5列格式
    """
    print("正在加载数据并可视化样本...")
    
    # 加载原始数据（不进行归一化，方便可视化）
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 收集每个类别的一个样本
    class_samples = {}
    for img, label in train_dataset:
        if label not in class_samples:
            class_samples[label] = (img, label)
        if len(class_samples) == 10:  # 收集到10个类别
            break
    
    # 创建2行5列的图
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()  # 展平为一维数组
    
    for i in range(10):
        if i in class_samples:
            img, label = class_samples[i]
            img_np = img.squeeze().numpy()  # 从(1,28,28)转换为(28,28)
            
            axes[i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'{i}: {class_names[label]}', 
                             color='green', fontsize=10, fontweight='bold')
            axes[i].axis('off')
    
    plt.suptitle('Fashion-MNIST数据集样本展示', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png', dpi=300, bbox_inches='tight')
    print("已保存图片: fashion_mnist_samples.png")
    plt.show()

# ============================================================================
# 任务1：基础模型训练与评估
# ============================================================================

def task1_basic_model():
    """
    任务1：基础模型训练与评估
    """
    print("=" * 60)
    print("任务1：基础模型训练与评估")
    print("=" * 60)
    
    # 1. 数据加载与预处理
    print("\n1. 数据加载与预处理...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]
    ])
    
    # 加载Fashion-MNIST数据集
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset):,} 张图片")
    print(f"验证集: {len(val_dataset):,} 张图片")
    print(f"测试集: {len(test_dataset):,} 张图片")
    
    # 2. 可视化数据集样本
    visualize_dataset_samples()
    
    # 3. 模型构建
    print("\n2. 构建CNN模型...")
    
    class FashionCNN(nn.Module):
        """简单的CNN模型用于Fashion-MNIST分类"""
        def __init__(self):
            super(FashionCNN, self).__init__()
            # 卷积层
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            # 池化层
            self.pool = nn.MaxPool2d(2, 2)
            
            # 归一化和Dropout
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.dropout = nn.Dropout(0.25)
            
            # 全连接层
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            # 卷积块1: 28x28 -> 14x14
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            
            # 卷积块2: 14x14 -> 7x7
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            
            # 卷积块3: 7x7 -> 3x3
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
            # 展平
            x = x.view(-1, 128 * 3 * 3)
            
            # 全连接层
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            
            return x
    
    # 创建模型
    model = FashionCNN().to(device)
    
    # 打印模型信息
    print("\n模型结构:")
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 训练模型
    print("\n3. 训练模型...")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 10
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
    
    # 5. 可视化训练过程
    print("\n4. 可视化训练过程...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, num_epochs + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    ax1.set_xlabel('训练轮次', fontsize=10)
    ax1.set_ylabel('损失值', fontsize=10)
    ax1.set_title('训练和验证损失曲线', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='训练准确率')
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='验证准确率')
    ax2.set_xlabel('训练轮次', fontsize=10)
    ax2.set_ylabel('准确率 (%)', fontsize=10)
    ax2.set_title('训练和验证准确率曲线', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("已保存图片: training_curves.png")
    plt.show()
    
    # 6. 模型评估
    print("\n5. 在测试集上评估模型...")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    print(f"测试集准确率: {test_accuracy:.2f}%")
    
    # 7. 可视化测试结果
    print("\n6. 可视化测试结果...")
    
    # 获取测试集的一些样本
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images, labels = images[:10], labels[:10]
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # 创建2行5列的图显示预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(10):
        # 反归一化图片以便显示
        img = images[i].squeeze().cpu().numpy()
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        confidence = confidences[i].item() * 100  # 转换为百分比
        
        # 显示图片
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # 设置标题 - 使用绿色显示结果
        if true_label == pred_label:
            title_color = 'green'
            title_text = f"✓ 正确\n真: {class_names[true_label]}\n预: {class_names[pred_label]}\n信: {confidence:.1f}%"
        else:
            title_color = 'red'
            title_text = f"✗ 错误\n真: {class_names[true_label]}\n预: {class_names[pred_label]}\n信: {confidence:.1f}%"
        
        axes[i].set_title(title_text, color=title_color, fontsize=9, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('测试集预测结果（2行5列展示）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_predictions_grid.png', dpi=300, bbox_inches='tight')
    print("已保存图片: test_predictions_grid.png")
    plt.show()
    
    # 8. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': test_accuracy,
        'class_names': class_names,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }, 'fashion_mnist_model.pth')
    
    print(f"\n模型已保存为: fashion_mnist_model.pth")
    print(f"最终测试准确率: {test_accuracy:.2f}%")
    
    return model, test_accuracy, all_predictions, all_labels

# ============================================================================
# 任务2：实际应用测试
# ============================================================================

def task2_application_test():
    """
    任务2：实际应用测试
    """
    print("\n" + "=" * 60)
    print("任务2：实际应用测试")
    print("=" * 60)
    
    # 1. 检查模型文件
    if not os.path.exists('fashion_mnist_model.pth'):
        print("错误: 找不到模型文件 'fashion_mnist_model.pth'")
        print("请先运行任务1训练模型")
        return None
    
    print("\n1. 加载预训练模型...")
    
    # 定义模型结构（必须与训练时相同）
    class FashionCNN(nn.Module):
        def __init__(self):
            super(FashionCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.dropout = nn.Dropout(0.25)
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 128 * 3 * 3)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # 创建并加载模型
    model = FashionCNN().to(device)
    checkpoint = torch.load('fashion_mnist_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_accuracy = checkpoint.get('accuracy', 0)
    print(f"模型加载成功！测试准确率: {test_accuracy:.2f}%")
    
    # 2. 测试随机样本
    print("\n2. 测试随机样本...")
    
    # 加载测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=False,
        transform=transform
    )
    
    # 随机选择10个测试样本
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    
    # 收集样本
    test_images = []
    test_labels = []
    for idx in indices:
        img, label = test_dataset[idx]
        test_images.append(img)
        test_labels.append(label)
    
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(test_images.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # 3. 可视化预测结果
    print("\n3. 可视化预测结果（2行5列）...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    correct_count = 0
    
    for i in range(10):
        # 反归一化图片
        img = test_images[i].squeeze().cpu().numpy()
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        
        true_label = test_labels[i].item()
        pred_label = predictions[i].item()
        confidence = confidences[i].item() * 100
        
        # 统计正确预测
        if true_label == pred_label:
            correct_count += 1
            title_color = 'green'
            status = "✓ 正确"
        else:
            title_color = 'red'
            status = "✗ 错误"
        
        # 显示图片
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # 设置标题
        title_text = f"{status}\n"
        title_text += f"真实: {class_names[true_label]}\n"
        title_text += f"预测: {class_names[pred_label]}\n"
        title_text += f"置信度: {confidence:.1f}%"
        
        axes[i].set_title(title_text, color=title_color, fontsize=9, fontweight='bold')
        axes[i].axis('off')
    
    accuracy = 100 * correct_count / 10
    plt.suptitle(f'实际应用测试结果（准确率: {accuracy:.1f}%）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('application_test_results.png', dpi=300, bbox_inches='tight')
    print(f"已保存图片: application_test_results.png")
    print(f"本次测试准确率: {accuracy:.1f}% ({correct_count}/10)")
    plt.show()
    
    # 4. 分析结果
    print("\n4. 结果分析:")
    
    if correct_count < 10:
        print("错误预测分析:")
        for i in range(10):
            if test_labels[i].item() != predictions[i].item():
                true_name = class_names[test_labels[i].item()]
                pred_name = class_names[predictions[i].item()]
                confidence = confidences[i].item() * 100
                print(f"  样本{i+1}: {true_name} → {pred_name} (置信度: {confidence:.1f}%)")
    
    # 5. 显示所有类别的预测概率
    print("\n5. 详细预测概率:")
    
    # 选择一个样本进行详细分析
    sample_idx = 0
    with torch.no_grad():
        sample_output = model(test_images[sample_idx:sample_idx+1].to(device))
        sample_probs = F.softmax(sample_output, dim=1)
    
    # 显示概率分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左侧：显示图片
    sample_img = test_images[sample_idx].squeeze().cpu().numpy()
    sample_img = (sample_img + 1) / 2
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title(f'测试样本\n真实类别: {class_names[test_labels[sample_idx].item()]}', fontsize=12)
    ax1.axis('off')
    
    # 右侧：显示概率分布
    probs = sample_probs.squeeze().cpu().numpy()
    bars = ax2.barh(range(10), probs * 100)
    
    # 为预测正确的类别着色
    true_label = test_labels[sample_idx].item()
    pred_label = predictions[sample_idx].item()
    
    for i, bar in enumerate(bars):
        if i == true_label and i == pred_label:
            bar.set_color('green')  # 正确预测
        elif i == true_label:
            bar.set_color('blue')   # 真实类别
        elif i == pred_label:
            bar.set_color('red')    # 错误预测
        else:
            bar.set_color('gray')   # 其他类别
    
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel('概率 (%)', fontsize=10)
    ax2.set_title('各类别预测概率分布', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("已保存图片: detailed_analysis.png")
    plt.show()
    
    return model

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数 - 控制整个实验流程
    """
    print("=" * 60)
    print("Fashion-MNIST服装图像分类实验")
    print("=" * 60)
    
    # 检查必要的库
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ 所有必要的库都已安装！")
    except ImportError as e:
        print(f"✗ 缺少库: {e}")
        print("请运行: pip install torch torchvision numpy matplotlib")
        return
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {device}")
    
    # 执行任务1
    print("\n" + "=" * 60)
    print("开始执行任务1: 基础模型训练与评估")
    print("=" * 60)
    
    try:
        model, test_accuracy, predictions, labels = task1_basic_model()
        task1_completed = True
    except Exception as e:
        print(f"任务1执行出错: {e}")
        task1_completed = False
    
    # 执行任务2
    if task1_completed:
        print("\n" + "=" * 60)
        print("开始执行任务2: 实际应用测试")
        print("=" * 60)
        
        try:
            model = task2_application_test()
        except Exception as e:
            print(f"任务2执行出错: {e}")
    
    # 实验总结
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    
    print("\n实验输出文件:")
    print("1. fashion_mnist_samples.png    - 数据集样本展示")
    print("2. training_curves.png          - 训练过程曲线")
    print("3. test_predictions_grid.png    - 测试集预测结果")
    print("4. fashion_mnist_model.pth      - 训练好的模型")
    print("5. application_test_results.png - 实际应用测试结果")
    print("6. detailed_analysis.png        - 详细概率分析")
    
    print("\n实验总结:")
    print("1. 成功实现了Fashion-MNIST服装图像分类")
    print("2. 构建了包含3个卷积层的CNN模型")
    print("3. 完成了模型训练、评估和可视化")
    print("4. 进行了实际应用测试")
    print("5. 分析了模型性能和预测结果")
    
    # 与MNIST对比
    print("\n与MNIST手写数字识别对比:")
    print("-" * 50)
    print("| 对比项         | MNIST           | Fashion-MNIST   |")
    print("-" * 50)
    print("| 数据集复杂度  | 简单            | 较复杂          |")
    print("| 类别数量      | 10              | 10              |")
    print("| 类别相似度    | 低              | 高              |")
    print("| 图像特征      | 简单笔画        | 复杂纹理        |")
    print("| 典型准确率    | 99%+            | 90-93%          |")
    print("| 训练难度      | 容易            | 中等            |")
    print("-" * 50)
    
    print("\n常见易混淆类别:")
    print("1. T恤/上衣 ↔ 衬衫 ↔ 外套 (形状相似)")
    print("2. 凉鞋 ↔ 运动鞋 ↔ 踝靴 (鞋类区分)")
    print("3. 套头衫 ↔ 连衣裙 (轮廓相似)")

# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists('./data'):
        os.makedirs('./data', exist_ok=True)
    
    # 运行主程序
    main()