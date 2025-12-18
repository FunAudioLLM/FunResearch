import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class LipEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super(LipEncoder, self).__init__()
        
        # 编码器部分 - 使用卷积层逐步提取特征
        self.encoder = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),  # 下采样2倍
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 再下采样2倍
            
            # 第二层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 下采样2倍
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 下采样2倍
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层卷积块
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 下采样2倍
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 全连接层用于特征压缩
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, feature_dim)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 编码器前向传播
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # 全连接层
        output = self.fc(features)
        
        # L2归一化，使得特征向量在单位球面上
        output = F.normalize(output, p=2, dim=1)
        
        return output

class LipEncoderWrapper:
    def __init__(self, model_path=None, device='auto'):
        """
        唇部编码器封装类
        """
        self.device = self._get_device(device)
        self.model = LipEncoder().to(self.device)
        
        if model_path is not None:
            self.load_model(model_path)
        
        self.model.eval()
    
    def _get_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def load_model(self, model_path):
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f"模型已从 {model_path} 加载")
    
    def preprocess_image(self, image_bgr):
        """
        预处理输入图像
        
        Args:
            image_bgr: cv2读取的BGR图像
            
        Returns:
            预处理后的tensor
        """
        # BGR转RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸到合适大小 (假设输入为128x128)
        target_size = (128, 128)
        if image_rgb.shape[:2] != target_size:
            image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
        else:
            image_resized = image_rgb
        
        # 归一化到 [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 标准化 (使用ImageNet的均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # 转换维度: HWC -> CHW
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def encode(self, image_bgr):
        """
        提取唇部特征
        
        Args:
            image_bgr: cv2读取的BGR图像
            
        Returns:
            512维唇部特征向量 (numpy数组)
        """
        with torch.no_grad():
            # 预处理图像
            input_tensor = self.preprocess_image(image_bgr)
            
            # 提取特征
            features = self.model(input_tensor)
            
            # 转换为numpy数组
            features_np = features.cpu().numpy().flatten()
            
            return features_np
    
    def encode_batch(self, images_bgr):
        """
        批量提取唇部特征
        
        Args:
            images_bgr: cv2读取的BGR图像列表
            
        Returns:
            512维唇部特征向量数组
        """
        batch_tensors = []
        
        for img in images_bgr:
            input_tensor = self.preprocess_image(img)
            batch_tensors.append(input_tensor)
        
        batch = torch.cat(batch_tensors, dim=0)
        
        with torch.no_grad():
            features = self.model(batch)
            features_np = features.cpu().numpy()
            
            return features_np

if __name__ == "__main__":
    # 初始化编码器
    lip_encoder = LipEncoderWrapper()
    
    # 读取图像
    image_path = "lip_image.jpg"  # 替换为您的图像路径
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is not None:
        # 提取特征
        features = lip_encoder.encode(image_bgr)
        
        print(f"特征维度: {features.shape}")
        print(f"特征范数: {np.linalg.norm(features):.4f}")  # 应该是1.0 (L2归一化)
        print(f"特征示例 (前10维): {features[:10]}")
    else:
        print("无法读取图像")