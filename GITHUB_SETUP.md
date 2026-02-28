# GitHub 仓库设置指南

## 快速开始

### 1. 创建GitHub仓库

访问 https://github.com/new 并创建新仓库：
- **Repository name**: `fy-4b-superResolution`
- **Description**: FY-4B卫星图像超分辨率 - PFT-SR
- **Visibility**: Public 或 Private
- **Initialize**: 不要勾选任何选项

### 2. 推送代码

#### 方式1：使用脚本 (推荐)
```bash
cd /root/codes/sp0301
bash push_to_github.sh billy
```

#### 方式2：手动推送
```bash
cd /root/codes/sp0301

# 添加远程仓库
git remote add origin https://github.com/billy/fy-4b-superResolution.git

# 推送代码
git branch -M main
git push -u origin main
```

### 3. 认证

推送时会要求输入：
- **Username**: `billy`
- **Password**: 使用GitHub Personal Access Token (不是登录密码)

#### 创建Personal Access Token (PAT)
1. 访问 https://github.com/settings/tokens
2. 点击 **Generate new token (classic)**
3. 选择有效期和权限 (至少勾选 `repo`)
4. 生成后复制令牌使用

## 验证推送

推送成功后，访问：
https://github.com/billy/fy-4b-superResolution

## 后续更新

```bash
# 添加更改
git add .
git commit -m "描述你的更改"
git push
```
