#!/bin/bash
# 推送到GitHub的辅助脚本
# 使用方法: bash push_to_github.sh YOUR_GITHUB_USERNAME

set -e

USERNAME=${1:-billy}
REPO_NAME="fy-4b-superResolution"

echo "=========================================="
echo "FY-4B Super Resolution - GitHub推送脚本"
echo "=========================================="
echo ""
echo "GitHub用户名: $USERNAME"
echo "仓库名称: $REPO_NAME"
echo ""

# 检查git状态
echo "检查Git状态..."
git status

echo ""
echo "添加远程仓库..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"

echo ""
echo "推送代码到GitHub..."
echo "提示: 需要输入GitHub个人访问令牌(PAT)作为密码"
echo ""

git branch -M main
git push -u origin main

echo ""
echo "=========================================="
echo "推送完成!"
echo "访问: https://github.com/$USERNAME/$REPO_NAME"
echo "=========================================="
