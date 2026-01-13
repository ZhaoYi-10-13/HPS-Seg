#!/bin/bash

# HPS-Seg PR 创建脚本
# 在将公钥添加到 GitHub 后运行此脚本

set -e

echo "=== HPS-Seg Pull Request 创建脚本 ==="
echo ""

# 检查私钥是否存在
if [ ! -f "/workspace/hps_seg_github_key" ]; then
    echo "错误: 私钥文件不存在！"
    exit 1
fi

# 配置 SSH
echo "1. 配置 SSH..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 配置 SSH config
cat > ~/.ssh/config << EOF
Host github.com
    HostName github.com
    User git
    IdentityFile /workspace/hps_seg_github_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

chmod 600 ~/.ssh/config

# 测试 SSH 连接
echo "2. 测试 SSH 连接到 GitHub..."
ssh -T git@github.com || true  # 忽略退出码，因为 GitHub 会返回非零退出码

# 进入项目目录
cd /workspace/HPS-Seg

# 配置 git 使用 SSH
echo "3. 配置 git remote 使用 SSH..."
git remote set-url origin git@github.com:ZhaoYi-10-13/HPS-Seg.git

# 创建新分支
BRANCH_NAME="feature/enhanced-hps-seg-$(date +%Y%m%d-%H%M%S)"
echo "4. 创建新分支: $BRANCH_NAME"
git checkout -b "$BRANCH_NAME"

# 添加 .gitignore 如果不存在
if [ ! -f .gitignore ]; then
    echo "5. 创建 .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.log

# Output
output/
*.pth
*.pkl

# Datasets (通常不提交)
datasets/
*.zip
*.tar.gz

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
fi

# 添加更改
echo "6. 添加更改..."
git add -A

# 检查是否有更改
if git diff --cached --quiet; then
    echo "警告: 没有检测到需要提交的更改！"
    exit 1
fi

# 提交更改
echo "7. 提交更改..."
git commit -m "feat: Enhanced HPS-Seg with HPA and AFR innovations

- Add Hyperspherical Prototype Alignment (HPA)
- Add Adaptive Feature Rectification (AFR)
- Update model configurations
- Add training and evaluation scripts
- Update documentation"

# 推送到 GitHub
echo "8. 推送到 GitHub..."
git push -u origin "$BRANCH_NAME"

echo ""
echo "=== 完成 ==="
echo "分支已推送到: $BRANCH_NAME"
echo ""
echo "下一步:"
echo "1. 访问 https://github.com/ZhaoYi-10-13/HPS-Seg"
echo "2. 点击 'Compare & pull request' 按钮"
echo "3. 填写 PR 描述并提交"
echo ""
echo "或者使用 GitHub CLI (如果已安装):"
echo "gh pr create --title 'feat: Enhanced HPS-Seg with HPA and AFR' --body 'This PR adds HPA and AFR innovations to HPS-Seg'"
