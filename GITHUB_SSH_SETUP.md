# GitHub SSH 密钥设置说明

## 生成的 SSH 密钥对

### 公钥 (Public Key)
请将此公钥添加到你的 GitHub 账户：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJgKEK7D9UDhh7gnjtj9GnizTfVfddn4Ye6lpjLH1ppM hps-seg-pr@github
```

### 私钥 (Private Key)
**注意**: 私钥不应提交到代码仓库。私钥文件已保存在本地，请妥善保管。

## 如何将公钥添加到 GitHub

1. 登录 GitHub 账户
2. 点击右上角头像 → **Settings**
3. 在左侧菜单中找到 **SSH and GPG keys**
4. 点击 **New SSH key** 按钮
5. 填写信息：
   - **Title**: `HPS-Seg PR Key` (或任意名称)
   - **Key**: 粘贴上面的公钥内容
6. 点击 **Add SSH key**

## 添加完成后

添加完公钥后，请告诉我，我将继续执行以下操作：
1. 配置 git 使用 SSH 连接
2. 创建新的分支
3. 提交你的更改
4. 推送到 GitHub
5. 创建 Pull Request

## 注意事项

- 私钥文件权限已设置为 600（仅所有者可读写）
- 请勿将私钥分享给他人或提交到代码仓库
- 如果不再需要此密钥，可以在 GitHub Settings 中删除对应的公钥
