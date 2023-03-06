# GIT 配置

## Homebrew安装

官网中提供了几种，包括先下载Homebrew，再从Homebrew下载Git，从Xcode下载Git等。之前mac下载过Xcode，但太大了，而且不怎么常用这个IDE，所以新电脑我就不想下Xcode，选择用Homebrew下载

1. 安装Homebrew
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

2. 按照安装好Homebrew后cmd给的提示Next Steps输入那两行代码[![在这里插入图片描述](https://img-blog.csdnimg.cn/b0948632fa6b4ce5a63701d1a9af762d.png)](https://ispacesoft.com/wp-content/plugins/wp-fastest-cache-premium/pro/images/blank.gif)

3. `brew install git` (结束了之后，但没结束哦，用`git --version` 检查显示`git version 2.32.1 (Apple Git-133) `那说明还是mac自带的git）下一步我们要修改环境变量。

4. 输入 `vim ~/.bash_profile` 打开bash_profile文件修改环境变量，输入`i`进入编辑模式

5. 添加环境变量：

   ```java
   export PATH="/opt/homebrew/bin:${PATH}"
   ```

   

6. esc, wq保存, 输入`source ~/.bash_profile`

7. 用`which git`和`git --version`检查一下

8. 设置参数

   注意： git config -–global参数，表示这台机器上的所有的git仓库都会使用这个配置，当然也可对某个仓库指定不同的用户名和邮箱，更多参数我们也可以通过git config提示查看，还可以使用git config --list或git config -l来查看已经配置的信息。

```shell
siyuanluo@siyuandeMacBook-Pro ~ % vim ~/.bash_profile
siyuanluo@siyuandeMacBook-Pro ~ % source ~/.bash_profile
siyuanluo@siyuandeMacBook-Pro ~ % which git
/opt/homebrew/bin/git
siyuanluo@siyuandeMacBook-Pro ~ % git --version
git version 2.39.2
siyuanluo@siyuandeMacBook-Pro ~ % git config --global user.name "siyuanluo"
siyuanluo@siyuandeMacBook-Pro ~ % git config --global user.email "1257480002@qq.com"
siyuanluo@siyuandeMacBook-Pro ~ % git config -l
credential.helper=osxkeychain
user.name=siyuanluo
user.email=1257480002@qq.com
siyuanluo@siyuandeMacBook-Pro ~ % 

```

