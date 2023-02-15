> Created on Wed Feb 15 20:57:27 2023 by Richie Bao

# [Linux常用命令](https://www.digitalocean.com/community/tutorials/linux-commands)<sup>①</sup>

有些程序需要在Linux下书写和运行，因此掌握Linux系统操作的常用命令可以方便的操作该系统。命令通常为英文的缩写形式，因此保留了原文的英文解释，有助于命令的记忆。

|序号|  命令（commands） | 解释  | 英文解释  |
|---|---|---|---|
|1| ls  | 查看目录。列出当前工作目录（文件夹）下的文件和文件夹名称。|  The most frequently used command in Linux to **list directories** |
|2| pwd  | 打印（显示）当前工作目录  | **Print working directory** command in Linux  |
| 3  |  cd | 切换目录。`cd <directory path>`。除直接跟目录名，也可以增加符号执行特殊操作，例如`cd -`返回到前一工作目录；`cd ..`返回到当前目录的父级目录；`cd /`返回到系统工作目录；`cd`不带任何选项则返回到默认工作目录|Linux command to navigate through directories|
| 4  | mkdir  | 创建目录。`mkdir <folder name>` |Command used to create/**make directories** in Linux|
| 5  | mv  | 移动或者重命名（即移动至同一目录下，含新文件名）。`mv <source> <destination>`  |**Move** or rename files in Linux|
| 6  | cp  | 复制文件。`cp <source> <destination>`  |Similar usage as mv but for **copying** files in Linux|
| 7  |  rm | 删除文件。`rm <file name>`；如果要删除目录需要增加参数，如`rm -r <folder/directory name>` 。`rm *`会移除当前文件目录下所有文件 | Delete files or directories|
| 8  |  touch | 创建空文件。`touch <file name>`  |Create blank/empty files|
| 9  |  ln  |  创建链接文件。`ln -s <source path> <link name>` |Create symbolic **links** (shortcuts) to other files|
| 10  |  cat | 打印（查看）文件到命令行 。`cat <file name>` |Display file contents on the terminal|
|  11 | clear  |  清屏。清除终端已显示的内容 |**Clear** the terminal display|
|  12 | echo  | 输出信息。`echo <Text to print on terminal>`  |Print any text that follows the command|
|  13 | less  | 部分显示（分页输出）。当任何命令打印输出内容大于屏幕空间，并且需要滚动显示时，使用。允许使用enter或者space键分解输出并滚动浏览。例如`cat /output/agent_house.csv  \| less`  |Linux command to display paged outputs in the terminal|
|  14 | man  |查看命令的帮助文件 。`man <command>`  |Access **manual** pages for all Linux commands|
| 15  | uname  | 显示系统信息。`uname -a`，返回信息为`Linux LAPTOP-IT5E8TLL 5.10.16.3-microsoft-standard-WSL2 #1 SMP Fri Apr 2 22:23:49 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux` ，参数`a`代表全部 |Linux command to get basic information about the OS|
| 16  | whoami  | 显示当前操作用户  | Get the active username|
|  17 |  tar | 压缩命令行`tar -cvf <archive name> <files seperated by space>`；解压命令行`tar -xvf <archive name>`  |Command to extract and compress files in Linux|
|  18 | grep  | 在文件中查找某个字符串。 `<Any command with output> \| grep "<string to find>"` |Search for a string within an output|
|  19 | head  |  显示文件头内容。`head <file name>` |Return the specified number of lines from the top|
|  20 | tail  | 显示文件尾内容。`tail <file name>`  |Return the specified number of lines from the bottom|
|  21 | diff  | 以逐行的方式,比较文本文件的异同处。`diff <file 1> <file 2>` |Find the **difference** between two files|
| 22  | cmp  |  比较两个文件是否有差异，返回不一致位置的行号。`cmp <file 1> <file 2>` |Allows you to check/**compare** if two files are identical|
| 23  |  comm |  会一列列地比较两个已排序文件的差异,并将其结果显示出来。`comm <file 1> <file2>` |Combines the functionality of diff and cmp|
|  24 | sort  | 针对文本文件的内容,以行为单位来排序 。`sort <filename>` | Linux command to **sort** the content of a file while outputting|
|  25 | export  |  设置或显示环境变量，可新增，修改或删除环境变量,供后续执行的程序使用。`export <variable name>=<value>` |**Export** environment variables in Linux|
|  26 | zip  |  压缩文件。`zip <archive name> <file names separated by space>` |Zip files in Linux|
| 27  | unzip  |  解压文件。`unzip <archive name>` |Unzip files in Linux|
|  28 |  ssh |是 [OpenSSH](https://www.openssh.com)<sup>②</sup> 套件的组成部分，为远程登录服务 SSH 的客户端程序，用于登录远程主机 。`ssh username@hostname`  |**Secure Shell** command in Linux|
|  29 | service  | 用于对系统服务进行管理。比如启动 (start) 、停止 (stop)、重启 (restart)、重新加载配置 (reload)、 查看状态 (status)等 。例如`service ssh status`、`service ssh stop`和`service ssh start`  |Linux command to start and stop **services**|
| 30  | ps  | 查看进程的命令  |Display active **processes**|
|  31 | kill and killall  | 删除执行中的程序或工作。`kill <process ID>`；`killall <process name>` |**Kill** active processes by process ID or name|
| 32  | df  | 检查磁盘空间使用情况。`df -h`中`h`参数可以使得返回信息易于阅读  |Display **disk filesystem** information|
|  33 | mount  |  可以将分区挂接到Linux的一个文件夹下，从而将分区和该目录联系起来，因此只要访问这个文件夹，就相当于访问该分区。例如`mount /content/gdrive` |Mount file systems in Linux|
| 34  |  chmod |设置文件或目录的权限。可用字符形式和数字形式表达，具体的解释此处略。例如`chmod +x loop.sh`等  |Command to change file permissions. **Change mode**|
| 35  | chown  | 用于设置文件所有者和文件关联组的命令。  例如`chown root:root loop.sh`|Command for granting/**change ownership** of files or folders|
| 36  | ifconfig  |  可设置网络设备的状态，或是显示当前的设置。如无此命令，可以通过`sudo apt install net-tools`方法安装 |Display network interfaces and IP addresses|
| 37  |  traceroute |  追踪网络数据包的路由途径。`traceroute <destination address>` |Trace all the network hops to reach the destination|
| 38  | wget  | 从Web下载文件的命令行工具，支持 HTTP、HTTPS及FTP协议下载文件，且wget提供了很多选项，例如下载多个文件、后台下载，使用代理等。` wget <link to file>`，如果增加参数`wget -c <link to file>`，则允许恢复终端的下载 |Direct download files from the internet|
| 39  | ufw  |  `ufw status`查看防火墙状态；`ufw enable`开启防火墙； `ufw disable`关闭防火墙； `ufw reset`重置防火墙，将删除之前定义的所有过滤规则；`ufw allow`允许通过；`ufw deny`禁止通过等。例如`ufw allow 80` |**Firewall** command|
| 40  | iptables  | 用来设置、维护和检查Linux内核的IP包过滤规则。例如`iptables -A INPUT -p tcp -m tcp --dport 80 -j ACCEPT`  | Base firewall for all other firewall utilities to interface with|
|  41 | apt、pacman、yum和rpm  | Linux中的包管理器。不同Linux发行版使用不同德包管理器。 apt：Debian and Debian-based distros；pacman:Arch and Arch-based distros；yum：Red Hat and Red Hat-based distros；rpm：Fedora and CentOS  |Package managers depending on the distro|
|  42 | sudo  | linux系统管理指令，是允许系统管理员让普通用户执行一些或者全部的root命令的一个工具，如halt，reboot，su等 。`sudo <command you want to run>` | Command to escalate privileges in Linux|
|  43 | cal  |  显示日历。例如`cal`为当前月日历，或指定日期`cal Jan 2023` 的公历|View a command-line **calendar**|
|  44 |  alias |  设置命令的别名，以简写命令，提高操作效率。例如`alias lsl="ls -l"`，`alias rmd="rm -r"`等 | Create custom shortcuts for your regularly used commands|
|  45 | dd  | 磁盘维护命令，可以转换和复制来自多种文件系统格式的文件。目前，该命令仅用于为 Linux 创建可启动 USB。例如`dd if = /dev/sdb of = /dev/sda`  |Majorly used for creating bootable USB sticks|
|  46 | whereis  |用于查找文件。 该指令会在特定目录中查找符合条件的文件。这些文件应属于原始代码、二进制文件,或是帮助文件 。例如`whereis sudo` |Locate the binary, source, and manual pages for a command|
| 47  | whatis  |  用于查询其他命令的用途。 例如`whatis sudo`|Find what a command is used for|
| 48  | top  | 监视进程和Linux整体性能  |View active processes live with their system usage|
|  49 |useradd 和usermod   | 建立和修改用户帐号。 例如`useradd JournalDev -d /home/JD`和`usermod JournalDev -a -G sudo, audio, mysql`等 |**Add new user** or change existing users data|
|  50 |  passwd | 用来更改使用者的密码  |Create or update **passwords** for existing users|


---

注释（Notes）：

① Linux常用命令，（<https://www.digitalocean.com/community/tutorials/linux-commands>）。

② OpenSSH，（<https://www.openssh.com>）。