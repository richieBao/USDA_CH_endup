> Created on Sat Nov 18 16:24:40 2023  @author: Richie Bao-caDesign设计(cadesign.cn)

# USDA 工具包

### 2.2.1 USDA 工具包的基本情况

构建城市空间数据分析方法理论集成研究框架时的代码书写有两种方式，一种是，用 Python 及其相关库完全书写，基础试验部分基本全部采用该种方式；另一种是，构建 USDA 工具包，通过调用 USDA 工具完成书写，从而大幅度减少代码量，而更多关注于理论方法本身，专项探索部分大部分采用该种方式。

* USDA 工具包源码托管于 GitHub 仓库[USDA_PyPI](https://github.com/richieBao/USDA_PyPI)，地址为：<https://github.com/richieBao/USDA_PyPI>;
* USDA 工具包存储于 Python Package Index (PyPI)，一个 Python 编程语言的软件存储库，索引名为[usda](https://pypi.org/project/usda)，地址为：<https://pypi.org/project/usda>。通过`pip install usda` 方式安装。如果指定版本则为`pip install usda==0.0.30`；
* USDA 工具包的说明文档为[城市空间数据分析方法-USDA 库手册](https://richiebao.github.io/USDA_PyPI)，地址为：<https://richiebao.github.io/USDA_PyPI>。

截至本书完成时，USDA 工具包的版本为 `usda 0.0.30`。

因为 USDA 工具包主要用于阐述城市空间数据分析集成的理论方法，涉及的内容广泛庞杂，用到的依赖库数量较多且依赖库之间可能存在兼容性等问题，因此 USDA 工具包不强制安装相关依赖库；并且，不同工具往往是独立的，方便直接调用或者直接复制所用源码于具体的计算过程。

### 2.2.2 USDA 工具包的内容（0.0.30 版本）

USDA 工具包在集成城市空间数据分析方法理论过程中逐步形成，且将不断调整和更新。具体的模块及相关简要介绍如下：

* `data_process`：常用数据（集）的预处理和信息读取，及常用数据处理工具；
* `datasets`：实验性数据集，及数据集检索工具；
* `database`：数据库读写工具；
* `utils` 和 `tools`：通用类一般工具集；
* `data_visualization` 和 `data_visual`：数据可视化工具；
* `stats`：统计类工具；
* `geodata_process` 和 `geodata_process_opt`：地理信息数据处理工具（栅格数据和矢量数据等）；
* `models`：模型类；
* `maths`：基本数学计算类；
* `indices`：指数类；
* `pattern_signature`：标记距离与模式；
* `weight`：多准则决策法（权重决策）；
* `meta_heuristics`：元启发式算法；
* `network`：图与复杂网络；
* `net`：深度学习网络；
* `migrated_project`：迁移的代码库，包括 invest（生态系统服务计算工具），pass_panoseg（全景图语义分割），pix2pix（cGAN 通用框架），RL_an_introduction（强化学习），stylegan（生成对抗网络）等；
* `imgs_process`：图像处理；
* `pgm`：概率图（概率论）；
* `rl`：强化学习；
* `mpe_realworld`：自定义多智能体强化学习（MARL）环境示例；
* `demo`：说明类演示图表；
* `manifold`：流形学习（维度空间）；
* `pano_projection_transformation`：全景语义分割图的投影变换。

> USDA 工具包及其说明文档尚不完善，这将需要更长的时间来调整更新，但是当前 0.0.30 版本作为本书的配套工具，可以辅助读者完成书中演示的内容。如果因为依赖库冲突等问题，可以直接对应复制迁移 USDA 源码进行试验。