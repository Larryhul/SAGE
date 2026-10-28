# SAGE

**SAGE 将异构交互编码、联合轨迹扩散与安全社会引导结合起来，同时生成机器人计划和周围实体的未来轨迹。** 推理时通过可微能量引导调整机器人运动，无需重新训练。

![SAGE 方法概览](docs/assets/hero.png)

[English](README.md) · [安装](docs/installation.md) · [复现实验](docs/reproduction.md) · [精选结果](results/README.md) · [论文（arXiv）](https://arxiv.org/abs/2607.16619)

## 开始运行

按[安装说明](docs/installation.md)建立独立环境，在仓库根目录运行：

```bash
python -m sage.experiments.smoke_test --device cpu --stride 10
```

该示例无需外部数据，检查前向计算、反向传播和带引导的轨迹生成。论文结果还需要对应数据、配置和训练权重。

## 仓库内容

- `sage/`：核心模型、引导、数据适配器和实验入口。
- `results/`：44 份精选 CSV，涵盖基准、消融、引导、规模与模型选择。
- 论文与版本记录：[arXiv:2607.16619](https://arxiv.org/abs/2607.16619)。
- `provenance/`：文件来源、校验值和 30 个保留权重的清单。

原始数据、权重、缓存和内部文档未纳入源码仓库。下载情况和所需路径见[资产说明](docs/assets.md)。ETH/UCY 使用行人轨迹中的机器人代理与速度伪角色，其证据范围与真实语义标签实验分别说明。
