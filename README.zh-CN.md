# SAGE

**SAGE 联合生成机器人计划与周围实体的未来轨迹，并通过安全、社会规范及任务目标引导调整机器人运动。**

[English](README.md) · [论文](https://arxiv.org/abs/2607.16619) · [实验说明](docs/reproduction.md) · [模型](docs/models.md) · [结果](results/README.md)

## 快速开始

推荐 Python 3.12。按[安装说明](docs/installation.md)建立环境，在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python -m sage.demo --output outputs/demo
```

示例展示交叉场景中的轨迹优化，周围实体的预测保持固定。输出动画、最终轨迹图和能量变化记录。

![SAGE CPU 演示](docs/assets/demo.gif)

示例的输入与输出说明见[演示指南](docs/demo.md)。

## 模型与结果

提供 ETH/UCY 五个场景与 SDD 的六份原始目标条件研究检查点。

```bash
python tools/download_models.py
```

下载完成后自动校验文件。[模型说明](docs/models.md)给出模型身份与使用方式。[44 份精选结果](results/README.md)按实验协议组织；数据准备、训练和评估命令见[实验指南](docs/reproduction.md)。

## 开发检查

```bash
python tools/check_repository.py
python -m pytest -q
```

Linux 与 Windows 均运行 CPU 检查。依赖归档实验资产或 CUDA 的测试会明确报告跳过原因。

## 引用

使用 SAGE 开展研究时，请引用[论文](https://arxiv.org/abs/2607.16619)。[BibTeX](docs/citation.bib) · [CITATION.cff](CITATION.cff)。
