# NPC EITN

鼻咽癌MRI空间治疗反应表型EITN的可复现分析项目。**版本0.2.3**。

本项目提供已配准影像EITN提取、空间与结局关联分析、M1–M4内部开发及冻结外部评价，
配套固定依赖、命令行入口、合成示例、自动测试和论文汇总参考。

## 安装与快速运行

使用Python 3.11，在仓库根目录执行：

```sh
python3.11 -m venv /tmp/npc_eitn_env
. /tmp/npc_eitn_env/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/check_package.py
python -B -m unittest discover -s tests -v
task_dir=$(mktemp -d /tmp/npc_eitn_demo.XXXXXX)
python scripts/run_demo.py --output "$task_dir/results"
```

合成示例使用虚构影像、特征和随访资料，完成EITN提取及四模型流程；采用缩小参数网格、
2外折/2内折、1轮及36个月，展示运行方式。临床参考结果保存在results目录。
输出目录须位于仓库外且尚不存在。

## 功能入口

| 功能 | 脚本 |
| --- | --- |
| 已配准NIfTI提取EITN | `scripts/extract_eitn.py` |
| 整理预测输入 | `scripts/prepare_inputs.py` |
| 内部开发与冻结外部评价 | `scripts/run_analysis.py` |
| 固定预测评价 | `scripts/evaluate_predictions.py` |
| 整理关联输入 | `scripts/prepare_associations.py` |
| Cox、landmark与CIF | `scripts/run_associations.py` |
| 绘制汇总AUC | `scripts/plot_results.py` |

完整命令和输出说明见[工作流](docs/workflows.md)，字段和影像几何要求见
[输入契约](docs/input_contracts.md)，环境与测试见[环境](docs/environment.md)和
[验证](docs/verification.md)。

## 分析设置

- 正式表型与主关联为内部517/外部203例，预测为372/197例；206人外部关联单列为较宽敏感性。
- EITN要求放疗前后T1、T1CE和T2共六幅MRI；放疗后检查在3个月内，随访从放疗结束起算。
- M1为四临床ridge；M2为影像与EITN共同筛选和Elastic Net；M3不含EITN；M4共享M3影像筛选与惩罚，EITN单独正则化。
- 常规影像池为1848项。插补、标准化、筛选和调参限定在内部训练折；临床设置为六轮5外折/4内折、36和60个月。
- 外部使用冻结的预处理与模型。IPCW使用G(t-)和0.05下限，Brier使用全队列分母。
- 研究为探索性分析，研究设计、报告选择及局限以配套论文为准。

## 目录与复现

src提供算法模块，scripts提供运行入口，configs提供特征定义和配置示例，
results保存汇总指标与14张论文表格，provenance保存方法和参考指纹，tests提供测试。
GitHub Actions先执行完整性检查、25项测试和合成流程，再生成正式版本包。

本项目从已配准影像或整理好的表格开始；DICOM转换、配准、人工临床审核和上游常规特征
提取由输入流程提供。真实患者行、影像和临床模型不随软件分发；临床运行结果应存放在
机构批准的位置。具体范围见[说明](docs/scope_and_provenance.md)。

公开仓库：[0xHAng/NPC-EITN](https://github.com/0xHAng/NPC-EITN)。
版本与引用见[CITATION](CITATION.md)，复用条款见[LICENSE_STATUS](LICENSE_STATUS.md)。
