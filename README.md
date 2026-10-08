# H&M 个性化推荐与冷启动商品建模

一个基于购买历史的两阶段推荐项目：用多路召回扩大候选覆盖，用时间安全的行为特征和协同表示学习排序，再用内容表示补足缺少交互的商品。工程重点是**把召回、排序和最终推荐动作分别验证，让模型收益能被定位、解释和复核**。

技术栈：Python · DuckDB · pandas · LightGBM · PyTorch · Item2Vec · FashionCLIP。

## 项目完成了什么

- **推荐主干**：六路行为与属性召回生成前100候选，Item2Vec补充最多200件独有商品；两个基础排序器分别使用84维特征、84维加2维BPR匹配特征，通过倒数名次融合形成推荐顺序。
- **保守尾部重排**：在前50候选上使用99维购买二分类与组内排序模型，保护原前7名，最多替换原第8–12名中的两件商品；没有近期购买历史的用户保留候选顺序。
- **冷商品内容建模**：以购买序列学到的Item2Vec关系为监督，训练图像与商品属性的Student编码器，将购买关系迁移到零交互商品。
- **可核查评测**：时间切分、多季节开发窗口、候选预算对齐、分组诊断、真实替换动作重放，以及树模型特征贡献分析。

Item2Vec是行业常见的商品嵌入方法；BPR是基于隐式反馈的成对排序目标，这里用于生成用户与商品的匹配特征。Student是本项目内容编码器的称呼，以协同商品关系作为训练监督。LambdaRank是行业通用的组内学习排序方法。完整结构见[架构与代码路线](docs/architecture.md)。

## 已测量结果与价值

以下为既有实验的汇总结果；整理展示仓库时没有重新训练或重跑大规模评测。开发窗口为2020-01-22、03-18、06-24、08-19起的各一周，曾用于模型开发。

| 验证对象 | 对照 → 改进 | 说明与实际价值 |
|---|---|---|
| 零交互商品候选召回率 | 原始FashionCLIP **2.79% → 多模态Student 3.93%**，相对提高约41.05% | 同样最多50件内容候选，与相同基础候选取并集。购买关系监督使内容表示更适合召回，补足纯行为召回的覆盖缺口。 |
| 低频商品候选内Recall@5 | **43/104 → 90/104**，即41.35% → 86.54% | 四窗合计104个未来正例用户—商品对；排名来自完整50件冷商品候选。生命周期信号帮助少量历史交互商品进入候选前五。 |
| 零交互商品候选内Recall@5 | **22/160 → 16/160**，即13.75% → 10.00% | 同一时效排序器对零交互商品收益不成立，支持把零交互与低频商品分别建模、分别检查。 |

**统计口径**：零交互指截止日前全历史交易事件数为0；低频指1–5次，均为本项目分组。第一行先对每个有该组未来购买的用户计算候选覆盖比例，再按用户平均，最后对四窗等权平均。后两行按正例用户—商品对合计计算，分母只包含已经进入冷候选池的未来正例。因此两类Recall不能直接比较，也不能相乘推算最终推荐收益。

这些结果证明了内容召回和低频候选排序各自的价值。最终前12推荐还要通过跨来源准入与替换收益检查：候选内提升并不自动带来整体MAP提升。现有冷商品融合没有通过最终推荐收益门槛，项目保留经过验证的行为推荐主干。公开汇总见[结果数据](docs/results.json)及[评测说明](docs/evaluation.md)。

## 快速运行：无需下载数据

建议Python 3.11。以下命令在仓库根目录执行，适用于PowerShell；已有可用环境也可以直接安装项目。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
hm-recsys validate --transactions examples/demo_transactions.csv --cutoff 2020-09-01 --sample-rate 1 --history-weeks 8 --candidate-k 20 --output-dir artifacts/demo
```

示例中的`u1`等用户和商品编号是合成数据。命令运行“复购 + 近期热度”基线，输出候选Recall、用户命中率、Oracle MAP与MAP，并将结果写入`artifacts/demo`。**它验证评测管线可运行，不复现上表的大规模实验结果。**

运行测试：

```powershell
python -m unittest discover -s tests -v
```

整理时在既有项目环境中验证：773项测试，719项通过，54项因历史资产或既有可选检查跳过；合成样例成功运行。记录见[验证结果](docs/validation.json)。未验证全新机器上的依赖安装。

首次安装会包含PyTorch等研究依赖；GPU计算需要与本机驱动匹配的PyTorch版本。示例基线可以在CPU运行。

## 使用完整数据

数据来自[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)。请在数据平台接受适用条款，自行取得数据；本仓库不分发数据、图片、客户记录或预测结果。

将`transactions_train.csv`、`articles.csv`和`customers.csv`放入`data/raw/`。需要下载工具时，复制`.env.example`为本地`.env`并填写自己的凭据，然后执行：

```powershell
hm-recsys data status
hm-recsys data download --tabular
hm-recsys validate --transactions data/raw/transactions_train.csv --cutoff 2020-09-16 --sample-rate 0.01 --history-weeks 12 --candidate-k 100 --output-dir artifacts/baseline-sample
```

最后一条是1%用户样本基线验证，不能代表完整人口表现。高级排序与内容研究需要相应的预处理表、嵌入、训练合同和模型；具体入口及可复现范围见[运行说明](docs/reproduction.md)。

## 仓库导航

```text
src/hm_recsys/    时间切分、召回、排序、内容建模与历史研究模块
tests/           合成数据上的算法、边界和管线测试
examples/        微型合成交易样例
docs/            架构、指标口径、复现范围、脱敏记录与汇总结果
data/            数据占位目录，真实数据不进入Git
artifacts/       运行产物占位目录，模型与预测不进入Git
```

本仓库以独立Git历史整理。研究模块中的冻结分支、历史提交和证据检查予以保留；依赖原实验合同的入口属于历史实现，不能当作一键训练入口。它们的用途是展示方法、实现和工程约束。

## 数据与展示边界

所有行为统计只使用预测截止日前交易；同日共同购买是用户—日期篮子，不能当作真实会话或订单。购买是隐式正反馈，未购买不代表明确不喜欢。已知完整商品目录用于离线评测，不模拟精确上架时间；没有曝光、库存日志或线上A/B结果。

展示仓库排除了原Git历史、私人环境、服务器与SSH配置、原始报告、逐用户结果、模型、提交文件和个人简历。脱敏范围与路径修改列表见[迁移记录](docs/export-manifest.json)。
