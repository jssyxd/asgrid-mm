# Kaggle 关联资源 (Kaggle Resources) — 策略 012 外汇与微观高频数学网格

> 与同目录 `THEORY_PAPERS.md`（理论论文）并列：本文件汇总 **Kaggle 平台上可复用的数据集 / Notebook / 竞赛**，用于该子项目的深入复现与实验对照。

> 检索时间：2026-10-08 · 工具：Kaggle CLI 2.2.4（`kaggle datasets list -s` / `kaggle kernels list -s` / `kaggle competitions list -s`）。

---

## 一、检索词

`limit order book`、`forex tick data`、`grid trading`

---

## 2、数据集 (Datasets)

| ref | 标题 | 规模(B) | 下载量 | 票数 | 链接 | 为什么相关 |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| `martinsn/high-frequency-crypto-limit-order-book-data` | High Frequency Crypto Limit Order Book Data | 1,041,602,262 | 5,715 | 44 | <https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data> | 高频 LOB 数据，网格/做市的最优控制实验基础。 |
| `sergionefedov/synthetic-limit-order-book-market-microstructure` | Limit Order Book 10 days · 3 market regimes  | 16,204,538 | 33 | 2 | <https://www.kaggle.com/datasets/sergionefedov/synthetic-limit-order-book-market-microstructure> | 可复现的合成 LOB（含多市场状态），适合验证库存与防御状态机。 |
| `waelelghazzawi/limit-order-book-depth-order-flow-imbalance` | Limit Order Book Depth & Order Flow Imbalance | 3,085,996 | 8 | 0 | <https://www.kaggle.com/datasets/waelelghazzawi/limit-order-book-depth-order-flow-imbalance> | LOB 深度与订单流不平衡，用于动态价差与逆向选择度量。 |

下载：`kaggle datasets download -d <ref> -p data/ --unzip`

---

## 3、Notebooks (Kernels)

| ref | 标题 | 作者 | 票数 | 链接 | 为什么相关 |
| :--- | :--- | :--- | ---: | :--- | :--- |
| `anshumandey2103/avellaneda-stoikov-engine` | Avellaneda Stoikov Engine | Anshuman_Dey_9505 | 0 | <https://www.kaggle.com/code/anshumandey2103/avellaneda-stoikov-engine> | A-S 做市引擎实现，与本策略保留价格/最优价差公式直接对应。 |
| `minglunw/grid-trading-model` | Grid trading model | MinglunW | 6 | <https://www.kaggle.com/code/minglunw/grid-trading-model> | 网格交易模型实现，可作传统网格与本策略「受约束最优控制网格」的对照。 |
| `karandeepsinghsodhi/lab-06-avellaneda-stoikov-rl-execution` | Lab 06: Avellaneda-Stoikov & RL Execution | Karandeep Singh Sodhi | 0 | <https://www.kaggle.com/code/karandeepsinghsodhi/lab-06-avellaneda-stoikov-rl-execution> | A-S + 强化学习执行实验，用于参数在线自适应。 |

拉取：`kaggle kernels pull <ref> -p notebooks/`

---

## 4、使用与注意

- 上表 `ref` 可直接用于对应 CLI：`kaggle datasets download -d <ref>`、`kaggle kernels pull <ref>`、`kaggle competitions download -c <ref>`。
- Kaggle 资源为社区上传，**许可与数据口径各异**，使用前请核对各自页面的 License 与字段说明。
- 本文件仅为检索索引，不代下载；如需落盘到本目录 `data/`，请自行执行命令并遵守其许可。
