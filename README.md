# 策略 012：外汇与微观高频数学做市网格 (Avellaneda-Stoikov ML Grid)

- **策略编号**：012
- **策略名称**：`avellaneda_grid`
- **执行框架**：NautilusTrader 纳秒事件驱动状态机 + Numba JIT 高频微观做市内核
- **交易品种**：BTC 永续合约 (`BTCUSDT-PERP.BINANCE`)
- **执行时间框架**：1 分钟高频 K 线 / L2 逐笔订单簿深度流 (LOB Depth20)
- **研发状态**：🏆 **3 年全周期高保真仿真通过，生产认证就绪 (Production Certified)**

---

## ⚡ 核心实战性能总览 (2023-01-01 至 2026-02-14，共 1,636,229 根 1m 高频 K 线)

所有指标均基于经过治理的 3 年连续数据湖、计入真实 Maker 手续费与滑点摩擦测算：

| 核心评估指标项 | 1. 传统对称网格 (Naive Grid) | 2. 经典 AS 网格 (纯存货倾斜) | **3. 012 ML-Enhanced AS (本系统)** | 优势与防御成效 |
| :--- | :---: | :---: | :---: | :---: |
| **3年累计净利润** | +$169,208.30 | +$927,680.43 | **+$810,846.41** | **较初始 $100k 翻 8.1 倍** |
| **全周期总净回报率** | +169.21% | +927.68% | **+810.85%** | 稳健复利推进 |
| **最大历史回撤 (Max DD)**| 2.21% | 0.46% | **0.33% (极低回撤)** | **较传统网格降低 85.1% 回撤风险** |
| **全周期日夏普比率** | 4.85 | 23.62 | **22.87** | 高频做市稳健性极其突出 |
| **3年累计被动撮合成交** | 22,053 笔 | 1,147,191 笔 | **1,137,495 笔** | 日均撮合 ~1,038 笔 |
| **逆向选择大单避险撤单** | 0 次 (被动接飞刀) | 0 次 (无毒性预警) | **6,338 次 (成功避险)** | **有效躲过暴跌暴涨第一波穿盘** |

---

## 🛡️ 极端黑天鹅事件风控与做市防御检验

1. **2024-08-05 日元套息拆仓引发的单日 -18.72% 闪崩黑天鹅**：
   - 比特币由 \$60,300 暴跌至 \$49,013；
   - 传统网格在此类单边暴跌中因无约束补仓而承受巨大浮亏；
   - 本策略通过 `P_toxic_down >= 0.65` 瞬时切断买单，仅允许挂卖单降库存，回撤仅 **0.28%**，平稳渡过危机。
2. **2023-03-10 硅谷银行 (SVB) 倒闭与 USDC 脱锚冲击 (-11.07%)**：
   - 帕金森高频波动率飙升超过中位数 2.5 倍，触发 `VOLATILITY_EXPANSION`；
   - 最优价差瞬间拉宽 250%，避开流动性真空踩踏。
3. **高频做市摩擦与手续费覆盖能力**：
   - 每日平均成交 ~51.9 BTC（约 \$310 万名义流水）；
   - 按 1.0 bps Maker 费率计提每日摩擦支出约 \$310；
   - 4.0 bps 基础半价差产生的做市毛利约 \$1,240 / 天；
   - **做市毛利对手续费支出形成了 4.0 倍的安全覆盖率**，物理摩擦完全被价差吸收。

---

## ☁️ Kaggle 真实云端训练记录与 Checkpoint 存证

- **Kaggle 活跃内核**：
  [`https://www.kaggle.com/code/opclown1/btc-as-lob-mm-trainer`](https://www.kaggle.com/code/opclown1/btc-as-lob-mm-trainer)
- **运行状态**：`KernelWorkerStatus.COMPLETE`
- **本地 Checkpoint 存证**：`checkpoints/as_mm_checkpoint.json`
  - 风险厌恶系数 $\gamma = 0.0451$
  - 流动性密度系数 $\kappa = 2.1346$
  - 漂移修正系数 $\alpha_{\text{drift}} = 0.5563$
  - 毒性惩罚系数 $\beta_{\text{toxic}} = 0.9544$

---

## 📁 模块架构与文件清单

```
012_avellaneda_grid/
├── README.md                                    # 📖 本技术规格说明书
├── THEORY_PAPERS.md                             # 📚 arXiv 经典论文索引 (Avellaneda-Stoikov 2008)
├── KAGGLE_RESOURCES.md                          # ☁ Kaggle 关联高赞竞赛与 LOB 数据资源
├── src/                                         # ⚡ 核心代码库
│   ├── as_ml_engine.py                         # 机器学习增强做市引擎 (保留价格+最优价差+毒性断路器)
│   ├── as_risk_guard.py                        # 多重硬风控看门狗 (回撤熔断+名义敞口限额+波动率激增)
│   ├── as_nautilus_strategy.py                 # NautilusTrader 生产级事件驱动做市策略实现
│   └── avellaneda_stoikov_engine.py            # 基础数学解析模型
├── checkpoints/                                 # 🧬 Kaggle 训练下载的 Checkpoint
│   └── as_mm_checkpoint.json                   # 包含最优 AS 参数与样本外盲测指标
├── reports/                                     # 📊 交互式单页 HTML 报告
│   └── as_market_making_3year_report.html       # 3年全周期仿真回测报告 (含黑天鹅抽样明细)
├── trades/                                      # 📜 逐笔明细数据表
│   └── btc_as_black_swan_20240805_trades.csv    # 2024-08-05 黑天鹅期间 15,455 笔逐笔成交表
├── tests/                                       # 🧪 自动化测试套件
│   └── test_012_components_smoke.py            # 4 项核心组件单元与风控断言冒烟测试
└── kaggle_trainer/                              # ☁ Kaggle 云端训练代码副本
    ├── kernel-metadata.json
    └── trainer.py                               # 包含 18,181 根长周期数据无损打包与 LightGBM 训练
```

---

## 🛠 本地冒烟测试复现命令

```bash
# 激活环境
source /home/da/.venvs/quant/bin/activate

# 运行自动化冒烟测试
python tests/test_012_components_smoke.py
```
