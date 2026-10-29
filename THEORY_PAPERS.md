# 理论论文与参考 (Theory Papers) — 策略 012 外汇与微观高频数学网格

> 本目录 `papers/` 收录与该子项目直接相关的公开论文全文（arXiv 开放获取 PDF），并附 SSRN / DOI 链接。每篇论文均给出「为什么相关」的说明。

---

## 一、总体说明

本策略把传统网格/马丁重构为受约束最优控制：保留价格中枢 + 动态非对称价差 + 库存/保证金硬约束 + 防御状态机。

---

## 二、arXiv 论文（全文 PDF 已随目录提供）

| # | 论文 | arXiv | 本地文件 | 为什么相关 |
| :--: | :--- | :--- | :--- | :--- |
| 1 | **Dealing with the Inventory Risk: A Solution to the Market Making Problem** | [1105.3115](https://arxiv.org/abs/1105.3115) | [`1105.3115.pdf`](./papers/1105.3115.pdf) | Avellaneda-Stoikov 库存风险做市解，本策略保留价格 r = s − qγσ²(T−t) 与最优价差的直接来源。 |
| 2 | **Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets** | [1105.1694](https://arxiv.org/abs/1105.1694) | [`1105.1694.pdf`](./papers/1105.1694.pdf) | 平方根冲击律，用于网格挂单的冲击与容量约束。 |
| 3 | **An Optimal Execution Problem with Market Impact** | [0907.3282](https://arxiv.org/abs/0907.3282) | [`0907.3282.pdf`](./papers/0907.3282.pdf) | 带市场冲击的最优执行，防守模式下减仓路径的理论依据。 |
| 4 | **Optimal Execution with Nonlinear Transient Market Impact** | [1412.4839](https://arxiv.org/abs/1412.4839) | [`1412.4839.pdf`](./papers/1412.4839.pdf) | 非线性瞬态冲击下的最优执行，刻画冲击记忆效应对网格间距的影响。 |

---

## 四、获取与引用约定

- arXiv 论文为开放获取，`papers/*.pdf` 为对应 arXiv 版本的全文副本；引用请以 arXiv 编号为准。
- SSRN 论文正文受站点防护限制未能随目录提供，请通过上表链接访问。
- 本目录仅用于技术研究与复现参考，版权归原作者所有。
