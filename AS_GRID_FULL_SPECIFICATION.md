# Strategy 012 / 003 Avellaneda-Stoikov ML Grid 全生命周期研发白皮书与跨平台复现实操手册
(Avellaneda-Stoikov Microstructure High-Frequency Grid Engine Specification & Replication Guide)

> **版本**：v2.4.0 (Production Verified & Cross-Exchange Ready)  
> **面向对象**：量化研究员、高频做市工程人员、系统部署运维架构师  
> **适用场景**：高频做市刷量（asgrid-mm）、价差套利盈利（asgrid-arb）、多交易所移植、新资产标的适配。

---

## 目录
1. [策略第一性原理与微观数学推导](#一策略第一性原理与微观数学推导)
2. [数据源全景指南与数据治理标准](#二数据源全景指南与数据治理标准)
3. [多层机器学习模型架构与特征工程](#三多层机器学习模型架构与特征工程)
4. [模型离线训练与增量演化流水线](#四模型离线训练与增量演化流水线)
5. [VPS 纳秒/亚毫秒级在线推理与极限优化](#五vps-纳秒亚毫秒级在线推理与极限优化)
6. [订单生命周期管理与做市防坑铁律](#六订单生命周期管理与做市防坑铁律)
7. [资金自适应动态风险缩放模型](#七资金自适应动态风险缩放模型)
8. [高保真回测环境搭建与订单簿撮合仿真](#八高保真回测环境搭建与订单簿撮合仿真)
9. [实盘复现全流程与跨交易所迁移指南](#九实盘复现全流程与跨交易所迁移指南)
10. [常见陷阱排查手册与故障自愈清单](#十常见陷阱排查手册与故障自愈清单)

---

## 一、策略第一性原理与微观数学推导

经典网格策略假设资产价格在固定区间内围绕均值波动，采取机械的等间距挂单。
在遭遇单边趋势性破位或瞬时流动性踩踏时，传统网格必然持续逆势加仓，直至保证金耗尽触发清算。
Avellaneda-Stoikov (2008) 框架引入最优随机控制理论，将做市商行为建模为最大化终端期望效用并最小化存货方差的动态优化过程。

### 1.1 资产中枢价值动态漂移与微观修正
设当前盘口中间价为 $S_t = \frac{P^a_1 + P^b_1}{2}$。
在微观订单流不平衡下，真实公允微观价格偏离中间价：
$$P^{\text{micro}}_t = \frac{P^b_1 \cdot V^a_1 + P^a_1 \cdot V^b_1}{V^b_1 + V^a_1}$$

结合机器学习预测的价格前瞻漂移 $\mu_{\text{drift}}$，做市商内部保留价格（Reservation Price）修正为：
$$r(S_t, q_t, t) = S_t - q_t \cdot \gamma \cdot \sigma^2 \cdot (T - t) \cdot \lambda_{\text{inv}} + \alpha_{\text{drift}} \cdot \mu_{\text{drift}} \cdot (S_t \cdot \kappa_{\text{drift}})$$

参数含义：
- $S_t$：当前标的物理中间价。
- $q_t$：做市商净存货敞口（正数为多头，负数为空头）。
- $\gamma$：绝对风险厌恶系数（Risk Aversion Coefficient）。
- $\sigma$：标的瞬时微观波动率（由加曼-克拉斯与帕金森极值方差加权得出）。
- $(T - t)$：做市周期的剩余标准化时间视野（实盘中离散化为常数阻尼项）。
- $\lambda_{\text{inv}}$：存货惩罚权重乘子。
- $\alpha_{\text{drift}}$：机器学习漂移增益调节系数。
- $\mu_{\text{drift}}$：LightGBM 模型输出的未来短周期收益率漂移预测值。

当做市商持有多头库存 ($q_t > 0$) 时，保留价格 $r$ 自动向下倾斜。
这使得买单挂单价进一步远离盘口，卖单挂单价更加贴近盘口，从而在概率上抑制被动补仓，促成存货快速平仓出清。

### 1.2 最优非对称做市价差（Optimal Spread）
做市商在买卖双侧挂出不对称报价：
$$P^b_t = r(S_t, q_t, t) - \delta^b_t$$
$$P^a_t = r(S_t, q_t, t) + \delta^a_t$$

基础半价差（Base Half-Spread）由流动性强度参数 $\kappa$ 与风险厌恶系数 $\gamma$ 决定：
$$\delta_{\text{base}} = \max\left(S_t \cdot \text{Spread}_{\min}, \; \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right) \cdot \psi_{\text{scale}}\right)$$

为了抵御知情交易者（Informed Traders）的大单逆向选择冲击，引入机器学习毒性膨胀系数：
$$\delta^b_t = \delta_{\text{base}} \cdot \left(1.0 + \beta_{\text{toxic}} \cdot P_{\text{toxic\_down}}\right)$$
$$\delta^a_t = \delta_{\text{base}} \cdot \left(1.0 + \beta_{\text{toxic}} \cdot P_{\text{toxic\_up}}\right)$$

当模型识别到下行暴跌概率 $P_{\text{toxic\_down}}$ 超过警戒阈值时，买单价差急剧外扩甚至直接切断挂单；
当模型识别到上行挤压概率 $P_{\text{toxic\_up}}$ 增高时，卖单价差同步膨胀。

---

## 二、数据源全景指南与数据治理标准

做市商策略对回测和离线训练数据的精度要求远高于普通趋势策略。
禁止使用简单合成数据或纯低频（如 1d/4h）K 线进行高频做市仿真。

### 2.1 高质量盘口与订单簿数据检索定位

| 数据平台 | 推荐检索关键词 / 资产目录 | 关键数据集标识 (Dataset Slug / ID) | 数据规格与层级 |
| :--- | :--- | :--- | :--- |
| **Kaggle** | `high frequency crypto limit order book`, `bitcoin perpetual lob` | `martinsn/high-frequency-crypto-limit-order-book-data`<br>`siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data` | 逐笔快照 (100ms LOB Snapshots), Depth 20, 包含真实盘口买卖档量价。 |
| **Hugging Face** | `crypto orderbook`, `tardis binance l2` | `datasets/m-bain/crypto-lob-high-frequency`<br>`datasets/open-crypto-metrics/binance-l2-deltas` | Parquet 格式存储，纳秒级时间戳，带 Bid/Ask 增量 Diff 与主动成交流 (AggTrades)。 |
| **GitHub** | `crypto-orderbook-dataset`, `nautilus-crypto-datacatalog` | `jssyxd/nautilus-crypto-datacatalog`<br>`jssyxd/multi-asset-ohlcv` | 符合 NautilusTrader 官方 ParquetDataCatalog 规范的清洗后本地湖。 |

### 2.2 数据格式与治理准入红线（Data Governance Rules）
1. **时间戳连续性与无死锁要求**：
   - 必须使用服务端撮合时间戳（`exchange_timestamp_utc`），杜绝本地接收时间偏差；
   - 严禁存在时间倒流（$T_{k+1} < T_k$）；遇到重复时间戳必须通过序列号（Sequence ID）做确定性排序。
2. **微观深度有效性**：
   - 每一个盘口快照必须严格保证 $P^a_1 > P^b_1$（严禁出现交叉盘或倒挂盘口，异常跨度数据必须打标剔除）；
   - 剔除挂单量为 0 或负值的坏数据。
3. **K 线极值逻辑自洽**：
   - 1m/15m OHLCV 数据中必须满足 $\text{High} \ge \max(\text{Open}, \text{Close})$ 且 $\text{Low} \le \min(\text{Open}, \text{Close})$；
   - 缺失周期必须采用上根 K 线收盘价向前填充（Forward Fill），成交量置 0。

---

## 三、多层机器学习模型架构与特征工程

为了在 VPS 资源极度受限（如 1 核 1GB 内存）的环境下稳定达成 600ms 以内的全流程推断，系统采用两阶段异构架构：
**向量化微观特征引擎 + 3 组紧凑型决策树模型（LightGBM Boosters）**。

```
[原始输入] ──┬──> 1m OHLCV 时序矩阵 (30~60 根) ────> Parkinson/GK 波动率 + ATR + 均线偏离度
            │                                                                         │
            └──> L2 深度 20 档订单簿快照 ────────> 20档 OBI + Cont-Kukanov-Stoikov OFI ──┼──> 特征向量 X (7 维)
                                                                                      │
                                                                                      ▼
                                                                     ┌─────────────────────────────────┐
                                                                     │ LightGBM 紧凑推理矩阵             │
                                                                     │ 1. bst_drift (50 树, 预测价格漂移)│
                                                                     │ 2. bst_tox_down (40 树, 下行冲击)│
                                                                     │ 3. bst_tox_up (40 树, 上行挤压)  │
                                                                     └────────────────┬────────────────┘
                                                                                      │
                                                                                      ▼
[做市中枢定价] <── 结合实时仓位 q 实施非对称挂单报价 <── [输出] mu_drift, p_down, p_up ────────┘
```

### 3.1 7 维高频核心特征矩阵构建

```python
import math
import numpy as np
import pandas as pd

def extract_microstructure_features(candles_1m: np.ndarray, book_l2: dict) -> np.ndarray:
    """
    candles_1m 格式: [N, 6] -> [timestamp, open, high, low, close, volume]
    book_l2 格式: {'bids': [[px, sz], ...], 'asks': [[px, sz], ...]}
    """
    o = candles_1m[:, 1].astype(float)
    h = candles_1m[:, 2].astype(float)
    l = candles_1m[:, 3].astype(float)
    c = candles_1m[:, 4].astype(float)
    v = candles_1m[:, 5].astype(float)
    
    n = len(c)
    # 1. 帕金森极值波动率 (Parkinson Volatility)
    log_hl = np.log(np.maximum(h, 1e-6) / np.maximum(l, 1e-6))
    p_var = (log_hl ** 2) / (4.0 * math.log(2.0))
    p_vol = np.sqrt(pd.Series(p_var).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1] * (365 * 1440))
    p_vol = max(0.08, float(p_vol))
    
    # 2. 加曼-克拉斯极值与收开波动率 (Garman-Klass Volatility)
    log_co = np.log(np.maximum(c, 1e-6) / np.maximum(o, 1e-6))
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
    gk_vol = np.sqrt(pd.Series(gk_var).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1] * (365 * 1440))
    gk_vol = max(0.08, float(gk_vol))
    
    # 3. 真实波幅均值 ATR(14)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    atr14 = max(0.5, float(pd.Series(tr).rolling(min(n, 14), min_periods=3).mean().bfill().iloc[-1]))
    
    # 4. 均线归一化价格偏离度 (f_dev)
    ma20 = float(pd.Series(c).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1])
    f_dev = float((c[-1] - ma20) / atr14)
    
    # 5. 1 周期微观收益率 (s_r1)
    s_r1 = float((c[-1] - c[-2]) / (c[-2] + 1e-9)) if n >= 2 else 0.0
    
    # 6. 成交量加速度比率 (vol_accel)
    vol_ma = float(pd.Series(v).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1])
    vol_accel = float(np.clip(v[-1] / (vol_ma + 1e-9), 0.01, 50.0))
    
    # 7. 订单流不平衡量 (OFI) 与 20 档订单簿偏度 (OBI)
    bids = book_l2.get('bids', [])
    asks = book_l2.get('asks', [])
    depth_n = min(20, len(bids), len(asks))
    tot_b_vol = sum(float(bids[i][1]) for i in range(depth_n))
    tot_a_vol = sum(float(asks[i][1]) for i in range(depth_n))
    obi = (tot_b_vol - tot_a_vol) / (tot_b_vol + tot_a_vol + 1e-6)
    bar_ofi = float(np.clip(obi * 2.0, -3.0, 3.0))
    
    # 紧凑 7 维特征向量输出
    return np.array([[p_vol, gk_vol, atr14, bar_ofi, f_dev, s_r1, vol_accel]], dtype=np.float64)
```

---

## 四、模型离线训练与增量演化流水线

### 4.1 训练标签体系构建（多视野目标定义）
针对未来 $H=5$ 根 K 线的时间窗口生成训练目标，杜绝时间泄露：
- **前瞻标准化收益率**：
  $$y_{\text{drift}} = \frac{C_{t+H} - C_t}{\max(\text{ATR}_{14}(t), \; 0.05)}$$
- **下行暴跌毒性冲击标签（Adverse Selection Downward Shock）**：
  $$y_{\text{toxic\_down}} = \mathbb{I}\left(\frac{\min_{1 \le k \le H} L_{t+k} - C_t}{\max(\text{ATR}_{14}(t), \; 0.05)} \le -1.8\right)$$
- **上行剧烈挤压冲击标签（Adverse Selection Upward Shock）**：
  $$y_{\text{toxic\_up}} = \mathbb{I}\left(\frac{\max_{1 \le k \le H} H_{t+k} - C_t}{\max(\text{ATR}_{14}(t), \; 0.05)} \ge 1.8\right)$$

### 4.2 LightGBM 关键参数配置
```python
import lightgbm as lgb

# 漂移回归模型参数（限制叶子节点数，严格防止过拟合并压缩推断延迟）
params_drift = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.08,
    'num_leaves': 15,
    'max_depth': 4,
    'min_data_in_leaf': 50,
    'verbosity': -1,
    'n_jobs': 2
}

# 毒性二分类模型参数
params_toxic = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.08,
    'num_leaves': 12,
    'max_depth': 4,
    'min_data_in_leaf': 30,
    'verbosity': -1,
    'n_jobs': 2
}
```

### 4.3 AlphaGPT 与大模型变异优化引擎
引入大语言模型（Gemini Flash/Pro）与演化算法结合，迭代寻找最佳做市参数：
1. 提取当前回测/实盘的卡玛比率、最大回撤、胜率与夏普比率。
2. 构造特定 Prompt 请求大模型在保持风控边界前提下提出变异候选方案。
3. 通过 Numba 极速向量化回测引擎评测突变体的适应度（Fitness）：
   $$\text{Fitness} = \text{TotalReturn} \times 10.0 + \text{Calmar} \times 5.0 - 30.0 \times (\text{MaxDrawdown})^2$$
4. 超过基准的优异基因输出更新至 `as_mm_checkpoint.json`。

---

## 五、VPS 纳秒/亚毫秒级在线推理与极限优化

云服务器通常 CPU 主频受限且内存吃紧（960MB）。
必须在每个环节进行极致的延迟剥离，将单轮循环由最初的数秒压缩至稳定 400ms~600ms。

### 5.1 强制 IPv4 与连接池常驻
- **问题**：Debian/Ubuntu 双栈网络下，`urllib`/`requests` 会首先尝试 IPv6，引发 DNS 寻址往返延迟及 OKX IP 白名单报错（`50110`）。
- **解决方案**：在底层直接劫持 `socket.create_connection` 与 `socket.getaddrinfo`，强制绑定 `socket.AF_INET`。
- **连接复用**：使用带有连接池的 `requests.Session()`，保持与交易所网关的 TCP Keep-Alive 长连接，省去每次下单三次握手与 TLS 协商时间（单次节省 150ms 以上）。

### 5.2 批量撤单与非阻塞循环
```python
# 每次做市循环刷新报价时，禁止逐笔发送 DELETE 请求
# 统一汇聚为 batch 结构单次往返执行
pending_orders = client.get_pending_orders(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
if pending_orders:
    cancel_list = [{"instId": "BTC-USDT-SWAP", "ordId": p.get("ordId")} 
                   for p in pending_orders if p.get("ordId")]
    client.cancel_batch_orders(cancel_list) # 单次往返处理最多 20 个订单
```

### 5.3 内存治理与 OOM 防御
- 注入环境变量 `PYTHONMALLOC=malloc`，强制 Python 解释器在特征矩阵计算完毕后立即将堆内存释放归还操作系统内核。
- 在 Docker 编排中设定 `limits.memory: 300M`，彻底避免突发内存暴涨击垮宿主机其他进程。

---

## 六、订单生命周期管理与做市防坑铁律

在此前版本迭代与第三方审计（Grok 复核）中，我们挖掘出了几个极具隐蔽性的工程致命缺陷，本节确立为不可逾越的纪律红线。

### 6.1 杜绝“挂单即成交”的假流水 Bug
- **历史严重缺陷**：代码中曾因将 `client.place_order` 返回 `code == "0"`（仅代表报单成功被交易所接收）直接累加进 `matched_trades += 1`，导致统计出 7.8 万笔虚假成交。
- **真理口径**：**成交判定必须且仅能以成交账单为准**。
  必须通过 WebSocket 的 `orders` 频道的 `state == "filled"` 推送，或者定期轮询 `GET /api/v5/account/bills` 流水账单中的实际成交记录。

### 6.2 强制 Post-Only（只做 Maker，绝不承受 Taker 惩罚）
- **致命隐患**：当行情剧烈单边跳水或拉升时，若使用普通 `limit` 限价单挂出可能直接发生交叉撮合，立刻被交易所按 0.05% Taker 吃单费率扣除。
- **铁律**：所有做市报文必须显式指定：
  ```python
  ord_type = "post_only"
  ```
  若订单在进入订单簿瞬间能与现有对手单成交，交易所撮合引擎将直接撤销该单，确保做市商永远只赚取流动性价差与返佣。

### 6.3 避免频繁全撤全挂导致队列重置
频繁取消挂单会失去在盘口中的时间优先排队权（Queue Priority）。
优化方案优先使用 `amend-order`（改单接口）微调价格，或者当市场最新中间价偏离挂单价格未超过设定波动阈值时保持静默不动。

---

## 七、资金自适应动态风险缩放模型

同一套数学参数绝不能在 35 USDT 微账户与 100,000 USDT 机构账户之间混用。
我们建立了账户权益动态感知模块 `as_dynamic_risk.py`。

### 7.1 资金阶梯动态参数映射表

| 资金梯队 (Equity USD) | 账户定位 | 风险厌恶系数 $\gamma$ | 最大净库存上限 `max_inventory` | 单笔挂单规格 `lot_sz` | 最优价差下限 `min_spread_bps` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$\le$ 100 USDT** | **微账户防爆仓** (如 35U 实盘) | **3.00** | **0.05 张** (~$40 市值) | `0.01` (最低合约) | **10.0 bps** (确保覆盖两倍手续费) |
| **100 ~ 1,000 USDT** | 小型测试账户 | **1.50** | **0.20 张** | `0.01` | **5.0 bps** |
| **1,000 ~ 10,000 USDT** | 中型账户 | **0.50** | **1.00 张** | `0.05` | **3.0 bps** |
| **> 10,000 USDT** | 机构级主力账户 | **0.0451** | **5.00 张** | `0.10` | **2.0 bps** |

### 7.2 核心自适应逻辑实现
```python
def compute_adaptive_as_parameters(equity_usd: float) -> dict:
    if equity_usd <= 100.0:
        return {
            "tier": "MICRO_ACCOUNT",
            "gamma_risk_aversion": 3.00,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.05,
            "lot_sz": "0.01",
            "min_spread_bps": 10.0
        }
    elif equity_usd <= 1000.0:
        return {
            "tier": "SMALL_ACCOUNT",
            "gamma_risk_aversion": 1.50,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.20,
            "lot_sz": "0.01",
            "min_spread_bps": 5.0
        }
    elif equity_usd <= 10000.0:
        return {
            "tier": "MEDIUM_ACCOUNT",
            "gamma_risk_aversion": 0.50,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 1.00,
            "lot_sz": "0.05",
            "min_spread_bps": 3.0
        }
    else:
        return {
            "tier": "INSTITUTIONAL_ACCOUNT",
            "gamma_risk_aversion": 0.0451,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 5.00,
            "lot_sz": "0.10",
            "min_spread_bps": 2.0
        }
```

---

## 八、高保真回测环境搭建与订单簿撮合仿真

禁止使用传统“触碰即成交”的极简回测。
必须在回测引擎中模拟真实的订单排队队列（Queue Position Model）。

### 8.1 真实排队与部分成交仿真机制
1. **盘口进入**：当做市商在 $P^b$ 价格挂单时，首先统计当前该价位已有挂单量 $V_{\text{ahead}}$。
2. **队列衰减**：后续收到的逐笔主动卖单流（Taker Sell Trades）逐步抵消前序挂单量：
   $$V_{\text{ahead}}(t) = V_{\text{ahead}}(t-1) - V_{\text{trade}}(t)$$
3. **正式成交**：仅当 $V_{\text{ahead}} \le 0$ 时，做市商订单才开始按照真实到达的成交量发生实质撮合。
4. **延迟惩罚**：所有订单发送与撤单操作强制加入 20ms~50ms 随机网络延迟与滑点惩罚。

---

## 九、实盘复现全流程与跨交易所迁移指南

### 9.1 VPS 环境一键拉起命令
```bash
# 1. 登录云服务器，拉取项目代码
git clone https://github.com/jssyxd/asgrid-mm.git /opt/nautilus_cluster
# 或盈利导向分支: git clone https://github.com/jssyxd/asgrid-arb.git /opt/nautilus_cluster
cd /opt/nautilus_cluster

# 2. 注入环境变量与实盘/模拟盘凭据
cat << 'EOF' > config/credentials.env
OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_secret
OKX_API_PASSPHRASE=your_passphrase
OKX_SIMULATED_TRADING=0
EOF

# 3. 构建并启动多容器微服务集群
docker compose up -d --build

# 4. 启动后台 10 分钟自动巡检与 Telegram 监控哨兵
crontab -l | { cat; echo "*/10 * * * * python3 /opt/nautilus_cluster/cron_hot_monitor.py 1 >> /opt/nautilus_cluster/logs/cron_hot.log 2>&1"; } | crontab -
```

### 9.2 跨交易所移植（适配 Binance / Bybit / Hyperliquid）
系统底层通信完全解耦为统一标准网关：
- 移植到 **Binance Futures**：只需替换 `gateway_okx_client.py` 中的 REST URL 格式（`fapi.binance.com`）与 HMAC-SHA256 签名头（`X-MBX-APIKEY`），核心做市算法与 ML 矩阵无需修改。
- 移植到 **Hyperliquid (Perp DEX)**：在网关层通过 `eth_account` 签名 EIP-712 报文，直接享受零 Gas 与纯正做市返佣。

### 9.3 切换交易标的（如从 BTC 切换至 DOGE / SOL）
1. 修改 `docker-compose.yml` 中的标的代号：`instId = "DOGE-USDT-SWAP"`。
2. 在 `compute_adaptive_as_parameters` 中将 `lot_sz` 调整为标的的最小合约张数（如 DOGE 单笔为 1 张 = 1000 DOGE）。
3. 重新校准该标的的基准 Parkinson/GK 波动率分位数。

---

## 十、常见陷阱排查手册与故障自愈清单

| 故障现象 | 根因诊断 | 处置方案与自愈措施 |
| :--- | :--- | :--- |
| **HTTP Error 401: Unauthorized (50110)** | VPS 发起请求时 IPv6 泄漏，命中了未在 OKX 白名单登记的 IPv6 地址 | 确认 `gateway_okx_client.py` 中强制注入了 `socket.AF_INET` 套接字拦截补丁。 |
| **Order failed: 51008 (Insufficient Margin)** | 账户余额过小，挂单数量超过最大允许保证金 | 检查 `as_dynamic_risk.py` 是否准确识别到 `MICRO_ACCOUNT` 并将下单手数锁定在最小单位 `0.01`。 |
| **频繁下单却几乎无实际成交** | 价差拉得过宽（$\gamma$ 偏大）或市场陷入超低波动死水期 | 结合近期 ATR 动态下调 `min_spread_bps`，或者切换至高 Beta 活跃代币。 |
| **实盘胜率很高但总资金持续微跌** | 产生大量被动换手，但价差毛利无法击穿交易所双边手续费 | 启用 `asgrid-arb` 方案，强制锁定基础价差 $\ge 10.0\text{ bps}$，彻底阻断无效超高频磨损。 |
| **容器内存缓慢爬升** | Python 垃圾回收机制在长周期事件循环中未及时将堆空间交还系统 | 检查 Docker 运行时是否配置了 `PYTHONMALLOC=malloc` 环境变量与 300MB 资源硬限制。 |
