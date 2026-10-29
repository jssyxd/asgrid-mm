# 架构教训与避坑备忘录 (Lessons Learned & Anti-Recurrence Guide)

本文档记录 Strategy 012 (Avellaneda-Stoikov ML Grid) 在高频回测与 OKX 实盘（accountbot3 30 USDT 微账户）运行中暴露的五大核心缺陷与工程整改方案，杜绝未来部署重犯相同错误。

---

## 1. 微账户资金与机构级参数错配漏洞 (Capital-Scale Mismatch)

### 故障现象 (Measured)
- 初始本金仅 30.0 USDT，杠杆倍数 3x；
- 实盘在短时间内累积多头持仓达 **0.11 张 BTC-USDT-SWAP 合约**（约 \$89.10 USD 名义敞口）；
- 占用初始保证金 **29.70 USDT**，可用资金降至 **0.25 USDT**（保证金占用率 99.1%）；
- 触发 OKX 交易限制，买单全部被拒（`BID_REJ(All operations failed)`），系统丧失双向做市能力。

### 根因剖析
- 原版代码沿用了 Kaggle 离线以 100,000 USDT 训练的**机构级参数**（$\gamma=0.0451$）；
- 机构账户规模庞大，单位库存引起的保留价格向下偏置微乎其微：
  $$\text{skew} = q \cdot \gamma \cdot \sigma^2 \cdot S \cdot c$$
  在 $q=0.05$ 时偏置仅有几美分，导致策略完全不急于降库存；
- 结合缺少小额硬风控上限，多头持续累积直至耗尽保证金。

### 整改措施 (Standardized)
- 引入微账户专用参数（$Equity \le 100\text{ USDT}$）：
  - **风险厌恶系数 $\gamma = 3.00$**（将保留价格倾斜力度放大 66 倍，一旦持有多头立即显著下调报价平仓）；
  - **最大持仓硬上限 $\text{max\_inventory} = 0.05\text{ 张}$**（5 笔 0.01 张，名义敞口约 \$40.50，占用保证金仅 \$13.50，占本金 45%）；
  - **保留 55% 资金缓冲**（应对行情波动与维持保证金，杜绝打满）。

---

## 2. 挂单当成交的虚假流水漏洞 (Phantom Trade Accounting)

### 故障现象 (Measured)
- 实盘运行日志中频繁出现上万笔交易记录，但交易所实际成交记录仅寥寥数笔；
- 产生了虚假的高额名义流水与手续费统计。

### 根因剖析
- 旧代码在向 OKX 下达 Post-Only 限价单后，直接以 API 响应 `code == "0"` 判定为成交并写入 `recent_fills`；
- `code == "0"` 仅代表交易所系统接收并挂单成功，并未发生撮合；
- 2 秒后下一轮循环执行全额撤单，未撮合的挂单被取消，虚增了海量伪流水。

### 整改措施 (Standardized)
- 彻底剥离报单确认与成交核算；
- 接入 OKX 私有 WebSocket `orders` 频道与 `/api/v5/trade/fills` 补差核验；
- 只有当交易所推送 `state in ["filled", "partially_filled"]` 且 `fillSz > 0` 时，才执行成交记账。

---

## 3. 全撤重挂导致的队列优先权摧毁 (Queue Priority Destruction)

### 故障现象
- 策略长时间挂单却极难成交，或一旦成交往往是被大单深度击穿。

### 根因剖析
- 每次主循环（约 1~2 秒）无脑对所有挂单执行批量撤销并重新挂单；
- 交易所撮合引擎遵循价格优先与时间优先 (FIFO)；
- 频繁撤单重挂使订单永远位于排队队列的最末端，彻底丧失 Maker 被动成交权。

### 整改措施 (Standardized)
- 增加**队列感知挂单保持机制 (Queue Priority Preservation)**；
- 轮询新计算的最优报价：若当前已有挂单且价格偏离 $\le 1.0\text{ USD}$，保留该订单（`BID_QUEUE_HOLD` / `ASK_QUEUE_HOLD`），绝不撤销；
- 仅在价格偏离超标或触发毒性熔断时才撤单。

---

## 4. 缺乏 Post-Only 导致的 Taker 滑点与费率倒挂 (Taker Crossing Risk)

### 故障现象
- 在行情剧烈跳空时，挂出的限价单意外被当作 Taker 吃单撮合，扣除 0.05% 高额手续费。

### 根因剖析
- 报单使用了普通的 `ord_type="limit"`，一旦报价穿透对手盘买一/卖一，将直接以市价吃单撮合。

### 整改措施 (Standardized)
- 全流程强制 `ord_type="post_only"`；
- 若订单穿透对手盘，交易所直接拒绝并返回，坚决不交纳 Taker 惩罚性费率。

---

## 5. 特征工程未来函数泄漏与时间尺度错配 (Feature Lookahead)

### 故障现象
- Kaggle 训练集上收益极高，但样本外测试集收益为负（-5.03%）。

### 根因剖析
- `trainer.py` 中对 `rolling(20)` 调用了 `.bfill()`，使得前 20 根 Bar 的历史数据被未来的均值填充，产生了前瞻泄漏；
- 预测目标采用未来 5 根 Bar（整整 5 分钟后）的价格漂移，无法有效指导秒级高频撤单。

### 整改措施 (Standardized)
- 彻底剔除 `.bfill()`，统一采用纯因果的 `rolling(20, min_periods=1).mean()`；
- 预测目标调整为微观 $H=1$（下一根 1m Bar）与实时 Tick OFI 动态更新。
