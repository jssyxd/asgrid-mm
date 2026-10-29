# Avellaneda-Stoikov ML Grid V2 完整决策流程与信息流全景规范

本文档详尽拆解实盘系统（`accountbot3`）从外部微观数据输入、因果特征计算、LightGBM 实时推理、随机最优控制定价到交易所真实成交记账的全流程。

---

## 阶段 1：网络工程与双向 WebSocket 监听

系统在网络层实行强制 IPv4 解析，保证发往 OKX 的底层 TCP 连接来自 `155.254.60.38`，严格匹配 API 白名单。

1. **公共 WebSocket 流 (`wss://ws.okx.com:443/ws/v5/public`)**：
   - 订阅 `books5`（五档盘口，100ms 连续推送）；
   - 订阅 `trades`（逐笔实时成交，带成交量与主动买卖方向）。
2. **私有 WebSocket 流 (`wss://ws.okx.com:443/ws/v5/private`)**：
   - HMAC-SHA256 签名登录；
   - 订阅 `orders` 频道（监听实盘订单执行事件）；
   - 订阅 `account` 频道（监听保证金、总权益、未实现盈亏）。

---

## 阶段 2：实时微观特征工程与动态计算

系统在内存环形队列维护近 60 秒的微观状态，完全因果无未来函数。

1. **Cont-Kukanov-Stoikov 连续订单流不平衡量 (OFI)**：
   $$\Delta W_b = \begin{cases} v_{b1}(t), & b_1(t) > b_1(t-1) \\ v_{b1}(t) - v_{b1}(t-1), & b_1(t) = b_1(t-1) \\ -v_{b1}(t-1), & b_1(t) < b_1(t-1) \end{cases}$$
   $$\Delta W_a = \begin{cases} -v_{a1}(t), & a_1(t) > a_1(t-1) \\ v_{a1}(t) - v_{a1}(t-1), & a_1(t) == a_1(t-1) \\ v_{a1}(t-1), & a_1(t) < a_1(t-1) \end{cases}$$
   $$\text{OFI} = \Delta W_b - \Delta W_a$$
2. **Garman-Klass 高频波动率**：
   $$\sigma_{\text{GK}}^2 = \frac{1}{N}\sum_{i=1}^N \left[ 0.5 \left(\ln\frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_i}{O_i}\right)^2 \right]$$
3. **均线偏离度与动量因子**：
   $$f_{\text{dev}} = \frac{C - \text{MA}_{20}}{\max(\text{ATR}_{14}, 0.05)}$$

---

## 阶段 3：多任务机器学习实时推理

输入特征向量：
$$X = [\sigma_{\text{parkinson}}, \sigma_{\text{GK}}, \text{ATR}_{14}, \text{OFI}_{\text{norm}}, f_{\text{dev}}, s_{r1}, \text{Vol}_{\text{accel}}]$$

LightGBM 多任务推理引擎产出三项核心微观信号：
1. **短周期价格漂移 $\mu_{\text{pred}}$**：预测价格未来微观倾斜；
2. **下砸毒性概率 $P_{\text{toxic\_down}}$**：预测未来一分钟是否出现超过 1.8 ATR 的暴跌穿盘；
3. **冲顶毒性概率 $P_{\text{toxic\_up}}$**：预测向上暴力拉升冲击。

---

## 阶段 4：Avellaneda-Stoikov 随机最优控制定价

基于 Hamilton-Jacobi-Bellman (HJB) 随机最优控制理论计算保留价格与非对称价差：

1. **内部保留价格 (Reservation Price)**：
   $$r(s, q, t) = s - q \cdot \gamma \cdot \sigma^2 \cdot s \cdot c_1 + \alpha_{\text{drift}} \cdot \mu_{\text{pred}} \cdot s \cdot c_2$$
   - 持有多头时，保留价格向下偏置，促使卖单接近盘口加速平仓；
   - 持有空头时，保留价格向上偏置，促使买单抬高迅速补空。
2. **显式手续费门槛与最优价差**：
   $$\delta_{\text{base}} = \max\left( s \cdot 0.0004, \, \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right) s \cdot 0.0004 \right)$$
   - 强制半价差 $\ge 4.0\text{ bps}$（双边总价差 $\ge 8.0\text{ bps}$），对 0.02% 双向 Maker 费率形成 2.0 倍以上利润覆盖。

---

## 阶段 5：动态风控看门狗与断路器判定

报单前必须依次通过四道硬风控过滤：

1. **波动率激增保护**：若 $\sigma_{\text{GK}} > 2.0 \times \text{Vol}_{\text{median}}$，价差扩大 2.0 倍；
2. **毒性大单流断路器**：
   - 若 $P_{\text{toxic\_down}} \ge 0.65$ 且均线破位：切断买单（`BID_CUT`），严禁接飞刀；
   - 若 $P_{\text{toxic\_up}} \ge 0.65$：切断卖单（`ASK_CUT`）。
3. **库存限额保护**：持仓达到 $\pm 0.05$ 张时切断同向开仓；
4. **保证金安全保护**：针对 30 USDT 小微本金，占用资金严格控制在 13.50 USDT (45%) 以内。

---

## 阶段 6：队列感知生命周期管理与严格成交核算

克服盲目撤单重挂造成的 FIFO 队列位置丢失：

1. **队列优先权保持 (Queue Position Preservation)**：
   - 检查现有挂单，若价差偏离 $\le 1.0\text{ USD}$，保留该订单（`BID_QUEUE_HOLD` / `ASK_QUEUE_HOLD`）；
   - 仅在价格偏离超标或触发风控断路时才批量撤单。
2. **严格 Post-Only 报单**：
   - 强制使用 `ord_type="post_only"`，若订单穿透盘口则直接取消，绝不交纳 0.05% Taker 费率。
3. **严格成交记账 (Strict Fill Accounting)**：
   - 仅由私有 WebSocket `orders` 频道的 `state in ["filled", "partially_filled"]` 触发成交流水记录；
   - 杜绝挂单确认当成真实成交的严重错误。
