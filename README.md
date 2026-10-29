# asgrid-mm: 做市商 (Market Maker) 刷量与返佣套利执行体系

本仓库专为 **做市商 (Market Maker) 刷量导向、交易手续费返佣套利、积分所空投挖矿** 设计。
系统基于 Avellaneda-Stoikov 微观结构高频做市模型，结合 L2 订单簿深度与 LightGBM 毒性订单流保护，实现大吞吐量被动流动性挂单。

---

## 核心定位：做市商刷量导向 (Volume & Rebate Mining)

本策略不追求单边方向性收益，而是追求在维持极低价格敞口风险的前提下，最大化制造有效交易量（Volume Generation）。
核心变现路径为：
1. **交易所做市商负费率返佣 (Maker Fee Rebate)**；
2. **新兴交易所与 DEX 积分交易挖矿 (Trade Mining / Points Farming)**；
3. **项目方做市服务与流动性外包支持 (Liquidity Provision Partnership)**。

---

## 一、 如何对接合作做市商与获取做市扶持？

大型量化团队与做市机构通常有专门的做市合作计划，普通交易者可以通过以下步骤接入：

### 1. 达到 VIP / 做市商准入门槛
- 各大中心化交易所（如 OKX、Binance、Bybit、Gate、Bitget）均设有 VIP 做市商等级。
- **做市商特权**：Maker 费率通常为 **负费率**（如 -0.005% 至 -0.015%），挂单被撮合成交不仅不收手续费，交易所反而向做市商支付现金补贴。
- **免审核绿通策略**：若已有其他平台 VIP 证明（如 Binance VIP 3），可直接向目标交易所商务申请 30 至 60 天的 **VIP 等级体验期（Status Match）**，直接享受最高档做市负费率。

### 2. 项目方与做市商联名合作 (KOL / Agency Volume)
- 许多新上线代币项目方急需维持盘口深度与成交量，以满足 CoinMarketCap、CoinGecko 或头部 CEX 上币的活跃度要求。
- 可以使用本系统的微观做市脚本，配置中性价差（Spread = 5~10 bps），为项目方在指定币对提供被动双边深度，收取固定的做市月费或按成交量提成。

---

## 二、 优质返佣交易所与积分交易所精选名单

做市刷量策略的核心瓶颈是**手续费摩擦**。必须在以下两类平台上运行方能产生正向净收益：

### 1. 负费率 / 极高返佣的头部中心化交易所 (CEX)
- **OKX 做市商计划 (OKX MM Program)**：
  - 合约做市商最高享受 -0.015% 的挂单返现。
  - 支持高频 API、每秒上百次并发下单、批量撤单接口（`cancel-batch-orders`）。
- **Gate.io / Bitget 做市商体系**：
  - 对中小资金做市商准入门槛相对较低（月交易量达 500 万美元即可申请 Maker 零费率或轻微负费率）。
- **Bybit 做市商通道**：
  - 提供专属的低延迟 WebSocket 专线与反洗钱绿名单支持。

### 2. 积分所与永续 DEX 交易挖矿 (Points Farming / Airdrop DEX)
在去中心化衍生品交易所（Perp DEX）上，刷量做市商是获取代币空投和交易奖励的核心主力：
- **Hyperliquid (HL)**：
  - 拥有极深的原生 L1 订单簿与零 Gas 撮合环境。
  - 提供 Maker 费用返佣及高额做市积分（Hyperliquid Points）。
- **Aevo / Drift Protocol / Orderly Network**：
  - 采用订单簿架构，对双边挂单深度提供交易竞赛奖励与代币空投补贴。
- **Paradex / dYdX v4**：
  - 拥有专门的流动性提供激励池（LP Rewards），按挂单距中间价的深度和持续时间自动结算代币奖励。

---

## 三、 执行架构与核心防坑规范

### 1. 强制只做 Maker (Post-Only Enforced)
代码中所有限价单必须强制启用 `ordType="post_only"`。
当订单可能与对手方挂单直接撮合时，交易所会主动取消该挂单，绝不以 Taker 身份成交，避免被收取高达 0.05% 的吃单惩罚性手续费。

### 2. 智能改单取代暴力重挂 (Modify Order Protocol)
避免使用“全部撤单再全量重挂”的原始做法，改用 OKX 的 `amend-order` 接口直接调整价格。
保留原单在撮合队列中的优先位置，降低 API 频控消耗与被恶意插队的风险。

### 3. 双边非对称价差与毒性避险 (OFI + Toxic Breaker)
通过 Cont-Kukanov-Stoikov 订单流不平衡量（OFI）结合 LightGBM 毒性预测模型：
- 当识别到大单连续下砸（$P_{\text{toxic\_down}} \ge 0.70$）时，毫秒级切断买单，防止被动接飞刀；
- 当市场平稳震荡时，迅速贴近盘口双边报价，最大化吞吐刷量流水。

---

## 四、 自动化实盘运行指南

```bash
# 1. 配置实盘 API 密钥与环境变量
cp live_cluster/config/okx_credentials.env.example live_cluster/config/okx_credentials.env

# 2. 启动 Docker 做市商微服务集群
cd live_cluster && docker compose up -d

# 3. 实时查看做市挂单与撮合流水
docker logs -f nautilus_vps_003_as_grid --tail 30
```
