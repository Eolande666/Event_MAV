# 固定机位共享刚体事件检测

每个候选只计算一次IOC。候选定位使用完整33.333 ms事件窗口，旋转速度场默认使用窗口末尾9 ms。GT只用于离线评价，不参与检测。

## 算法流程

1. 按像素统计正负极性转换次数以及转换覆盖的时间区间，生成候选显著图。
2. 从显著图提取候选，并计算重复活动分数 `P` 和空间结构分数 `C`。
3. 每窗最多对4个候选计算IOC。正、负极性的有效IOC像素按5×5非重叠块求均值，降低相邻速度估计之间的相关性。
4. 两种极性共同拟合一个角速度：

       v+(x,y)   = t+ + omega * [-(y-cy), x-cx]
       s*v-(x,y) = t- + omega * [-(y-cy), x-cx]

   `s` 在 `+1/-1` 中选择残差较小者，`t+` 和 `t-` 分别吸收两种极性的整体平移。
5. 从共享刚体模型计算三个无量纲旋转量：

       S = 共享角速度相对纯平移模型的F检验累计概率
       F = 1 - RSS_rotation / RSS_translation
       G = E_skew / (E_skew + E_symmetric)
       Rot = S * F * G

   `G` 比较仿射速度梯度中的反对称旋转能量和对称形变能量。少于3个不共线有效块时，旋转记为不可观测。
6. 使用固定联合评分：

       Q = 0.55*Rot + 0.30*P + 0.15*C

   `Q >= 0.34` 时立即输出，并通过最近中心关联维持检测ID。

## 运行

输出目录必须尚不存在。

```powershell
& D:\anaconda3\envs\EVENT\python.exe E:\WYY\xuantie\Fixed_event_det\script\run_detector.py `
  --input E:\WYY\RGBEVENT_OCCrestoration\28\Event\h5 `
  --out E:\WYY\xuantie\Fixed_event_det\outputs\28_rigid `
  --width 1280 --height 720 --time-unit us `
  --config E:\WYY\xuantie\Fixed_event_det\script\config.default.json `
  --overlap-policy trim --save-frames --log-every 100
```

209数据只需替换输入和输出目录。`--show-candidates` 可额外绘制未通过候选。`--rotation-window-ms 33.333` 仍只计算一次IOC，但使用完整检测窗口。

## 输出

框上数值：

- `Q`：最终联合分数。
- `Rot`：进入联合评分的 `S*F*G`。
- `Om`：共享有符号角速度，单位约为 `s^-1`。
- `F`：刚体旋转相对纯平移的解释率。
- `G`：旋转能量占旋转与形变总能量的比例。
- `Rot=N/A`：有效块不足或空间分布退化。

CSV同时保存旋转显著度、有效块数、残差、极性符号及两种极性的单独角速度。绿色框是输出检测；显示候选时，橙色表示评分未通过，灰色表示IOC预算用尽。

## 完整数据结果

IoA@0.5：

| 序列 | TP | FP | FN | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 28 | 1552 | 206 | 1168 | 88.28% | 57.06% | 69.32% |
| 209 | 2383 | 2164 | 883 | 52.41% | 72.96% | 61.00% |

纯检测耗时：28号平均40.80 ms/窗、P95 48.68 ms；209号平均69.85 ms/窗、P95 120.27 ms。

## 测试

```powershell
cd E:\WYY\xuantie\Fixed_event_det\script
& D:\anaconda3\envs\EVENT\python.exe -m unittest discover -v
```
