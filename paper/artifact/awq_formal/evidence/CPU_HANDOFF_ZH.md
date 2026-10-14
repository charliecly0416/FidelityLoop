# AWQ 正式 GPU 结果 CPU 交接说明

正式 12 格已全部结束，campaign 已退出，状态完整。物理包、模型、策略、窗口、计时和冻结 package 均未修改；未发生技术重试。

## 结果

- 10 格 `COMPLETED`，2 格技术有效的 `NEGATIVE_RESULT`。
- 两个负结果均为 H / steady：944 个在线请求全部按时，120 个离线请求中 119 个按时、1 个在 2100 秒观察窗口内未完成；两次重复一致。
- all2 recovery 两格、all2 steady 两格均通过容量锚点；guard steady 两格和其他 recovery 比较格均完成并已封存。
- 每格原始 attempt、controller/worker 日志、CELL_RESULT、CELL_MANIFEST、请求终态、账本和 guard 诊断均保留；CELL_01_AUDIT 至 CELL_12_AUDIT 为独立 CPU 复核。

## 完整性与后处理

最终 CPU 检查 23/23 通过，正式入口 verify、预检入口 verify、最终 status 均通过，GPU 释放快照显示两张卡显存约 1 MiB、利用率 0%。冻结包 SHA256 仍为 `505a6f6365a819482dd787891f025e75ad88a1ada8a985305b03ac65d307f310`，campaign lock 为 `d43fc9f561c7a25820e91b6295067803a688a49e1475d6f598261225eeb56801`。

包内原 `summarize` 的确定性问题已保留原始失败日志：`awq_analysis.py` 使用了未导入的 `verify_attempt_identity`。包外 adapter 只在该精确 NameError 下绑定冻结 `e6_runtime.py` 中的同名函数，在新目录重算；源码和科学证据字节不变。原始 return ZIP 另存，最终 ZIP 额外包含 `analysis/`、逐格账本和全部核验材料。

CPU 端应以最终 ZIP 的 `analysis/SUMMARY.json`、`analysis/accounting/`、逐格原始 output 和本说明为分析依据；负结果应按有效科学结果处理，不作为技术失败重跑。
