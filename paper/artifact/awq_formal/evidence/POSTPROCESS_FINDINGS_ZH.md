# 正式包后处理静态检查

以下问题来自随包原始源码，不影响当前 GPU 物理入口；冻结 package 未修改。

1. awq_analysis.py 的 recompute 调用 verify_attempt_identity，但模块没有导入该函数。真实有效格进入该分支后会 NameError。修复应只补入 e6_runtime 原有函数，不改变验收逻辑。
2. awq_formal_entry.py 的 build_return 接受并验证 analysis/SUMMARY.json，但 roots 只有 source_package 与 output，未把 analysis 加入归档。必须在最终封装时包含独立 analysis/，并重新生成 RETURN_MANIFEST；仅 verify-return PASS 不足以证明 analysis 存在。

上述缺陷尚未触发在实际终态分析上。计划在实验结束后先执行原命令、保留原错误或缺失证据，再用包外后处理适配补齐导入与归档成员；原 source_package 的哈希保持不变，任何适配代码与说明一并回传。不会因此重跑物理格或修改计时、策略、模型、预算。
