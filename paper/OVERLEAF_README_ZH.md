# FidelityLoop Overleaf 源文件

主文件为 `main.tex`（正文十页，含参考文献与 AI 披露共十二页）；补充材料主文件为 `supplement.tex`（36 页）。使用 pdfLaTeX 和 BibTeX，必要时 Recompile from scratch。

本轮将 36 次实际 HTTP 推理的前瞻选择研究融入摘要、贡献、方法、结果、讨论与结论；补充 Section R 给出完整结果与审计边界。原始 PPO 公式、关键负例、机制及已有扩展证据均保留。部分详细参数移至 `protocol_details.tex`，未改动官方样式。

`artifact/realapi_prospective/` 包含密封便携证据与 CPU 检查，不参与 LaTeX 编译；它需要同步到匿名材料站，供审稿人复算。本工程中的匿名网址不代表网站已自动更新。

本轮对外提供的是累积替换包。覆盖原完整 v9r19 工程中的同名文件并新增文件即可，无需依次应用历史补丁。投稿主论文与补充 PDF 分开上传；源文件 ZIP 不是主论文 PDF。
