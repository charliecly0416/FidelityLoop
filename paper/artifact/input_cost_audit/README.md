# Input isolation and fixed-policy cost audit

Run `python -B artifact/input_cost_audit/verify.py` from the paper root (Python standard library only). This recomputes 28 online source-row intersections, three validation-file hash bindings, fixed-policy resource/API means, and the 16-ratio observed-repeat cost envelope.

`INPUT_ROWS.json` is a prompt-free projection of eight archived input files. Online raw-row hashes and timestamp offsets are retained, including duplicate hashes. Deployment hashes cover retained online entries in the source-row maps, matching the generated online arrivals. Source-file and generator hashes identify the archive inputs; verification of the reduced projection does not independently authenticate those original files or regenerate prompts. Source-row separation does not imply session/prompt independence; constructed offline jobs are outside this check.

`ENGINE_ROUTE_COUNTS.json` projects API terminal counts for the fixed PPO-7/guard pair. It binds terminal-source hashes and reduced record hashes to the existing `../decision_price/INPUTS.json`, which supplies cost components. Online fractions use all 358/944 arrivals. GPU seconds use the frozen 0.001 scenario-USD per GPU-second price. Window means average repeats, then weight windows equally. It does not measure commercial API quality or total physical compute.

The envelope holds both policy identities fixed and crosses their four existing repeat combinations, producing 16 ratios. It is descriptive, not a confidence interval or a new policy-selection result. Full-source checks were also rerun locally; no GPU inference was rerun.
