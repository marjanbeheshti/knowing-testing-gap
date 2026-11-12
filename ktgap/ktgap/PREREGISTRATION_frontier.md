
# Pre-registration: RQ1 (and RQ2 self-review) on a frontier model

Written before any frontier-model results were generated.



Model: Qwen/Qwen3.8-27B via the Jetstream2 inference service (thinking disabled), chosen before running.

Protocol: identical to the four open models (same prompts, 5 solutions per task at T=0.8,

5 assertions per test set at T=0.2, code mode and spec mode, EvalPlus labels).



Primary prediction (RQ1): on MBPP+, tests written while seeing the code catch fewer of the

model's bugs than tests written from the task alone (paired task bootstrap, 95% CI of the

spec-minus-code difference excludes zero).

Secondary (descriptive): the same comparison on HumanEval+; self-review verdicts on green MBPP+ code.

Reporting: results will be reported in the paper whatever they show, including a null result

or too few buggy solutions to test.




## Amendment (before the self-review rerun)

The self-review step uses a fixed 256-token limit for the review. For Qwen3.8-27B, 443 of 544

reviews were cut off before the verdict line (353 end mid-sentence). We therefore rerun the

self-review with a 1024-token limit, keeping everything else identical, and will report both

runs (256 tokens: 101 parsed verdicts; 1024 tokens: as observed). RQ1 is unaffected.

