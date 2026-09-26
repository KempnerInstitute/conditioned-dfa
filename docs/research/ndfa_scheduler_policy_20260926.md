# nDFA account and partition routing

The user authorized `kempner_dev` and `kempner_bsabatini_lab` on `kempner_requeue` and `kempner_h100`, preferring the account with better fairshare. Existing authorization for `kempner_h100_priority` and `kempner_eng` remains. Preserve the accelerator type required by each experiment. Requeue array throttles remain 60.

On 26 September UTC, Sabatini account fairshare was approximately 0.584 versus zero for `kempner_dev`. Pending GPU tasks were moved to that account. Eight CIFAR error-factor tasks were assigned to `kempner_h100`; the other pending GPU tasks use requeue. Already running work was not restarted. The regular H100 partition has a 16-GPU per-user limit, so moving every task there would not provide unrestricted concurrency.

The Sabatini account is disallowed on the `shared` CPU partition. Coordination jobs therefore retain `kempner_dev`. The two pending controllers were replaced with equivalent jobs 48674497 and 48674505, preserving dependencies and scientific inputs. Their private PATH contains the operational `sbatch_fairshare.py` wrapper, installed as `scheduler/bin/sbatch`. It checks both account fairshares at each authorized GPU submission and records the actual command and account. CPU jobs retain the compatible account. If fairshare lookup fails, the wrapper records the error and falls back to the last verified GPU account.

This is a scheduling amendment. All frozen source hashes were rechecked; configurations, seeds, selection, endpoints, and the one-confirmation policy are unchanged. The batch script itself was restored to its original bytes after verifying the CPU account restriction. The operational wrapper is separate from the frozen scientific source.

The round directory retains `account_partition_amendment_20260926.json`, `scheduler/controller_replacement.json`, and per-submission receipts in `scheduler/submissions/`. These preserve the original job IDs, failed scheduler updates, replacement dependencies, and effective account choices. No other project's jobs were modified.

## Protect the joint confirmation from preemption

A subsequent queue review found substantial requeue preemption and five interrupted CIFAR error-development endpoints. Development records remain intact. Because the frozen primary confirmation contrasts require uninterrupted timed runs, future confirmation training now uses non-preemptive partitions: H100 on `kempner_h100_priority` with `kempner_dev` / `kemp_gpu16_id38`, and H200 on `kempner_eng` with `kempner_dev` / `normal`. Site limits continue to govern concurrency. This changes scheduling only; every condition retains its prescribed accelerator model, seeds, budget, operator, and selection rule.

The operational wrapper reads the already frozen study stage to apply this routing. Development and test-evaluation submissions keep their existing fairshare policy. H100/H200 routing, unchanged development/evaluation behavior, and retention of hardware and array arguments were checked before installing the update. Source hashes were verified again. The record is `scheduler/confirmation_protection.json`; the 04:18 UTC recovery and queue snapshot are in `scheduler/recheck_20260926T041848Z/`.
