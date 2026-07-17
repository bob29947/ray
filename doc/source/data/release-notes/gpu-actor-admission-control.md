# Ray Data GPU actor admission control

Ray Data now starts GPU-backed actor pools on demand when operator resource
reservation is enabled. The executor protects one actor's declared resources for
each admitted stage, queues a single request at the first topological stage that
does not fit, and prevents later stages from leapfrogging it. Idle and pending
actors are released promptly as stages drain, while pipelines with sufficient
GPUs continue to stream across multiple actor stages.

Set the internal `DataContext._enable_gpu_actor_admission_control` rollback flag
to `False` to restore the previous eager actor-pool lifecycle.
