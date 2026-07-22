# Ray Data resource admission for GPU actors and shuffle gangs

Ray Data now coordinates complete resource floors for statically declared GPU
actor pools and GPU shuffle rank gangs. The executor starts resource owners only
after a topological admission decision, protects the actor pool's configured
minimum or one complete gang for each admitted stage, and prevents later stages
from leapfrogging the first
non-fitting frontier. Idle and pending actors are released as stages drain,
active calls finish without preemption, and shuffle ranks use one atomic
placement group that is removed after extraction.

Admission safety remains active when proportional operator reservation is
disabled. Actor-based `map_batches`, actor cuDF `map_groups`, GPU shuffle, and
GPU hash aggregate are covered. Dynamic or constrained actor resources and
labeled shuffle placement retain legacy scheduling. Mixing one with managed
owners logs a topology warning and makes that topology entirely legacy, avoiding
unsafe hybrid ownership. Undeclared GPU operators remain unchanged.

Set the internal `DataContext._enable_resource_admission_control` rollback flag
to `False` to restore eager actor-pool and shuffle resource acquisition for the
declared adapters. The private aggregate-resource contract consists of
`ResourceAdmissionSpec(minimum_resources, unit_resources, min_units, max_units)`
and `ResourceAdmissionGrant(max_units, may_submit)`.
