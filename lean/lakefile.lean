import Lake
open Lake DSL

package dynamicDualEnergy where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.33.1"

@[default_target]
lean_lib DynamicDualEnergy where
