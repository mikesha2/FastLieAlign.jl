import Lake
open Lake DSL

package «FastLieAlignProjections»

require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib «FastLieAlignProjections»
lean_lib «FastLieAlignGlobal»
