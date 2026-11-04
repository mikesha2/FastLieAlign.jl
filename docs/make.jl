# Build the documentation site.  From the repository root:
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   JULIA_LOAD_PATH="@:$PWD:@stdlib" julia --project=docs docs/make.jl
# The package is loaded from the repository environment through the load path; the path
# must be absolute because the example blocks run from the build directory.
# The browser demo in web/ is copied into the site as demo/.
using Documenter, FastLieAlign

const USER = "USERNAME"                                # set to the GitHub user or organization before deploying
const REPO = "github.com/$USER/FastLieAlign.jl"

makedocs(;
    sitename = "FastLieAlign.jl",
    modules = [FastLieAlign],
    authors = "Congzhou M. Sha",
    repo = Remotes.GitHub(USER, "FastLieAlign.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://$USER.github.io/FastLieAlign.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "User guide" => "guide.md",
        "Reproducing the paper" => "reproduce.md",
        "Browser demo" => "demo.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

# ship the static WebAssembly/WebGPU page with the documentation
let demo = joinpath(@__DIR__, "build", "demo"), web = normpath(joinpath(@__DIR__, "..", "web"))
    rm(demo; force = true, recursive = true)
    cp(web, demo)
    rm(joinpath(demo, "src"); force = true, recursive = true)   # C sources are not needed on the site
end

deploydocs(; repo = REPO * ".git", devbranch = "main", push_preview = false)
