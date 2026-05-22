using Documenter
using StencilAssembly
using StencilCore

makedocs(;
    sitename = "StencilAssembly.jl",
    # StencilCore listed so the re-exported types' docstrings resolve in @docs.
    modules = [StencilAssembly, StencilCore],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API reference" => "api.md",
    ],
    checkdocs = :none,
    warnonly = [:cross_references],
)

deploydocs(;
    repo = "github.com/vlc1/StencilAssembly.jl",
    devbranch = "main",
)
