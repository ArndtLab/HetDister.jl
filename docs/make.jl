using HetDister
using Documenter

DocMeta.setdocmeta!(HetDister, :DocTestSetup, :(using HetDister); recursive=true)

makedocs(;
    modules=[HetDister],
    authors="Tommaso Stentella <stentell@molgen.mpg.de> and contributors",
    sitename="HetDister.jl",
    format=Documenter.HTML(;
        canonical="https://ArndtLab.github.io/HetDister.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Diagnostics" => "diagnostics.md"
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/ArndtLab/HetDister.jl",
    devbranch="main",
    versions=["stable" => "v^", "v#.#"],
)
