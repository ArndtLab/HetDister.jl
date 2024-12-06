using DemoInfer
using Documenter

DocMeta.setdocmeta!(DemoInfer, :DocTestSetup, :(using DemoInfer); recursive=true)

makedocs(;
    modules=[DemoInfer],
    authors="Tommaso Stentella <stentell@molgen.mpg.de> and contributors",
    sitename="DemoInfer.jl",
    format=Documenter.HTML(;
        canonical="https://ArndtLab.github.io/DemoInfer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/ArndtLab/DemoInfer.jl",
    devbranch="main",
)
