#!/usr/bin/env julia

using PlmDCA
using ArgParse
using FilePathsBase
using CSV, DataFrames

function main()
    parser = ArgParseSettings()
    @add_arg_table parser begin
        "alignment"
        help = "Input alignment file"
        arg_type = String
    end

    # parse command line 
    args = parse_args(parser)
    alignment_file = args["alignment"]

    # run PlmDCA
    println("Processing alignment file: $alignment_file")
    res = plmdca_frompy(alignment_file)

    # Save to CSV
    output_file = splitext(alignment_file)[1] * ".csv"
    println("Saving results to CSV file: $output_file")
    df = DataFrame(res1 = first.(res.score), res2 = getindex.(res.score, 2), score = last.(res.score))
    CSV.write(output_file, df)
    return 0
end


main()
