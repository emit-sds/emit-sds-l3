using ArchGDAL
using GDAL
using ArgParse2
using EllipsisNotation
using DelimitedFiles
using Logging
using Statistics
using Distributed
using Printf
using LinearAlgebra
using Combinatorics
using Random
include("src/datasets.jl")




function main()
    parser = ArgumentParser(prog="Orthorectifier", description="Apply a GLT to a source image to create an orthorectified output")
    add_argument!(parser, "glt_file")
    add_argument!(parser, "rawspace_file", help="filename of rawspace source file or, in the case of a mosaic_glt, a text-file list of raw space files")
    add_argument!(parser, "output_filename")
    add_argument!(parser, "--band_numbers", nargs="+", type=Int64, default=[-1], help="list of 0-based band numbers, or -1 for all")
    add_argument!(parser, "--n_cores", type=Int64, default=-1)
    add_argument!(parser, "--log_file", type=String, default=nothing)
    add_argument!(parser, "--log_level", type=String, default="INFO")
    add_argument!(parser, "--run_with_missing_files", type=Int64, default=0, choices=[0,1])
    add_argument!(parser, "--ip_head", type=String)
    add_argument!(parser, "--redis_password", type=String)
    add_argument!(parser, "--one_based_glt", type=Int64, choices=[0,1], default=0)
    add_argument!(parser, "--mosaic", type=Int64, choices=[0,1], default=0)
    add_argument!(parser, "--glt_nodata_value", type=Int64, default=-9999)
    args = parse_args(parser)

    if isnothing(args.log_file)
        logger = Logging.SimpleLogger()
    else
        logger = Logging.SimpleLogger(open(args.log_file, "w+"))
    end
    Logging.global_logger(logger)


    if args.mosaic == 1
        rawspace_files = readlines(open(args.rawspace_file, "r"))
        println(rawspace_files)
    else
        rawspace_files = [args.rawspace_file]
    end

    glt_dataset = ArchGDAL.read(args.glt_file)
    x_len = ArchGDAL.width(glt_dataset)
    y_len = ArchGDAL.height(glt_dataset)

    first_file_dataset = nothing
    for _ind in 1:length(rawspace_files)
        cont = false
        try
            first_file_dataset = ArchGDAL.read(rawspace_files[_ind])
        catch e
            cont = true
        end
        if cont == false
            break
        end
    end

    if args.band_numbers == -1
        output_bands = Array(1:ArchGDAL.nraster(first_file_dataset))
    else
        output_bands = Array(1:args.band_numbers)
    end

    initiate_output_datasets([args.output_filename], x_len, y_len, [length(output_bands)], glt_dataset)

    results = pmap(line->apply_mosaic_glt_line(line, args.glt_file, args.output_filename, rawspace_files, output_bands,
                                               line, args), 1:y_len)

end

@everwhere begin
    using ArchGDAL
    using EllipsisNotation
    using Logging
    using Statistics
    using Distributed
    using Printf
    using LinearAlgebra
    using Combinatorics
    using Random
    using Filesystem
    include("src/datasets.jl")

    function apply_mosaic_glt_line(line_index::Int64, glt_filename::String, output_filename::String,
                                   rawspace_files::Array{String}, output_bands: np.array, args)

        glt_dataset = ArchGDAL.read(glt_filename)
        x_len = ArchGDAL.width(glt_dataset)
        y_len = ArchGDAL.height(glt_dataset)

        if line % 100 == 0
            @info string("Applying line", line_index, "/", x_len)
        end

        img_dat = convert(Array{Float64},ArchGDAL.readraster(glt_filename)[:,line,:])
        valid_glt = all(glt_line .!= args.glt_nodata_value, dims=2)

        if sum(valid_glt) == 0
            return
        end

        glt_line[valid_glt, 2] = abs.(glt_line[valid_glt,2])
        glt_line[valid_glt, 1] = abs.(glt_line[valid_glt,1])
        if args.one_based_glt == 0
            glt_line[valid_glt,:] = glt_line[valid_glt,:] + 1
        end

        if args.mosaic == 1
            un_file_idx = unique(glt_line[valid_glt,end])
        else
            un_file_idx = [1]
        end

        output_dat = zeros(x_len, 1, length(output_bands)) .- 9999
        for _idx in un_file_idx
            if isfile(rawspace_files[_idx])
                if args.mosaic == 1
                    linematch = (glt_line[:,end] .== _idx) .& valid_glt
                else
                    linematch = valid_glt
                end

                if sum(linematch) > 0
                    output_dat[linematch, 1, :] = convert(Array{Float64}, ArchGDAL.readraster(rawspace_files[_idx])[glt_line[linematch,1], glt_line[linematch,2], output_bands])
                end
            end
        end
        outDataset = ArchGDAL.read(output_filename, flags=1)
        ArchGDAL.write!(outDataset, output_dat, [1:length(output_bands);], 0, line_index-1, x_len, 1)
        outDataset = nothing
        GC.gc()

    end

end

main()