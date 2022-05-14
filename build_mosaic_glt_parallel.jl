using ArchGDAL
using ArgParse2
using EllipsisNotation
using DelimitedFiles
using Logging
using Statistics


function main()

    parser = ArgumentParser(prog = "GLT Builder",
                        description = "Build GLTs from one or more files")

    add_argument!(parser, "output_filename", type = String, help = "File to write GLT results to")
    add_argument!(parser, "igm_file_list", type = String, help = "IGM file or list of files to build GLT from")
    add_argument!(parser, "target_resolution", type = Float64, nargs=2, help = "GSD (x and y).")
    add_argument!(parser, "--criteria_mode", type = String, default = "distance", help = "Band-ordering criteria mode.  Options are min or max (require criteria file), or distance (uses closest point)")
    add_argument!(parser, "--criteria_band", type = Int64, default = 1, help = "band of criteria file to use")
    add_argument!(parser, "--criteria_file_list", type = String, help = "file(s) to be used for criteria")
    add_argument!(parser, "--target_extent_ul_lr", type = Float64, nargs=4, help = "extent to build the mosaic of")
    add_argument!(parser, "--mosaic", type = Int32, default=1, help = "treat as a mosaic")
    add_argument!(parser, "--output_epsg", type = Int32, default=4326, help = "epsg to write to destination")
    add_argument!(parser, "--log_file", type = String, default = nothing, help = "log file to write to")
    args = parse_args(parser)

    if isnothing(args.log_file)
        logger = Logging.SimpleLogger()
    else
        logger = Logging.SimpleLogger(args.log_file)
    end
    Logging.global_logger(logger)

    if args.target_resolution[2] > 0
        args.target_resolution[2] *= -1
        @info string("Converting second resolution argument to be negative, currently necessary for parsing.  Revised to: ", args.target_resolution)
    end

    #if ! (args.criteria_mode ! in ["min","max","distance"])
    #    error("Invalid criteria_mode, expected on of min, max, distance")
    #end

    if args.mosaic == 1
        igm_files = readdlm(args.igm_file_list, String)
    else
        igm_files = [args.igm_file_list]
    end

    if args.criteria_mode != "distance"
        if args.mosaic == 1
            criteria_files = readdlm(args.criteria_file_list, String)
        else
            criteria_files = [args.criteria_file_list]
        end
        # TODO: add check to make sure criteria file dimensions match igm file dimensions
    end

    if length(args.target_extent_ul_lr) > 0
        ullr = args.target_extent_ul_lr
        min_x = ullr[1]
        max_y = ullr[2]
        max_x = ullr[3]
        min_y = ullr[4]
    else
        min_x, max_y, max_x, min_y = get_bounding_extent_igms(igm_files)
    end
    @info "IGM bounds: $min_x, $max_y, $max_x, $min_y"

    @info "Tap to a regular Grid"
    min_x = tap_bounds(min_x, args.target_resolution[1], "down")
    max_y = tap_bounds(max_y, args.target_resolution[2], "up")
    max_x = tap_bounds(max_x, args.target_resolution[1], "up")
    min_y = tap_bounds(min_y, args.target_resolution[2], "down")

    @info "Tapped bounds: $min_x, $max_y, $max_x, $min_y"

    x_size_px = Int32(ceil((max_x - min_x) / args.target_resolution[1]))
    y_size_px = Int32(ceil((max_y - min_y) / -args.target_resolution[2]))

    @info "Output Image Size (x,y): $x_size_px, $y_size_px.  Creating output dataset."
    if args.mosaic == 1
        output_bands = 3
    else
        output_bands = 2
    end
    outDataset = ArchGDAL.create(args.output_filename, driver=ArchGDAL.getdriver("ENVI"), width=x_size_px,
    height=y_size_px, nbands=output_bands, dtype=Float32)
    ArchGDAL.setproj!(outDataset, ArchGDAL.toWKT(ArchGDAL.importEPSG(args.output_epsg)))
    ArchGDAL.setgeotransform!(outDataset, [min_x, args.target_resolution[1], 0, max_y, 0, args.target_resolution[2]])

    @info "Populate target grid."
    grid = Array{Float64}(undef, y_size_px, x_size_px, 2)
    grid[..,1] = fill(1,y_size_px,x_size_px) .* LinRange(min_x + args.target_resolution[1]/2,min_x + args.target_resolution[1] * (1/2 + x_size_px - 1), x_size_px)[[CartesianIndex()],:]
    grid[..,2] = fill(1,y_size_px,x_size_px) .* LinRange(max_y + args.target_resolution[2]/2,max_y + args.target_resolution[2] * (1/2 + y_size_px - 1), y_size_px)[:,[CartesianIndex()]]

    @info "Create GLT."
    best = fill(1e12, y_size_px, x_size_px, 4)
    if args.criteria_mode == "max"
        best = best .* -1
    end
    best[..,1:3] .= -9999

    max_offset_distance = sqrt(sum(args.target_resolution.^2))*3
    pixel_buffer_window = 1
end



function get_bounding_extent_igms(file_list::Array{String}, return_per_file_xy::Bool=false)
    file_min_xy = Array{Float64}(undef,size(file_list)[1],2)
    file_max_xy = Array{Float64}(undef,size(file_list)[1],2)

    results = pmap(file_idx->read_igm_bounds(file_idx,file_list), 1:length(file_list))
    for res in results
        file_min_xy[res[1],:] = res[2]
        file_max_xy[res[1],:] = res[3]
    end

    min_x = minimum(filter(!isnan,file_min_xy[:,1]))
    min_y = minimum(filter(!isnan,file_min_xy[:,2]))
    max_x = maximum(filter(!isnan,file_max_xy[:,1]))
    max_y = maximum(filter(!isnan,file_max_xy[:,2]))

    if return_per_file_xy
        return min_x, max_y, max_x, min_y, file_min_xy, file_max_xy
    else
        return min_x, max_y, max_x, min_y
    end
end



@everwhere begin
using ArchGDAL
using EllipsisNotation
using DelimitedFiles
using Logging
using Statistics


function read_igm_bounds(file_idx::Int32, filenames::Array{String})

    dataset = ArchGDAL.read(filenames[file_idx])
    igm = ArchGDAL.read(dataset)
    file_min_xy = [minimum(igm[..,1]), minimum(igm[..,2])]
    file_max_xy = [maximum(igm[..,1]), maximum(igm[..,2])]

    return file_idx, file_min_xy, file_max_xy
end









main()