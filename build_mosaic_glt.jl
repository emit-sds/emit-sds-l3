using ArchGDAL
using ArgParse
using EllipsisNotation
using DelimitedFiles
using Logging
using Debugger
using Statistics


function main()

    s = ArgParseSettings()
    @add_arg_table s begin
        "output_filename"
            help = "File to write results to"
            arg_type = String
            required = true
        "igm_file_list"
            help = "File to write results to"
            arg_type = String
            required = true
        "target_resolution"
            help = "GSD to build the mosaic at"
            arg_type = Float64
            nargs = 2
            required = true
        "--criteria_mode"
            help= "Band-ordering criteria mode.  Options are min or max (require criteria file), or distance (uses closest point)"
            arg_type = String
            required = false
            default = "distance"
        "--criteria_band"
            help = "band of criteria file list to use"
            arg_type = Int64
            required = false
            default = 1
        "--criteria_file_list"
            help = "files used for criteria"
            arg_type = Float64
            required = false
        "--target_extent_ul_lr"
            help = "extent to build the mosaic of"
            arg_type = Float64
            nargs = 4
            required = false
        "--log_file"
            help = "log file to write to"
            arg_type = String
            required = false
            default = nothing
    end

    args = parse_args(ARGS, s)
    target_resolution = args["target_resolution"]

    if isnothing(args["log_file"])
        logger = Logging.SimpleLogger()
    else
        logger = Logging.SimpleLogger(args["log_file"])
    end
    Logging.global_logger(logger)

    #if ! (args["criteria_mode"] ! in ["min","max","distance"])
    #    error("Invalid criteria_mode, expected on of min, max, distance")
    #end

    igm_files = readdlm(args["igm_file_list"], String)
    if args["criteria_mode"] != "distance"
        criteria_files = readdlm(args["criteria_file_list"], String)
        # TODO: add check to make sure criteria file dimensions match igm file dimensions
    end

    if length(args["target_extent_ul_lr"]) > 0
        ullr = args["target_extent_ul_lr"]
        min_x = ullr[1]
        max_y = ullr[2]
        max_x = ullr[3]
        min_y = ullr[4]
    else
        min_x, max_y, max_x, min_y = get_bounding_extent_igms(igm_files)
    end
    @info "IGM bounds: $min_x, $max_y, $max_x, $min_y"

    @info "Tap to a regular Grid"
    min_x = tap_bounds(min_x, target_resolution[1], "down")
    max_y = tap_bounds(max_y, target_resolution[2], "up")
    max_x = tap_bounds(max_x, target_resolution[1], "up")
    min_y = tap_bounds(min_y, target_resolution[2], "down")

    @info "Tapped bounds: $min_x, $max_y, $max_x, $min_y"

    x_size_px = Int32(ceil((max_x - min_x) / target_resolution[1]))
    y_size_px = Int32(ceil((max_y - min_y) / -target_resolution[2]))

    @info "Output Image Size (x,y): $x_size_px, $y_size_px.  Creating output dataset."
    outDataset = ArchGDAL.create(args["output_filename"], driver=ArchGDAL.getdriver("ENVI"), width=x_size_px,
    height=y_size_px, nbands=3, dtype=Float32)
    ArchGDAL.setproj!(outDataset, ArchGDAL.toWKT(ArchGDAL.importEPSG(4326)))
    ArchGDAL.setgeotransform!(outDataset, [min_x, target_resolution[1], 0, max_y, 0, target_resolution[2]])

    @info "Populate target grid."
    grid = Array{Float64}(undef, y_size_px, x_size_px, 2)
    grid[..,1] = fill(1,y_size_px,x_size_px) .* LinRange(min_x + target_resolution[1]/2,min_x + target_resolution[1] * (1/2 + x_size_px - 1), x_size_px)[[CartesianIndex()],:]
    grid[..,2] = fill(1,y_size_px,x_size_px) .* LinRange(max_y + target_resolution[2]/2,max_y + target_resolution[2] * (1/2 + y_size_px - 1), y_size_px)[:,[CartesianIndex()]]

    @info "Create GLT."
    best = fill(1e12, y_size_px, x_size_px, 4)
    best[..,1:3] .= -9999

    max_offset_distance = sqrt(sum(target_resolution.^2))*3
    pixel_buffer_window = 1

    total_found = 0
    for (file_idx, igm_file) in enumerate(igm_files)
        @info "$igm_file"
        dataset = ArchGDAL.read(igm_file)
        igm = PermutedDimsArray(ArchGDAL.read(dataset), (2,1,3))
        if minimum(igm[..,1]) > grid[1,end-1,1] || maximum(igm[..,1]) < grid[1,1,1] ||
           minimum(igm[..,2]) > grid[1,1,2] || maximum(igm[..,2]) < grid[end-1,1,2]
            #println(minimum(igm[..,1]), " > ", grid[1,end-1,1], " ", maximum(igm[..,1]), " < ", grid[1,1,1])
            #println(minimum(igm[..,2]), " > ", grid[1,1,2], " ", maximum(igm[..,2]), " < ", grid[end-1,1,2])
            continue
        else
            println("Entering")
        end
        if args["criteria_mode"] != "distance"
            criteria_dataset = ArchGDAL.read(criteria_files[file_idx])
            criteria = PermutedDimsArray(ArchGDAL.read(criteria_dataset, args["criteria_band"]), (2,1))
        end
        for _y=1:size(igm)[1]
            for _x=1:size(igm)[2]
                pt = igm[_y,_x,1:2]
                closest_t = Array{Int64}([round((pt[2] - grid[1,1,2]) / target_resolution[2]), 
                                        round((pt[1] - grid[1,1,1]) / target_resolution[1])  ]) .+ 1

                closest = zeros(Int64,2)
                for xbuffer in -pixel_buffer_window:pixel_buffer_window
                    for ybuffer in -pixel_buffer_window:pixel_buffer_window
                        closest[1] = closest_t[1] + xbuffer
                        closest[2] = closest_t[2] + ybuffer

                        
                        if closest[1] < 1 || closest[2] < 1 || closest[1] > size(grid)[1] || closest[2] > size(grid)[2]
                            continue
                        end
                        dist = sum((grid[closest[1],closest[2],:] - pt).^2)

                        if dist < max_offset_distance

                            if args["criteria_mode"] in ["distance", "min"]
                                if args["criteria_mode"] == "distance"
                                    current_crit = dist
                                else
                                    current_crit = criteria[closest[1], closest[2]]
                                end

                                if current_crit < best[closest[1], closest[2], 4]
                                    best[closest[1], closest[2], 1:3] = [_x, _y, file_idx]
                                    best[closest[1], closest[2], 4] = current_crit
                                end
                            elseif args["criteria_mode"] == "max"
                                current_crit = criteria[closest[1], closest[2]]
                                if current_crit > best[closest[1], closest[2], 4]
                                    best[closest[1], closest[2], 1:3] = [_x, _y, file_idx]
                                    best[closest[1], closest[2], 4] = current_crit
                                end
                            end
                        end
                    end
                end

            end
        end
    end

    println(total_found, " ", sum(best[..,1] .!= -9999), " ", size(best)[1]*size(best)[2])
    output = Array{Int32}(permutedims(best[..,1:3], (2,1,3)))
    ArchGDAL.write!(outDataset, output, [1:size(output)[end];], 0, 0, size(output)[1], size(output)[2])

end


function tap_bounds(to_tap::Float64, res::Float64, type::String)

    mult = 1
    if to_tap < 0
        mult = -1
    end
    if type == "up" && mult == 1
        adj = abs(res) - Float64(mod(abs(to_tap), abs(res)))
    elseif type == "down" && mult == 1
        adj = -1 * Float64(mod(abs(to_tap), abs(res)))
    elseif type == "up" && mult == -1
        adj = -1 * Float64(mod(abs(to_tap), abs(res)))
    elseif type == "down" && mult == -1
        adj = abs(res) - Float64(mod(abs(to_tap), abs(res)))
    else
        throw(ArgumentError("type must be one of [\"up\",\"down\""))
    end

    return mult * (abs(to_tap) + adj)
end


function get_bounding_extent_igms(file_list::Array{String}, return_per_file_xy::Bool=false)
    file_min_xy = Array{Float64}(undef,size(file_list)[1],2)
    file_max_xy = Array{Float64}(undef,size(file_list)[1],2)

    for (_f, file) in enumerate(file_list)
        println(file)
        dataset = ArchGDAL.read(file)
        igm = ArchGDAL.read(dataset)
        println(size(igm))
        file_min_xy[_f,:] = [minimum(igm[..,1]), minimum(igm[..,2])]
        file_max_xy[_f,:] = [maximum(igm[..,1]), maximum(igm[..,2])]
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


main()

