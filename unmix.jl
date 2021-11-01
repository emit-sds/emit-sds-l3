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
include("src/endmember_library.jl")
include("src/datasets.jl")


function main()

    parser = ArgumentParser(prog = "Spectral Unmixer",
                            description = "Unmix spectral in one of many ways")

    add_argument!(parser, "reflectance_file", type = String, help = "Input reflectance file")
    add_argument!(parser, "endmember_file", type = String, help = "Endmember reflectance deck")
    add_argument!(parser, "endmember_class", type = String, help = "header of column to use in endmember_file")
    add_argument!(parser, "output_file_base", type = String, help = "Output file base name")
    add_argument!(parser, "--spectral_starting_column", type = Int64, default = 2, help = "Column of library file that spectral information starts on")
    add_argument!(parser, "--reflectance_uncertainty_file", type = String, default = "", help = "Channelized uncertainty for reflectance input image")
    add_argument!(parser, "--n_mc", type = Int64, default = 1, help = "number of monte carlo runs to use, requires reflectance uncertainty file")
    add_argument!(parser, "--mode", type = String, default = "sma", help = "operating mode.  Options = [sma, mesma, plots]")
    add_argument!(parser, "--refl_nodata", type = Float64, default = -9999.0, help = "nodata value expected in input reflectance data")
    add_argument!(parser, "--refl_scale", type = Float64, default = 1.0, help = "scale image data (divide it by) this amount")
    add_argument!(parser, "--normalization", type = String, default = "none", help = "flag to indicate the scaling type. Options = [none, brightness, specific wavelength]")
    add_argument!(parser, "--combination_type", type = String, default = "class-even", help = "style of combinations.  Options = [all, class-even]")
    add_argument!(parser, "--max_combinations", type = Int64, default = -1, help = "set the maximum number of enmember combinations (relevant only to mesma)")
    add_argument!(parser, "--num_endmembers", type = Int64, default = [3], nargs="+", help = "set the maximum number of enmember to use")
    add_argument!(parser, "--write_complete_fractions", type=Bool, default = 0, help = "flag to indicate if per-endmember fractions should be written out")
    add_argument!(parser, "--log_file", type = String, default = nothing, help = "log file to write to")
    args = parse_args(parser)

    if isnothing(args.log_file)
        logger = Logging.SimpleLogger()
    else
        logger = Logging.SimpleLogger(open(args.log_file, "w+"))
    end
    Logging.global_logger(logger)

    endmember_library = SpectralLibrary(args.endmember_file, args.endmember_class, args.spectral_starting_column, nothing, 1., [])
    load_data!(endmember_library)
    filter_by_class!(endmember_library)

    refl_file_wl = read_envi_wavelengths(args.reflectance_file)
    interpolate_library_to_new_wavelengths!(endmember_library, refl_file_wl)

    remove_wavelength_region_inplace!(endmember_library, true)

    reflectance_dataset = ArchGDAL.read(args.reflectance_file)
    x_len = ArchGDAL.width(reflectance_dataset)
    y_len = ArchGDAL.height(reflectance_dataset)

    if args.reflectance_uncertainty_file != ""
        reflectance_uncertainty_dataset = ArchGDAL.read(args.reflectance_uncertainty_file)
        if ArchGDAL.width(reflectance_uncertainty_dataset) != x_len error("Reflectance_uncertainty_file size mismatch") end
        if ArchGDAL.height(reflectance_uncertainty_dataset) != y_len error("Reflectance_uncertainty_file size mismatch") end
        reflectance_uncertainty_dataset = nothing
    end


    if args.mode == "plots"
        plot_mean_endmembers(endmember_library, string(args.output_file_base, "_mean_endmembers.png"))
        plot_endmembers(endmember_library, string(args.output_file_base, "_endmembers.png"))
        plot_endmembers_individually(string(args.output_file_base, "_endmembers_individually.png"))
        exit()
    end


    endmember_library.spectra = endmember_library.spectra[:,endmember_library.good_bands]

    n_classes = length(unique(endmember_library.classes))
    output_bands = [n_classes + 1]
    output_files = [string(args.output_file_base , "_fractional_cover")]

    if args.n_mc > 1
        push!(output_bands, n_classes + 1)
        push!(output_files,string(args.output_file_base , "_fractional_cover_uncertainty") )
    end

    if args.write_complete_fractions == 1
        push!(output_bands, size(endmember_library.spectra)[1] + 1)
        push!(output_files,string(args.output_file_base , "_complete_fractions") )
    end


    output_band_names = copy(endmember_library.class_valid_keys)
    push!(output_band_names, "Brightness")

    initiate_output_datasets(output_files, x_len, y_len, output_bands, reflectance_dataset)
    set_band_names(output_files[1], output_band_names)

    results = pmap(line->mesma_line(line,args.reflectance_file, args.mode, args.refl_nodata,
               args.refl_scale, args.normalization, endmember_library,
               args.reflectance_uncertainty_file, args.n_mc,
               args.combination_type, args.num_endmembers, args.max_combinations), 1:y_len)


    write_results(output_files, output_bands, x_len, y_len, results, args)

end




@everywhere begin
    using ArchGDAL
    using EllipsisNotation
    using DelimitedFiles
    using Logging
    using Statistics
    using PyCall
    using Distributed
    using Printf
    using LinearAlgebra
    using Combinatorics
    using Random
    include("src/solvers.jl")
    include("src/endmember_library.jl")
    include("src/datasets.jl")

    function wl_index(wavelengths::Array{Float64}, target)
        argmin(abs.(wavelengths .- target))
    end

    function scale_data(refl::Array{Float64}, wavelengths::Array{Float64}, criteria::String)

        if criteria == "none"
            return refl
        elseif criteria == "brightness"
            bad_regions_wl = [[1300,1500],[1800,2000]]
            good_bands = convert(Array{Bool}, ones(length(wavelengths)))
            for br in bad_regions_wl
                good_bands[wl_index(wavelengths, br[1]):wl_index(wavelengths, br[2])] .= false
            end
            norm = sqrt.(mean(refl[:,good_bands].^2, dims=2))
            return refl ./ norm
        elseif criteria == "brightness-vtir"
            split = 2.6
            good_bands = convert(Array{Bool}, zeros(length(wavelengths)))
            good_bands[wavelengths .<= split] .= true
            norm = sqrt.(mean(refl[:,good_bands].^2, dims=2))
            refl[:,good_bands] = refl[:, good_bands] ./ norm

            good_bands = convert(Array{Bool}, zeros(length(wavelengths)))
            good_bands[wavelengths .> split] .= true
            norm = sqrt.(mean(refl[:,good_bands].^2, dims=2))
            refl[:,good_bands]= refl[:, good_bands] ./ norm

            return refl
        else
            try
                target_wl = parse(Float64,criteria)
                norm = 0.5
                return refl ./ norm
            catch e
                throw(ArgumentError(string("normalization must be [none, brightness, or a specific wavelength].  Provided:", criteria)))
            end
        end

    end

    function mesma_line(line::Int64, reflectance_file::String, mode::String, refl_nodata::Float64,
                        refl_scale::Float64, normalization::String, library::SpectralLibrary,
                        reflectance_uncertainty_file::String = "", n_mc::Int64 = 1,
                        combination_type::String = "all", num_endmembers::Vector{Int64} = [2,3],
                        max_combinations::Int64 = -1)

        Random.seed!(13)
        println(line)
        img_dat, unc_dat, good_data = load_line(reflectance_file, reflectance_uncertainty_file, line, library.good_bands, refl_nodata)
        mesma_results = fill(-9999.0, sum(good_data), size(library.class_valid_keys)[1] + 1)
        if n_mc > 1
            mesma_results_std = fill(-9999.0, sum(good_data), size(library.class_valid_keys)[1] + 1)
        else
            mesma_results_std = nothing
        end

        if isnothing(img_dat)
            return line, nothing, good_data, nothing, nothing
        end
        scale_data(img_dat, library.wavelengths[library.good_bands], normalization)
        img_dat = img_dat ./ refl_scale

        if combination_type == "class-even"
            class_idx = []
            for uc in library.class_valid_keys
                push!(class_idx, (1:size(library.classes)[1])[library.classes .== uc])
            end
        end

        # Prepare combinations if relevant
        if mode == "mesma"
            if combination_type == "class-even"
                options = collect(Iterators.product(class_idx...))[:]
            elseif combination_type == "all"
                options = []
                for num in num_endmembers
                    combo = [c for c in combinations(1:length(library.classes), num)]
                    push!(options,combo...)
                end
            else
                error("Invalid combiation string")
            end
        end


        # Solve complete fraction set (based on full library deck)
        complete_fractions = zeros(size(img_dat)[1], size(library.spectra)[1] + 1)
        complete_fractions_std = zeros(size(img_dat)[1], size(library.spectra)[1] + 1)
        for _i in 1:size(img_dat)[1] # Pixel loop

            mc_comp_frac = zeros(n_mc, size(library.spectra)[1]+1)
            for mc in 1:n_mc #monte carlo loop
                Random.seed!(mc)

                d = img_dat[_i:_i,:]
                if isnothing(unc_dat) == false
                    d += (rand(size(d)) .* 2 .- 1) .* unc_dat[_i:_i,:]
                end


                if mode == "sma"
                    if num_endmembers[1] != -1
                        if combination_type == "class-even"

                            perm_class_idx = []
                            for class_subset in class_idx
                                push!(perm_class_idx, Random.shuffle(class_subset))
                            end

                            perm = []
                            selector = 1
                            while selector <= num_endmembers[1]
                                _p = mod(selector, length(perm_class_idx)) + 1
                                push!(perm, perm_class_idx[_p][1])
                                deleteat!(perm_class_idx[_p],1)

                                if length(perm_class_idx[_p]) == 0
                                    deleteat!(perm_class_idx,_p)
                                end
                                selector += 1
                            end

                        else
                            perm = randperm(size(library.spectra)[1])[1:num_endmembers[1]]
                        end

                        G = library.spectra[perm,:]
                    else
                        perm = convert(Vector{Int64},1:size(library.spectra)[1])
                        G = library.spectra
                    end

                    G = scale_data(G, library.wavelengths[library.good_bands], normalization)'

                    x0 = dolsq(G, d')
                    x0 = x0[:]
                    res, cost = bvls(G, d[:], x0, zeros(size(x0)), ones(size(x0)), 1e-3, 100, 1)
                    #res, cost = opt_solve(G, d[:], x0, 0, 1 )
                    #res = x0
                    mc_comp_frac[mc, perm] = res

                elseif occursin("mesma", mode)
                    solutions = []
                    costs = zeros(size(options)[1]).+1e12

                    if max_combinations != -1 && length(options) > max_combinations
                        perm = randperm(length(options))[1:max_combinations]
                    else
                        perm = 1:length(options)
                    end

                    for (_comb, comb) in enumerate(options[perm])
                        comb = [c for c in comb]
                        #G = hcat(library.spectra[comb,:], ones(size(library.spectra[comb,:])[1],1))
                        G = scale_data(library.spectra[comb,:], library.wavelengths[library.good_bands], normalization)'

                        x0 = dolsq(G, d')
                        if mode == "mesma_bvls"
                            ls, lc = bvls(G, d[:], x0, zeros(size(x0)), ones(size(x0)), 1e-3, 10, 1)
                            #ls, lc = opt_solve(G, d[:], x0, 0, 1)
                            costs[_comb] = lc
                        else
                            ls = x0
                            r = G * x0 - d[:]
                            costs[_comb] = dot(r,r)
                        end

                        push!(solutions,ls)

                    end
                    best = argmin(costs)

                    mc_comp_frac[mc, [ind for ind in options[perm][best]]] = solutions[best]
                else
                    error("Invalid mode provided")
                end
            end

            # Calculate the sum of values (inverse of shade), and then normalize
            mc_comp_frac[mc_comp_frac .< 0] .= 0
            mc_comp_frac[:,end] = sum(mc_comp_frac,dims=2)
            mc_comp_frac[:,1:end-1] = mc_comp_frac[:,1:end-1] ./ mc_comp_frac[:,end]

            #
            complete_fractions[_i,:] = mean(mc_comp_frac,dims=1)
            complete_fractions_std[_i,:] = std(mc_comp_frac,dims=1)

            # Aggregate results from per-library to per-unique-class
            for (_class, cl) in enumerate(library.class_valid_keys)
                mesma_results[_i, _class] = sum(complete_fractions[_i,1:end-1][cl .== library.classes])
            end
            mesma_results[_i, end] = complete_fractions[_i,end]

            #Aggregate uncertainty if relevant
            if n_mc > 1
                for (_class, cl) in enumerate(library.class_valid_keys)
                    mesma_results_std[_i, _class] = std(sum(mc_comp_frac[:,1:end-1][:,cl .== library.classes], dims=2))
                end
                mesma_results_std[_i, end] = std(mc_comp_frac[:,end])
            end

        end

        return line, mesma_results, good_data, mesma_results_std, complete_fractions

    end

end

main()
