using ArchGDAL
using ArgParse2
using EllipsisNotation
using DelimitedFiles
using Logging
using Statistics
using PyCall
using Distributed
using Printf
using LinearAlgebra
using Plots
using Combinatorics
using Random


function main()

    parser = ArgumentParser(prog = "Spectral Unmixer",
                            description = "Unmix spectral in one of many ways")

    add_argument!(parser, "reflectance_file", type = String, help = "Input reflectance file")
    add_argument!(parser, "endmember_file", type = String, help = "Endmember reflectance deck")
    add_argument!(parser, "endmember_class", type = String, help = "header of column to use in endmember_file")
    add_argument!(parser, "output_file_base", type = String, help = "Output file base name")
    add_argument!(parser, "--reflectance_uncertainty_file", type = String, default = "", help = "Channelized uncertainty for reflectance input image")
    add_argument!(parser, "--reflectance_uncertainty_covariance_file", type = String, default = "", help = "Input file reference to covariance uncertainty matrix.  If used totgether with the reflectance_uncertainty_file, this should have the instrument noise (diagonals) subtracted off.")
    add_argument!(parser, "--n_mc", type = Int64, default = 1, help = "number of monte carlo runs to use, requires reflectance uncertainty file")
    add_argument!(parser, "--mode", type = String, default = "sma", help = "operating mode.  Options = [sma, mesma, plot_endmembers]")
    add_argument!(parser, "--refl_nodata", type = Float64, default = -9999.0, help = "nodata value expected in input reflectance data")
    add_argument!(parser, "--refl_scale", type = Float64, default = 1.0, help = "scale image data (divide it by) this amount")
    add_argument!(parser, "--combination_type", type = String, default = "class-even", help = "style of combinations.  Options = [all, class-even]")
    add_argument!(parser, "--max_combinations", type = Int64, default = -1, help = "set the maximum number of enmember combinations (relevant only to mesma)")
    add_argument!(parser, "--num_endmembers", type = Int64, default = [3], nargs="+", help = "set the maximum number of enmember combinations (relevant only to mesma)")
    add_argument!(parser, "--write_complete_fractions", type=Bool, default = 0, help = "flag to indicate if per-endmember fractions should be written out")
    add_argument!(parser, "--log_file", type = String, default = nothing, help = "log file to write to")
    args = parse_args(parser)

    if isnothing(args.log_file)
        logger = Logging.SimpleLogger()
    else
        logger = Logging.SimpleLogger(args.log_file)
    end
    Logging.global_logger(logger)

    pushfirst!(PyVector(pyimport("sys")."path"), "")

    bel = pyimport("build_endmember_library")
    endmember_library = bel.SpectralLibrary(args.endmember_file, args.endmember_class,
                                            string.(Array{Int}(350:2:2499)),Array{Float32}(350:2:2499),
                                            scale_factor=1.)
    endmember_library.load_data()
    endmember_library.filter_by_class()
    endmember_library.scale_library()

    refl_file_bands = bel.get_refl_wavelengths(args.reflectance_file)
    endmember_library.interpolate_library_to_new_wavelengths(refl_file_bands)

    endmember_library.remove_wavelength_region_inplace(bel.bad_wv_regions, set_as_nans=true)
    endmember_library.set_good_bands_from_nan()

    reflectance_dataset = ArchGDAL.read(args.reflectance_file)
    x_len = ArchGDAL.width(reflectance_dataset)
    y_len = ArchGDAL.height(reflectance_dataset)

    if args.reflectance_uncertainty_file != ""
        reflectance_uncertainty_dataset = ArchGDAL.read(args.reflectance_uncertainty_file)
        if ArchGDAL.width(reflectance_uncertainty_dataset) != x_len error("Reflectance_uncertainty_file size mismatch") end
        if ArchGDAL.height(reflectance_uncertainty_dataset) != y_len error("Reflectance_uncertainty_file size mismatch") end
        reflectance_uncertainty_dataset = nothing
    end


    if args.mode == "plot_mean_endmembers"
        for (_u, u) in enumerate(endmember_library.unique_classes)
            mean_spectra = mean(endmember_library.spectra[endmember_library.classes .== u,:],dims=1)[:]
            if _u == 1
                plot(endmember_library.wavelengths, mean_spectra, label=u)
            else
                plot!(endmember_library.wavelengths, mean_spectra, label=u, xlim=[300,3200])
            end
        end
        xlabel!("Wavelength [nm]")
        ylabel!("Reflectance")
        xticks!([500, 1000, 1500, 2000, 2500, 3000])
        savefig("test.png")
        exit()
    end

    if args.mode == "plot_endmembers"
        for (_u, u) in enumerate(endmember_library.unique_classes)
            if _u == 1
                plot(endmember_library.wavelengths, endmember_library.spectra[endmember_library.classes .== u,:]', lab="", xlim=[300,3200], color=palette(:tab10)[_u],dpi=400)
            else
                plot!(endmember_library.wavelengths, endmember_library.spectra[endmember_library.classes .== u,:]', lab="",xlim=[300,3200], color=palette(:tab10)[_u])
            end
        end
        xlabel!("Wavelenth [nm]")
        ylabel!("Reflectance")
        xticks!([500, 1000, 1500, 2000, 2500, 3000])
        for (_u, u) in enumerate(endmember_library.unique_classes)
            plot!([1:2],[0,0.3], color=palette(:tab10)[_u], labels=u)
        end
        savefig(string(args.output_file_base, "_endmembers.png"))
        exit()
    end

    if args.mode == "plot_endmembers_individually"
        plots = []
        spectra = endmember_library.spectra
        classes = endmember_library.classes
        for (_u, u) in enumerate(endmember_library.unique_classes)
            sp = spectra[classes .== u,:]
            sp[broadcast(isnan,sp)] .= 0
            brightness = sum(sp, dims=2)
            toplot = spectra[classes .== u,:] ./ brightness
            #push!(plots, plot(endmember_library.wavelengths, toplot', title=u, color=palette(:tab10)[_u], xlabel="Wavelength [nm]", ylabel="Reflectance"))
            push!(plots, plot(endmember_library.wavelengths, toplot', title=u, xlabel="Wavelength [nm]", ylabel="Reflectance"))
            xticks!([500, 1000, 1500, 2000, 2500])
        end
        plot(plots...,size=(1000,600),dpi=400)
        savefig(string(args.output_file_base, "_endmembers_individually.png"))
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

    outDatasets = []
    for _o in 1:length(output_bands)
        @info "Output Image Size (x,y,b): $x_len, $y_len, $output_bands.  Creating output fractional cover dataset."
        outDataset = ArchGDAL.create(output_files[_o], driver=ArchGDAL.getdriver("ENVI"), width=x_len,
                                     height=y_len, nbands=output_bands[_o], dtype=Float64)
        ArchGDAL.setproj!(outDataset, ArchGDAL.getproj(reflectance_dataset))
        try
            ArchGDAL.setgeotransform!(outDataset, ArchGDAL.getgeotransform(reflectance_dataset))
        catch e
            println("No geotransorm avaialble, proceeding without")
        end
        push!(outDatasets, outDataset)
    end


    for _o in 1:(length(outDatasets))
        for _b in 1:(output_bands[_o]-1)
            ArchGDAL.setnodatavalue!(ArchGDAL.getband(outDatasets[_o],_b), -9999)
            if _o == 1
              oc = endmember_library.unique_classes[_b]
              println("Band $_b is of class: $oc")
            end
        end
        ArchGDAL.setnodatavalue!(ArchGDAL.getband(outDatasets[_o],output_bands[_o]), -9999)
        println("Band $output_bands is of class: Shade")
    end

    good_bands = convert(Array{Bool},endmember_library.good_bands)
    spectra = convert(Array{Float64},endmember_library.spectra)
    classes = convert(Vector{String},endmember_library.classes)
    un_classes = convert(Vector{String},endmember_library.unique_classes)


    results = pmap(line->mesma_line(line,args.reflectance_file, args.mode, args.refl_nodata, args.refl_scale, good_bands,
               spectra, classes, un_classes, args.reflectance_uncertainty_file, args.n_mc,
               args.combination_type, args.num_endmembers, args.max_combinations), 1:y_len)


    # Write primary output
    output = zeros(y_len, x_len, output_bands[1]) .- 9999
    for res in results
        if isnothing(res[2]) == false
            output[res[1],res[3], :] = res[2]
        end
    end
    output = permutedims( output, (2,1,3))
    ArchGDAL.write!(outDatasets[1], output, [1:size(output)[end];], 0, 0, size(output)[1], size(output)[2])
    outDatasets[1] = nothing

    ods_idx = 2

    # Write uncertainty output
    if args.n_mc > 1
        output = zeros(y_len, x_len, output_bands[ods_idx]) .- 9999
        for res in results
            if isnothing(res[4]) == false
                output[res[1],res[3], :] = res[4]
            end
        end

        output = permutedims( output, (2,1,3))
        ArchGDAL.write!(outDatasets[2], output, [1:size(output)[end];], 0, 0, size(output)[1], size(output)[2])
        outDatasets[ods_idx]= nothing
        ods_idx += 1
    end

    # Write complete fraction output
    if args.write_complete_fractions == 1
        output = zeros(y_len, x_len, output_bands[ods_idx]) .- 9999
        for res in results
            if isnothing(res[5])
                output[res[1],res[3], :] = res[5]
            else
        end

        output = permutedims( output, (2,1,3))
        ArchGDAL.write!(outDatasets[ods_idx], output, [1:size(output)[end];], 0, 0, size(output)[1], size(output)[2])
        outDatasets[ods_idx] = nothing
        ods_idx += 1

    end

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

    function load_line(reflectance_file::String, reflectance_uncertainty_file::String, line::Int64,
                       good_bands::Array{Bool}, refl_nodata::Float64)

        reflectance_dataset = ArchGDAL.read(reflectance_file)
        img_dat = convert(Array{Float64},ArchGDAL.readraster(reflectance_file)[:,line,:])
        img_dat = img_dat[:, good_bands]
        good_data = .!all(img_dat .== refl_nodata, dims=2)[:,1]
        img_dat = img_dat[good_data,:]

        if sum(good_data) > 1
            if reflectance_uncertainty_file != ""
                unc_dat = convert(Array{Float64},ArchGDAL.readraster(reflectance_uncertainty_file)[:,line,:])
                unc_dat = unc_dat[:, good_bands]
                unc_dat = unc_dat[good_data,:]
            else
                unc_dat = nothing
            end
        else
            return nothing, nothing, good_data
        end

        return img_dat, unc_dat, good_data
    end

    function dolsq(A, b)
        x = A \ b
        #x = pinv(A)*b
        #Q,R = qr(A)
        #x = inv(R)*(Q'*b)
        return x
    end

    function mesma_line(line::Int64, reflectance_file::String, mode::String, refl_nodata::Float64,
                        refl_scale::Float64, good_bands::Array{Bool}, spectra::Array{Float64},
                        classes::Vector{String}, unique_classes::Vector{String},
                        reflectance_uncertainty_file::String = "", n_mc::Int64 = 1,
                        combination_type::String = "all", num_endmembers::Vector{Int64} = [2,3],
                        max_combinations::Int64 = -1)

        Random.seed!(13)
        println(line)
        img_dat, unc_dat, good_data = load_line(reflectance_file, reflectance_uncertainty_file, line, good_bands, refl_nodata)
        mesma_results = fill(-9999.0, sum(good_data), size(unique_classes)[1] + 1)
        if n_mc > 1
            mesma_results_std = fill(-9999.0, sum(good_data), size(unique_classes)[1] + 1)
        else
            mesma_results_std = nothing
        end

        if isnothing(img_dat)
            return line, nothing, good_data, nothing, nothing
        end
        img_dat = img_dat ./ refl_scale

        if combination_type == "class-even"
            class_idx = []
            for uc in unique_classes
                push!(class_idx, (1:size(classes)[1])[classes .== uc])
            end
        end

        # Prepare combinations if relevant
        if mode == "mesma"
            if combination_type == "class-even"
                options = collect(Iterators.product(class_idx...))[:]
            elseif combination_type == "all"
                options = []
                for num in num_endmembers
                    combo = [c for c in combinations(1:length(classes), num)]
                    push!(options,combo...)
                end
            else
                error("Invalid combiation string")
            end
        end


        # Solve complete fraction set (based on full library deck)
        complete_fractions = zeros(size(img_dat)[1], size(spectra)[1] + 1)
        complete_fractions_std = zeros(size(img_dat)[1], size(spectra)[1] + 1)
        for _i in 1:size(img_dat)[1] # Pixel loop

            mc_comp_frac = zeros(n_mc, size(spectra)[1]+1)
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
                            while selector < num_endmembers[1]

                                _p = mod(selector, length(perm_class_idx))
                                push!(perm, pop!(perm_class_idx[_p][0]))

                                if length(per_class_idx[_p]) == 0
                                    pop!(perm_class_idx[_p])
                                end
                                selector += 1

                            end

                        else
                            perm = randperm(size(spectra)[1])[1:num_endmembers[1]]
                            G = spectra[perm,:]'
                        end
                    else
                        perm = 1:size(spectra)[2]
                        G = spectra'
                    end

                    x0 = dolsq(G, d')
                    x0 = x0[:]
                    res, cost = bvls(G, d[:], x0, zeros(size(x0)), ones(size(x0)), 1e-3, 10, 1)
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
                        #G = hcat(spectra[comb,:], ones(size(spectra[comb,:])[1],1))
                        G = spectra[comb,:]'

                        x0 = dolsq(G, d')
                        if mode == "mesma_bvls"
                            ls, lc = bvls(G, d[:], x0, zeros(size(x0)), ones(size(x0)), 1e-3, 10, 1)
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
            for (_class, cl) in enumerate(unique_classes)
                mesma_results[_i, _class] = sum(complete_fractions[_i,1:end-1][cl .== classes])
            end
            mesma_results[_i, end] = complete_fractions[_i,end]

            #Aggregate uncertainty if relevant
            if n_mc > 1
                for (_class, cl) in enumerate(unique_classes)
                    mesma_results_std[_i, _class] = std(sum(mc_comp_frac[:,1:end-1][:,cl .== classes], dims=2))
                end
                mesma_results_std[_i, end] = std(mc_comp_frac[:,end])
            end

        end

        return line, mesma_results, good_data, mesma_results_std, complete_fractions

    end

    function bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)
        n_iter = 0
        m, n = size(A)

        x = x_lsq
        on_bound = zeros(n)

        mask = x .<= lb
        x[mask] = lb[mask]
        on_bound[mask] .= -1

        mask = x .>= ub
        x[mask] = ub[mask]
        on_bound[mask] .= 1

        free_set = on_bound .== 0
        active_set = .!free_set
        free_set = (1:size(free_set)[1])[free_set .!= 0]

        r = A * x - b
        cost = 0.5 * dot(r , r)
        initial_cost = cost
        g = A' * r

        cost_change = nothing
        step_norm = nothing
        iteration = 0

        while size(free_set)[1] > 0
            if verbose == 2
                optimality = compute_kkt_optimality(g, on_bound)
            end

            iteration += 1
            x_free_old = x[free_set]

            A_free = A[:, free_set]
            b_free = b - A * (x .* active_set)
            z = dolsq(A_free, b_free)

            lbv = z .< lb[free_set]
            ubv = z .> ub[free_set]

            v = lbv .| ubv

            if any(lbv)
                ind = free_set[lbv]
                x[ind] = lb[ind]
                active_set[ind] .= true
                on_bound[ind] .= -1
            end

            if any(ubv)
                ind = free_set[ubv]
                x[ind] = ub[ind]
                active_set[ind] .= true
                on_bound[ind] .= 1
            end

            ind = free_set[.!v]
            x[ind] = z[.!v]

            r = A * x -b
            cost_new = 0.5 * dot(r , r)
            cost_change = cost - cost_new
            cost = cost_new
            g = A' * r
            step_norm = sum((x[free_set] .- x_free_old).^2)

            if any(v)
                free_set = free_set[.!v]
            else
                break
            end
        end

        if isnothing(max_iter)
            max_iter = n
        end
        max_iter += iteration

        termination_status = nothing

        optimality = compute_kkt_optimality(g, on_bound)
        for iteration in iteration:max_iter
            if optimality < tol
                termination_status = 1
            end

            if !isnothing(termination_status)
                break
            end

            move_to_free = argmax(g .* on_bound)
            on_bound[move_to_free] = 0

            x_free = copy(x)
            x_free_old = copy(x)
            while true

                free_set = on_bound .== 0
                sum(free_set)
                active_set = .!free_set
                free_set = (1:size(free_set)[1])[free_set .!= 0]

                x_free = x[free_set]
                x_free_old = copy(x_free)
                lb_free = lb[free_set]
                ub_free = ub[free_set]

                A_free = A[:, free_set]
                b_free = b - A * (x .* active_set)
                z = dolsq(A_free, b_free)

                lbv = (1:size(free_set)[1])[ z .< lb_free]
                ubv = (1:size(free_set)[1])[ z .> ub_free]
                v = cat(lbv, ubv, dims=1)

                if size(v)[1] > 0
                    alphas = cat(lb_free[lbv] - x_free[lbv], ub_free[ubv] - x_free[ubv],dims=1) ./ (z[v] - x_free[v])

                    i = argmin(alphas)
                    i_free = v[i]
                    alpha = alphas[i]

                    x_free .*= (1 .- alpha)
                    x_free .+= (alpha .* z)
                    x[free_set] = x_free

                    vsize = size(lbv)
                    if i <= size(lbv)[1]
                        on_bound[free_set[i_free]] = -1
                    else
                        on_bound[free_set[i_free]] = 1
                    end
                else
                    x_free = z
                    x[free_set] = x_free
                    @goto start
                end
            end #while
            @label start
            step_norm = sum((x_free .- x_free_old).^2)

            r = A * x - b
            cost_new = 0.5 * dot(r , r)
            cost_change = cost - cost_new

            combo = tol * cost
            if cost_change < tol * cost
                termination_status = 2
            end
            cost = cost_new

            g = A' * r
            optimality = compute_kkt_optimality(g, on_bound)
        end #iteration

        if isnothing(termination_status)
            termination_status = 0
        end

        x[x .< 1e-5] .= 0
        return x, cost
    end

    function compute_kkt_optimality(g, on_bound)
      g_kkt = g .* on_bound
      free_set = on_bound .== 0
      g_kkt[free_set] = broadcast(abs, g[free_set])

      return maximum(g_kkt)

    end
end

main()
