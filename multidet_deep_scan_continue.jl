import MPI
import Healpix
import Random
import CorrNoise
import Stripeline
import Dates
const Sl = Stripeline

import Statistics
using Printf




MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
commsize = MPI.Comm_size(comm)

printmsg(msg) = (rank == 0) && print(msg)

printmsg(@sprintf("""MPI parameters:

- Rank: %d
- Number of MPI processes: %d

""", rank, commsize))


num_of_polarimeters = 3
fsamp_hz = 50
NSIDE = 256
requested_time_days = 30

hpx_badval    = -1.6375e30
sidereal_day_s = 86164.0905
iseed          = 201337

printmsg(@sprintf(
    """Parameters of the simulation:


- Number of polarimeters: %d
- Sampling frequency: %.1f Hz
- Requested days: %d 

""",

    num_of_polarimeters,
    fsamp_hz,
    requested_time_days,
))




#input map file. must be an IQU map, but only Q,U fields are actually used
sky_map = "/home/users/giorgia.caruso1.stud/cmbgroup/users/giorgia.caruso/PySM_inputmap_nside256.fits"



#Choose simulation/mapmaker type:
# "Q" : demodulated Q tod
# "U" : demodulated U tod
# "sum"  : demodulated (Q+U) tod
# "diff" : demodulated (Q-U) tod
tod_mode = "Q"

#sim ID number.
#due to MPI memory leakages, generate sims one at the time but change random
#seed based on sim ID to ensure different sims correspond to different
#realizations.
isim    = 0
sim_num = string(isim,base=10,pad=5)

#we set horn ID and the pull all detectors parameters from the database
db = Sl.InstrumentDB()

horn_id = ["O1", "R2", "V4"] 
println("using polarimeters: ", horn_id)

hrn     = [db.focalplane[hid] for hid in horn_id]
horient = [h.orientation for h in hrn]
pol_id  = [h.polid for h in hrn]
det     = [Sl.detector(db, pid) for pid in pol_id]

fknee_hz_q = zeros(num_of_polarimeters)
fknee_hz_u = zeros(num_of_polarimeters)
alpha_q    = zeros(num_of_polarimeters)
alpha_u    = zeros(num_of_polarimeters)
tnoise_k  = zeros(num_of_polarimeters)


for i in 1:num_of_polarimeters
    d = det[i]


#Q branch
    if d.spectrum.fknee_q_hz > 0
        fknee_hz_q[i] = d.spectrum.fknee_q_hz
    else
        fknee_hz_q[i] = 0.06621791867490712
    end

    if d.spectrum.slope_q > 0
        alpha_q[i] = d.spectrum.slope_q
    else
        alpha_q[i] = 1.0
    end
#U branch

    if d.spectrum.fknee_u_hz > 0
        fknee_hz_u[i] = d.spectrum.fknee_u_hz
    else
        fknee_hz_u[i] = 0.06621791867490712
    end
    if d.spectrum.slope_u > 0
        alpha_u[i] = d.spectrum.slope_u
    else
        alpha_u[i] = 1.0
    end

    if d.tnoise.tnoise_k > 0
        tnoise_k[i] = d.tnoise.tnoise_k
    else
        tnoise_k[i] = 35.0
    end

end
println("tnoise_k: ", tnoise_k)
fknee_hz = (fknee_hz_q .+ fknee_hz_u) ./ 2
fknee_hz_q .= fknee_hz
fknee_hz_u .= fknee_hz
baseline_length_s = 1.0 ./(2.0 .* fknee_hz) #decimale
baseline_samples = round.(Int, baseline_length_s .* fsamp_hz) #intero per il round
println("baseline samples: ", baseline_samples)
requested_time_s = requested_time_days * 24 * 3600 #decimale float
requested_samples = round(Int, requested_time_s * fsamp_hz) #intero 
mission_duration_samples = requested_samples #intero
println("Mission duration samples: $mission_duration_samples")
alpha   = (alpha_q .+ alpha_u) ./2
alpha_q .= alpha
alpha_u .= alpha

tcmb_k = 2.7255
tatm_k = 15
ttel_k = 3
#tnoise_k = 35
β_hz = 7e9

fknee_tag = join((@sprintf("%04.3f", x) for x in fknee_hz), "-")
bline_tag = join([@sprintf("%04.3f", bl) for bl in baseline_length_s], "-")
horns_tag = join(horn_id, "-")
out_map_root = "/home/users/giorgia.caruso1.stud/cmbgroup/users/giorgia.caruso/sim_" * tod_mode *
               "_d" * lpad(requested_time_days, 3, '0') *
               "_" * sim_num * "_O1R2V4_test_array_prealloc_GC_random_indep_strategy_centre_continue_azdec"



#tsys_k = tnoise_k + tatm_k + ttel_k + tcmb_k
tsys_k = tnoise_k .+ tatm_k .+ ttel_k .+ tcmb_k
println("tsys_k: ", tsys_k)
τ_s = 1 / fsamp_hz
σ_k = (tsys_k ./ sqrt(β_hz * τ_s))


total_samples  = mission_duration_samples * num_of_polarimeters #intero
println("total samples: ", total_samples)

samples_per_process = Sl.split_into_n(total_samples, commsize) 

chunks = Sl.split_tod_mpi(
    mission_duration_samples,
    baseline_samples,
    samples_per_process,
    commsize
 )

 

#chunks = Sl.split_tod_mpi(
#    mission_duration_samples,
#    baseline_samples,
#    samples_per_process,
#    commsize
#)



this_rank_chunk = chunks[rank + 1]

(detector_number, first_time, last_time, num_of_baselines, num_of_samples) = Sl.get_chunk_properties(chunks, baseline_samples, fsamp_hz, rank)




printmsg("Reading map \"$(sky_map)\"\n")  
inputmap_q = Healpix.readMapFromFITS(
    sky_map,
    2,
    Float64,
)
inputmap_u = Healpix.readMapFromFITS(
    sky_map,
    3,
    Float64,
)
inputmap_resol = inputmap_q.resolution

resol = Healpix.Resolution(NSIDE)
order = collect(typeof(inputmap_q).parameters)[2]

num_of_pixels = resol.numOfPixels


printmsg("Generating the noise\n")

stokes = 2
tmp_seed_q = iseed .+ pol_id .* 999331 .+ isim .* 1867 .+ stokes .* 307
seed_vec_q = UInt32.(tmp_seed_q)

println("fknee_hz_q = ", fknee_hz_q)
println("alpha_q = ", alpha_q)
println("sigma_k = ", σ_k)
println("baseline_samples = ", baseline_samples)
println("samples_per_process = ", samples_per_process)
@assert sum(samples_per_process) == mission_duration_samples * num_of_polarimeters
noise_tod_q = Sl.generate_noise_mpi(       
    chunks,
    samples_per_process,
    baseline_samples,
    mission_duration_samples,
    fsamp_hz,
    σ_k,
    fknee_hz_q,
    alpha_q,
    rank = rank,
    comm = comm,
    input_seed = seed_vec_q,
)

stokes = 3
tmp_seed_u = iseed .* 19 .+ pol_id .* 999331 .+ isim .* 1867 .+ stokes .* 307
seed_vec_u = UInt32.(tmp_seed_u)
noise_tod_u = Sl.generate_noise_mpi(
    chunks,
    samples_per_process,
    baseline_samples,
    mission_duration_samples,
    fsamp_hz,
    σ_k,
    fknee_hz_u,
    alpha_u,
    rank = rank,
    comm = comm,
    input_seed = seed_vec_u,
)

lmode = lowercase(tod_mode)
if (lmode == lowercase("Q"))
    noise_tod = noise_tod_q
elseif (lmode == lowercase("U"))
    noise_tod = noise_tod_u
elseif (lmode == lowercase("sum"))
    noise_tod = noise_tod_q .+ noise_tod_u
elseif (lmode == lowercase("diff"))
    noise_tod = noise_tod_q .- noise_tod_u
else
    noise_tod = NaN
end

noise_tod_q = nothing
noise_tod_u = nothing

GC.gc()


printmsg("Projecting the map onto the TOD\n")

pix_idx = Int32[]
tod     = Float64[]
twopsi  = Float64[]

cumsamps = cumsum(num_of_samples)



for i in 1:length(this_rank_chunk)
    local ns = num_of_samples[i]


    if i == 1
        start_s = 1
    else
        start_s = cumsamps[i-1] + 1
    end


    end_s = cumsamps[i]

    noise_chunk = noise_tod[start_s:end_s]
    local times = first_time[i] .+ (0:(ns-1)) .* τ_s
    local polarid = detector_number[i]

    local (dirs, psi) = Sl.genpointings(
    horient[polarid],
    times,
    day_duration_s = 86164.0905,
    latitude_deg = 28.30026
    ) do time_s

  #Teide
        local lat_deg = 28.30026
        local lon_deg = -16.5100
        local lat = deg2rad(lat_deg)
        local lon = deg2rad(lon_deg)

#centro patch scelto (coord. equator.)
        local dec_center = deg2rad(75.0)  #verifico tracking per sopra orizzonte e notazione latitudine/colatitudine (Healpix usa colat.)
        local ra_center  = deg2rad(0.0)

#parametri movimento "azimuth" (direz. orizz.) rispetto al centro 
        local delta_xi_deg = 5.0
        local delta_xi = deg2rad(delta_xi_deg)
        local Txi = 60.0

#param. movimento "declin". (vert.) rispetto al centro (T = 8 h)

        local eta_step_deg = 0.5
        local max_eta_deg = 1.0
        local eta_block_time = 3600.0
    
#funzione azimuth triangolare, normalizzata tra -1 e +1 (continua: ok?)
        local function triwave(t, period)
            local p = mod(t, period) / period
            return 1.0 - 4.0 * abs(p - 0.5)
        end
        
#funzione declin. (a gradini)
        local function stepwave(t, step_time, step_deg, max_offset_deg)
            local nlev = Int(round(2 * max_offset_deg / step_deg)) + 1
            local k = floor(Int, t / step_time)
            local m = mod(k, 2 * (nlev - 1))
            local idx = m < nlev ? m : 2 * (nlev - 1) - m
            return -max_offset_deg + idx * step_deg
        end

#applicazione parametri azi - decl a funzioni rispetto al centro -> movimenti su direzioni "piatte" rispetto al centro
        local xi = tan(delta_xi * triwave(time_s, Txi))
        local eta = tan(deg2rad(stepwave(time_s, eta_block_time, eta_step_deg, max_eta_deg)))

#mappa movimenti sulla sfera -> proiezione 

        local sin_dec0 = sin(dec_center)
        local cos_dec0 = cos(dec_center)

        local denom = cos_dec0 - eta * sin_dec0  # termine correzione per curva sferica 
        local ra = ra_center + atan(xi, denom)
        local dec = atan(
            sin_dec0 + eta * cos_dec0,
            sqrt(denom^2 + xi^2)
        )

        ra = mod(ra, 2π)

#conversione in sistema locale telescopio 
        local omega = 2π / 86164.0905          #vel. ang. rotazione Terra
        local lst = omega * time_s + lon       # local sidereal time 

        local ha = mod(lst - ra + π, 2π) - π   # hour angle (lst - right asc.) tra -pi, +pi


        local sin_alt = sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(ha)    #conversione in alt
        local alt = asin(clamp(sin_alt, -1.0, 1.0))

        local y = -sin(ha) * cos(dec)                                         #conversione in azim
        local x =  sin(dec) * cos(lat) - cos(dec) * sin(lat) * cos(ha)
        local az = mod(atan(y, x), 2π)


        local zenith_angle = (π/2) - alt        #per convenzione genpointings (o no?)

        return (0.0, zenith_angle, az)

    end

    local partial_pix_idx = Healpix.ang2pixRing.(
        Ref(inputmap_resol),
        dirs[:, 1],
        dirs[:, 2],
    )
    local two_psi = -2 .* psi
    global twopsi = append!(twopsi, two_psi)
    local wts = Sl.get_QU_weights.(two_psi, tod_mode = tod_mode)
    local w8s = hcat([wt[1] for wt in wts], [wt[2] for wt in wts])
    local  noise_chunk = noise_tod[start_s : end_s]
        

    local partial_sky_tod = inputmap_q.pixels[partial_pix_idx] .* w8s[:,1] .+ inputmap_u.pixels[partial_pix_idx] .* w8s[:,2] .+noise_chunk
    
    global tod = append!(tod, partial_sky_tod)

    local partial_pix_idx = Healpix.ang2pixRing.(
        Ref(resol),
        dirs[:, 1],
        dirs[:, 2],
    )

    global pix_idx = append!(pix_idx, partial_pix_idx)



end


hits_pol_local = zeros(Int64, num_of_pixels)


for p in pix_idx
    hits_pol_local[p] += 1
end


hits_pol_global = MPI.Allreduce(hits_pol_local, +, comm)

if rank == 0
    hitmap_pol = Healpix.HealpixMap{Float64, Healpix.RingOrder}(NSIDE)
    hitmap_pol.pixels .= Float64.(hits_pol_global)

    fname = out_map_root * "_hits_QU.fits"
    Healpix.saveToFITS(hitmap_pol, fname, typechar="D")
    printmsg("Saved hit-count map: $fname\n")
end

#for now, assume uncorrelated noise between Q and U tod, so just sum in quadrature
σ0 = tsys_k ./ sqrt(β_hz * τ_s)
printmsg("input noise rms = $(σ0)\n")

total_samples_this_rank = sum(num_of_samples)
@assert total_samples_this_rank == samples_per_process[rank+1]

if (lmode == lowercase("Q") || lmode == lowercase("U"))
    σ_k_samp = vcat([ fill(σ_k[detector_number[i]], num_of_samples[i]) for i in 1:length(num_of_samples) ]...)
    @assert length(σ_k_samp) == total_samples_this_rank
    #σ_k = σ_k .* ones(Float64, total_samples_this_rank)
elseif (lmode == lowercase("sum") || lmode == lowercase("diff"))
    σ_k_samp = vcat([ fill(σ_k[detector_number[i]]*sqrt(2), num_of_samples[i]) for i in 1:length(num_of_samples) ]...)
    @assert length(σ_k_samp) == total_samples_this_rank
    #σ_k = σ_k*sqrt(2) .* ones(Float64, total_samples_this_rank)
else
    noise_tod = NaN
end

printmsg(@sprintf("average rms %f \n",Statistics.mean(σ_k)))

inputmap_q  = nothing
inputmap_u  = nothing
noise_tod   = nothing

GC.gc()




printmsg("Running the destriper\n")

function callback(
    ;
    iter_idx,
    max_iter,
    convergence_parameter,
    convergence_threshold,
)
    @printf("%d/%d, %e > %e\r", iter_idx, max_iter, convergence_parameter, convergence_threshold)
end
printmsg("================================\n")


num_of_baselines = round.(Int, num_of_baselines)
num_of_samples = round.(Int, num_of_samples)
#array con N elementi, uno per chunk appartenente a quel rank, con numero di campioni di ogni chunk (num of elements del chunk)
detector_list = detector_number
sigma_scalar = tsys_k ./ sqrt(β_hz * τ_s)


rms_list = [
      fill(sigma_scalar[detector_list[i]], num_of_samples[i])  #num of samples[i] = num of elements chunk i]
      for i in 1:length(num_of_samples) #lunghezza pari al n. di chunk del processo corrente
]



data_properties = Sl.build_noise_properties(detector_list, rms_list, num_of_baselines, num_of_samples, baseline_samples)

cond_results = Sl.condnumber_mpi(
        pix_idx, num_of_pixels, twopsi, data_properties;
        comm     = comm,
        tod_mode = tod_mode,
        unseen   = hpx_badval
    )
    condnum = cond_results[4, :]
    obscond = condnum[condnum .!= hpx_badval]
    medium  = Statistics.mean(obscond)
    med     = Statistics.median(obscond)
    dev     = Statistics.std(obscond)

    printmsg(@sprintf("""
Statistiche globali Condition Number (%d horn):
 - Media:      %e
 - Mediana:    %e
 - Deviazione: %e

""", num_of_polarimeters, medium, med, dev))

    if rank == 0
      let fname = out_map_root * "_cond_multi_samp_complete.fits"
        printmsg("Saving multi-horn condmap to \"$fname\"\n")
        condmap = Healpix.HealpixMap{Float64,Healpix.RingOrder}(NSIDE)
        condmap.pixels = condnum
        Healpix.saveToFITS(condmap, fname, typechar="D")
      end
    end


data_properties = Sl.build_noise_properties(detector_list, rms_list, num_of_baselines, num_of_samples, baseline_samples)

rnr_buffer = Sl.binned_noise_variance_mpi(
    pix_idx,
    num_of_pixels,
    twopsi,
    data_properties;
    comm     = comm,
    unseen   = hpx_badval,
    tod_mode = tod_mode
)


σ_pix = fill(hpx_badval, 2, num_of_pixels)
valid = (rnr_buffer[1, :] .> 0) .& (rnr_buffer[3, :] .> 0)
deter = rnr_buffer[1, valid] .* rnr_buffer[3, valid] .- rnr_buffer[2, valid].^2
d = deter .> 0
ok = findall(valid)[d]
σ_pix[1, ok] .= sqrt.(rnr_buffer[3, ok] ./ deter[d])
σ_pix[2, ok] .= sqrt.(rnr_buffer[1, ok] ./ deter[d])

if rank == 0

    for (i, suffix) in enumerate(("_Q_white_rms_samp_complete.fits", "_U_white_rms_samp_complete.fits"))
        let m = Healpix.HealpixMap{Float64, Healpix.RingOrder}(NSIDE)
            m.pixels = σ_pix[i, :]
            Healpix.saveToFITS(m, out_map_root * suffix, typechar="D")
            printmsg("Saved white-noise RMS map: $(out_map_root*suffix)\n")
        end
    end


   # vals = σ_pix[.!isnan.(σ_pix)] 
   vals = σ_pix[σ_pix .!= hpx_badval]
    μ_rms = Statistics.mean(vals)
    σ_rms = Statistics.std(vals)
    @printf("Pixel white-noise RMS (osservati): mean = %e, std = %e\n", μ_rms, σ_rms)
end

MPI.Barrier(comm)

printmsg("Destriping the TOD\n")
results = Sl.destripe(pix_idx, tod, num_of_pixels, twopsi, data_properties, rank, comm = comm ,unseen = hpx_badval ,max_iter = 10000 ,tod_mode = tod_mode,threshold=1.e-10, callback=callback)
printmsg(@sprintf("Reached %e in %d iterations \n",last(results.convergence_param_list),results.best_iteration))

if rank == 0
    out_map_name =  out_map_root*"_Q_multi_samp_complete.fits"
    printmsg("Saving the map in \"$(out_map_name)\"\n")
    mapfile = Healpix.HealpixMap{Float64, Healpix.RingOrder}(NSIDE)
    mapfile.pixels = results.best_sky_map[1,:]
    Healpix.saveToFITS(mapfile, out_map_name, typechar = "D")

    out_map_name =  out_map_root*"_U_multi_samp_complete.fits"
    printmsg("Saving the map in \"$(out_map_name)\"\n")
    mapfile = Healpix.HealpixMap{Float64, Healpix.RingOrder}(NSIDE)
    mapfile.pixels = results.best_sky_map[2,:]
    Healpix.saveToFITS(mapfile, out_map_name, typechar = "D")
end

MPI.Finalize()