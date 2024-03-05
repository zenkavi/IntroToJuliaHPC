using ADDM
using Base.Threads
using BenchmarkTools
using FLoops

#########################
# Define functions
#########################

function compute_trials_nll(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
                            return_trial_likelihoods = false)
  
  likelihoods = [likelihood_fn(;model = model, trial = OneTrial, likelihood_args...) for OneTrial in data]

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = max.(likelihoods, 1e-64)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(likelihoods))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end

end


function compute_trials_nll_threads(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  likelihoods = Float64[];

  # Are the likelihoods returned in the same order? No! 
  # This might require rethinking of how to compute posteriors.
  # Maybe likelihoods should have a different structure that contains info on the trial number
  @threads for OneTrial in data
    cur_lik = likelihood_fn(;model = model, trial = OneTrial, likelihood_args...)
    push!(likelihoods, cur_lik)
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = max.(likelihoods, 1e-64)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(likelihoods))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end
end

function compute_trials_nll_threads_dict(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  likelihoods = Dict{Int, Float64}()
  data_dict = Dict(zip(1:length(data), data))

  if sequential_model
    for (trial_number, one_trial) in data_dict 
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  else
    # Note using threads doesn't guaranteee likelihoods are returned in same order
    # That's why the likelihoods are stored as key, value pairs so they can be rearranged later if needed
    @threads for trial_number in collect(eachindex(data_dict))
      one_trial = data_dict[trial_number]
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = max.(values(likelihoods), 1e-64)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(values(likelihoods)))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end
end


function compute_trials_nll_floop(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  likelihoods = Dict{Int, Float64}()
  data_dict = Dict(zip(1:length(data), data))

  if sequential_model
    for (trial_number, one_trial) in data_dict 
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  else
    # Note using threads doesn't guaranteee likelihoods are returned in same order
    # That's why the likelihoods are stored as key, value pairs so they can be rearranged later if needed
    @floop for trial_number in collect(eachindex(data_dict))
      one_trial = data_dict[trial_number]
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = max.(values(likelihoods), 1e-64)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(values(likelihoods)))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end
end


#########################
# Setup
#########################

# Read in data
dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
krajbich_data = ADDM.load_data_from_csv(dp*"Krajbich2010_behavior.csv", dp*"Krajbich2010_fixations.csv");
data = krajbich_data["18"];

# Pass fixed parameters to the model
model = ADDM.aDDM()
cur_params = Dict(:d=>.00085, :sigma=> .055, :θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)
for (k,v) in cur_params setproperty!(model, k, v) end
ADDM.convert_param_symbols(model)

likelihood_fn = ADDM.aDDM_get_trial_likelihood

#########################
# Run Benchmarks
#########################

## Current legacy version in package
b1 = @benchmark compute_trials_nll(model, data, likelihood_fn; return_trial_likelihoods = false)
println("compute_trials_nll (no likelihoods) = $(minimum(b1.times)/10^6)")

## Equivalent without array comprehension has same performance
# bench_results = @benchmark compute_trials_nll2(model, data, likelihood_fn; return_trial_likelihoods = false)
# println("compute_trials_nll (no likelihoods) = $(minimum(bench_results.times)/10^6)")

## Returning trial likelihoods does not slow things down
b2 = @benchmark compute_trials_nll(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll (w likelihoods) = $(minimum(b2.times)/10^6)")

## Increasing from 1 to 3 threads cut time by half
b3 = @benchmark compute_trials_nll_threads(model, data, likelihood_fn; return_trial_likelihoods = false)
println("compute_trials_nll_threads (no likelihoods) = $(minimum(b3.times)/10^6)")

## Again returning trial likelihoods does not make much of a difference
b4 = @benchmark compute_trials_nll_threads(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll_threads (w likelihoods) = $(minimum(b4.times)/10^6)")

## Are likelihoods returned in the same order? No!
# nll, ls = compute_trials_nll(model, data, likelihood_fn; return_trial_likelihoods = true)
# nll_t, ls_t = compute_trials_nll_threads(model, data, likelihood_fn; return_trial_likelihoods = true)

## Does storing likelihoods in dictionary slow things down?
## No! If anything it makes it faster!
b5 = @benchmark compute_trials_nll_threads_dict(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll_threads_dict (w likelihoods) = $(minimum(b5.times)/10^6)")

## FLoops: Similar to/slightly worse than @threads 
## But with a warning for boxed variables
## Though note less variability in histograms
b6 = @benchmark compute_trials_nll_floop(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll_floop (w likelihoods) = $(minimum(b6.times)/10^6)")


# Usage
# julia --project=../aDDM-Toolbox/ADDM.jl --threads 3 compute_trials_nll_benchmarks.jl

# Output
# compute_trials_nll (no likelihoods) = 96.487875
# compute_trials_nll (w likelihoods) = 96.854334
# compute_trials_nll_threads (no likelihoods) = 46.646833
# compute_trials_nll_threads (w likelihoods) = 46.692333
# compute_trials_nll_threads_dict (w likelihoods) = 36.11225
# ┌ Warning: Correctness and/or performance problem detected
# │   error =
# │    HasBoxedVariableError: Closure __##reducing_function#293 (defined in Main) has 1 boxed variable: likelihoods
# │    See: https://juliafolds.github.io/FLoops.jl/dev/howto/avoid-box/
# │    NOTE: To disable this warning, call `FLoops.assistant(false)`.
# └ @ Main ~/Documents/RangelLab/IntroToJuliaHPC/compute_trials_nll_benchmarks.jl:105
# compute_trials_nll_floop (w likelihoods) = 50.781708

# Conclusion
# Replace compute_trials_nll with compute_trials_nll_threads_dict in ADDM.jl