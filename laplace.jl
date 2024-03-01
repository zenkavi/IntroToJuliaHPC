# Usage julia -t 3 --project laplace.jl

using Base.Threads
using BenchmarkTools

function lap2d!(u, unew)
  M, N = size(u)
  for j in 2:N-1
      for i in 2:M-1
          @inbounds unew[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
      end 
  end
end

function threads_lap2d!(u, unew)
  M, N = size(u)
  @threads for j in 2:N-1
      for i in 2:M-1
          @inbounds unew[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
      end 
  end
end

function setup(N=4096, M=4096)
  u = zeros(M, N)
  # set boundary conditions
  u[1,:] = u[end,:] = u[:,1] = u[:,end] .= 10.0
  unew = copy(u);
  return u, unew
end

u, unew = setup();

bench_results = @benchmark lap2d!(u, unew)
#println(minimum(bench_results.times))
println("serial time (small) = $(minimum(bench_results.times)/10^6)")

bench_results = @benchmark threads_lap2d!(u, unew)
# println(minimum(bench_results.times))
println("threads time (small) = $(minimum(bench_results.times)/10^6)")

u, unew = setup(8192, 8192);

bench_results = @benchmark lap2d!(u, unew)
#println(minimum(bench_results.times))
println("serial time (big) = $(minimum(bench_results.times)/10^6)")

bench_results = @benchmark threads_lap2d!(u, unew)
# println(minimum(bench_results.times))
println("threads time (big) = $(minimum(bench_results.times)/10^6)")