using Base.Threads
using BenchmarkTools
using FLoops

function floop_ex(xs, ys)
  @floop for (x, y) in zip(xs, ys)
    a = x + y
    b = x - y
    @reduce(s += a, t += b)
  end
  return (s,t)
end

function seq_ex(xs, ys)
  s = 0
  t = 0
  for (x, y) in zip(xs, ys)
    a = x + y
    b = x - y
    s += a
    t += b
  end
  return (s,t)
end

function threads_ex(xs, ys)
  s = 0
  t = 0
  @threads for (x, y) in collect(zip(xs, ys))
    a = x + y
    b = x - y
    s += a
    t += b
  end
  return (s,t)
end

xs = 1:30000
ys = 1:2:60000

# All three give the same output for s and t

b1 = @benchmark floop_ex(xs, ys)
println("floop_ex bechmark = $(median(b1.times)/10^6)")

b2 = @benchmark seq_ex(xs, ys)
println("seq_ex bechmark = $(median(b2.times)/10^6)")

b3 = @benchmark threads_ex(xs, ys)
println("threads_ex bechmark = $(median(b3.times)/10^6)")

# julia --project=../aDDM-Toolbox/ADDM.jl --threads 1 learning_floops.jl

# Surprisingly good for floop and Surprisingly terrible for threads
# Floop does use more memory than seq_ex but no even close to threads
# floop_ex bechmark = 0.00030094160583941606
# seq_ex bechmark = 0.014917
# threads_ex bechmark = 0.9588125

# julia --project=../aDDM-Toolbox/ADDM.jl --threads 3 learning_floops.jl

# floop_ex bechmark = 0.014042
# seq_ex bechmark = 0.014917
# threads_ex bechmark = 1.079