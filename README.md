# Julia Module for HypoSVI and Eikonet

### Training an Eikonet for computing travel times
```
include("Eikonet.jl")
Eikonet.train("eikonet_hk77_interp.json")
```
Generally speaking, you will want the test error for the Eikonet to be in the range of 1e-6, to ensure the solutions are sufficiently accurate numerically.

### Locating earthquakes with HypoSVI
Then you will need to exit and re-launch Julia in parallel with
```
julia -p n_procs
```
Then you can import HypoSVI with 
```
@everywhere include("HypoSVI.jl")
```
You can run the code using:
```
HypoSVI.locate_events_ssst_knn("hyposvi.json", initial_screening=true)
```
