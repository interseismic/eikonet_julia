# Julia Module for HypoSVI and Eikonet

If you use this module please cite the HypoSVI and Eikonet papers:
```
@article{smith2022hyposvi,
  title={HypoSVI: Hypocentre inversion with Stein variational inference and physics informed neural networks},
  author={Smith, Jonthan D and Ross, Zachary E and Azizzadenesheli, Kamyar and Muir, Jack B},
  journal={Geophysical Journal International},
  volume={228},
  number={1},
  pages={698--710},
  year={2022},
  publisher={Oxford University Press}
}
```
```
@article{smith2020eikonet,
  title={Eikonet: Solving the eikonal equation with deep neural networks},
  author={Smith, Jonathan D and Azizzadenesheli, Kamyar and Ross, Zachary E},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={12},
  pages={10685--10696},
  year={2020},
  publisher={IEEE}
}
```

### Training an Eikonet for computing travel times
```
include("Eikonet.jl")
Eikonet.train("eikonet.json")
```
Generally speaking, you will want the test error for the Eikonet to be in the range of 1e-6, to ensure the solutions are sufficiently accurate numerically. It is important for the velocity model to extend deep enough that ray paths are able to sample that part of the medium if necessary, or else the travel times for large distances may be inaccurate. For regional distances, this depth may need to be in the range of 50-100km 

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
HypoSVI.locate_events_ssst("hyposvi.json")
```
