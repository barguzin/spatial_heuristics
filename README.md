# Spatial Heuristics

> prepared by Evgeny Noi 

## Week 5 of GEOG 294 course at UC Santa Barbara

Toy examples for heuristics used in locational science.  

---

### Installation and usage

#### Installation: 
1. Clone repo via git or Github Desktop ```git clone https://github.com/barguzin/spatial_heuristics.git```
2. Navigate to the cloned repo directory ```cd spatial_heuristics```
3. Create a Python environment (conda or pip): ```conda create -n env_name```
4. Activate the newly created environment: ```conda activate env_name```
5. Install requirements using pip: ```pip install -r requirements.txt```

#### Usage: 
1. Run *check_funs.ipynb* notebook to see usage examples


---

The functions are located in *spam/core.py*: 

**make_points(n)** -  Generate set of points (sites) on a 2d plane using a normal distribution. Where $n$ is the number of points. Results are written into demand.csv in a root folder.  

**makde_grid(q)** - Generate regular square grid over a 2d plane. Where $q$ is the number of cells along an axis.  Yields q*q potential sites. Results are written into *facility.csv* in a root folder. 

**calculate_dist** - Calculate distance via scipy.distance.cdist and store results into *distance_matrix.csv* in a root directory.  

**get_covered(r)** - For each facility find the demand points that are covered given a radius ($r$). Saves *pairwise_dist.csv* (pairwise distance between facilities and demand points), *total_pop.csv* (total covered population for each facility), *covered.pickle* (id of covered demand for each facility_id)

**naive_greedy(n_sited, sort_par='cnt')** - Implements naive Greedy Addition algorithm for MCLP with $n\_cited$ denoting the number of facilities to be sited and $sort\_par$ defining what variables to use for sorting (either number of facilities covered by each facility or total covered population). Saves *sited_id* (id of sited facilities) and *covered_id* (id of covered demand). The output is also provided via logger available at *./out.txt*.

**plot_solution** - plots solution

---

### Implemented:

1. Greedy Addition for maximal covering 


### To be implemented: 

1. Teitz & Bart

---

### References: 

1. Teitz, M. B., & Bart, P. (1968). Heuristic methods for estimating the generalized vertex median of a weighted graph. Operations research, 16(5), 955-961.
