# Hacking Evolution - Beyond Genetic Algorithms
Repository for Artificial Intelligence Fundamentals course @Unipisa

Team Name: Hacktually Bunnies 🐰🐰🐰

Contributors: 
      <!-- Contributors -->
      <ul class="name-list">
        <li>@irenedovichi</li>
        <li>@lavo2</li>
        <li>@vornao</li>
      </ul>



# Introduction 🎬
In this project we present our idea of an Informed Genetic Algorithm. We present an approach that incorporates genetic algorithms, knowledge-based strategies, and informed mutations to solve the pathfinding problem within the NetHack game environment.

## Project Directory Structure 🗂️

```
./
├── 📂 project
│   ├── 📄 main.ipynb
│   ├── 📄 utils.py
│   ├── 📄 run_experiments.py
│   ├── 📄 requirements.txt
│   └── 📂 kb
│       └── 📄 kb.pl
│   └── 📂 maps
│       ├── 📄 real_maze.des
│       └── ...
│   └── 📂 experiments
│       └── 📂 exp_manhattan
│           └── 📂 run_8_map_real_maze
│                  ├── 📄 fitness.json
│                  └── 📄 stats.csv
│           └── 📂 run_16_map_real_maze
│                  └── ...
│           └── 📂 run_32_map_real_maze
│                  └── ... 
│       └── 📂 exp_informed
│           └── ...
│       └── 📂 exp_dynamic
│           └── ...
├── 📄 README.md
└── 📄 .gitignore
```

## Informed Genetic Algorithm 🧬

### Informed Mutations 🕵🏻

To enhance the standard genetic algorithm, we introduce a mutation operator called "informed mutations." This operator utilizes knowledge-based error bitmaps to prevent offspring from repeating the same mistakes as their parents. Three types of errors are considered: loops, dead ends, and wrong actions. The mutation probability is determined based on a softmax function applied to the error bitmaps.

### Python Implementation ⚙️

Our genetic algorithm follows a standard structure with population generation, fitness evaluation, selection of the best individuals, and offspring generation through crossover and informed mutations. The implementation is available in the `utils.py` file, specifically within the `run_genetic` function.

## Experimental Setting 👩🏻‍🔬

We conducted extensive experiments to assess the performance of our algorithm using different fitness functions and population sizes. A total of 900 experiments were run, comparing the results against a true random genetic algorithm without knowledge-based enhancements. Details of each experiment, including fitness, generation, errors, distance, and winner information, are stored in CSV files within the `results` folder.

## Running Instructions 🚶🏻‍♂️

To replicate and explore the experiments, follow these steps:

1. Install Python 3.10.
2. Install [pyswip](https://pypi.org/project/pyswip/) for Prolog.
3. Install dependencies from the `requirements.txt` file:

```bash
pip install -r project/requirements.txt
```

4. Run experiments using the provided script:

```bash
python project/run_experiments.py --experiments [ne] --individuals [ni] --max_generations [g] --workers [w] --fitness [f]
```
Adjust the `--fitness` parameter (0, 1, or 2) to choose between Manhattan Fitness, Informed Manhattan Fitness, and Dynamic Manhattan Fitness.

Feel free to explore the code, replicate experiments, and adapt the algorithms for your own projects. We welcome contributions and discussions to further enhance this path finding solution. 🔭