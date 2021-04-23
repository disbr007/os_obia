## Open-Source Object-Based-Image-Analysis Workflow

## Set up
Using conda, to create the python environment on a Windows machine:
```
conda env create -f win_environment.yml 
```

This has not been tested, but the following should create the python
environment regardless of the OS:
```
conda env create -f os_agnostic_env.yml
```

To activate the environment:
```
conda activate osobia
```

And to start the jupyter server:
```
jupyter notebook
``` 
or:
```
jupyter lab
```