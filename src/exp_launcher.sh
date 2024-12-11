#!/bin/bash
export PYTHONPATH="$PATH:$PWD"

declare -a env=("MiniGrid-Empty-16x16-v0"
                "MiniGrid-SimpleCrossingS9N1-v0"
                "MiniGrid-SimpleCrossingS11N1-v0"
                "MiniGrid-SimpleCrossingS13N1-v0"
                "MiniGrid-SimpleCrossingS15N1-v0"
                "MiniGrid-FourRooms-v0"
                )

declare -a exp=("config_files_sac/1-explore-sac-cv.yaml"
                "config_files_sac/2-explore-sac-mv.yaml"
                "config_files_sac/3-explore-sac.yaml"
                "config_files_sac/4-control-sac-cv.yaml"
                "config_files_sac/5-control-sac-mv.yaml"
                "config_files_sac/6-control-sac.yaml"
                )

for en in "${env[@]}"
do
  for (( i=0; i<${#exp[@]}; i++ ));
  do
    for s in {12345..12359}
    do
      path="saves-density-rl/$en-$i-$s"
      sbatch exp_slurm.sh $PWD ${exp[$i]} $s $en $path
    done
  done
done
