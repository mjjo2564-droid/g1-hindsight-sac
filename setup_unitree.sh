#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${project_dir}/third_party/unitree_rl_gym"
unitree_commit="276801e46c5d433564f24658bac64f254b7d2d4b"

mkdir -p "${project_dir}/third_party"

if [[ -d "${target_dir}/.git" ]]; then
    git -C "${target_dir}" fetch --depth 1 origin "${unitree_commit}"
else
    mkdir -p "${target_dir}"
    git -C "${target_dir}" init
    git -C "${target_dir}" remote add origin https://github.com/unitreerobotics/unitree_rl_gym.git
    git -C "${target_dir}" fetch --depth 1 origin "${unitree_commit}"
fi

git -C "${target_dir}" checkout --detach "${unitree_commit}"

test -f "${target_dir}/deploy/pre_train/g1/motion.pt"
test -f "${target_dir}/resources/robots/g1_description/scene.xml"

echo "Unitree G1 walking policy is ready: ${target_dir}"
