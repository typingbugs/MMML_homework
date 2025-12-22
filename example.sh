#!/bin/bash
set -ex

# clear files and db
rm -rf data/papers/*
rm -rf data/images/*
rm -rf data/vector_db/*

# ======== test paper functions ========
tree data/test_input/paper/

python main.py \
    add_paper "data/test_input/paper/tagged/deep search/Gao et al. - 2025 - SmartRAG Jointly Learn RAG-Related Tasks From the Environment Feedback.pdf" \
    --topics "deep search"

python main.py \
    add_paper "data/test_input/paper/tagged/deep search/Guan et al. - 2025 - DeepRAG Thinking to Retrieve Step by Step for Large Language Models.pdf" \
    --topics "deep search"

python main.py \
    add_paper "data/test_input/paper/tagged/RL algorithm/Schulman 等 - 2017 - Proximal Policy Optimization Algorithms.pdf" \
    --topics "RL algorithm"

python main.py \
    add_paper "data/test_input/paper/tagged/RL algorithm/Guo et al. - 2025 - DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.pdf" \
    --topics "RL algorithm,deepseek"

tree data/papers

python main.py \
    search_paper "deep reinforcement learning for multilingual reasoning" \
    --top_k 3

python main.py \
    add_paper "data/test_input/paper/untagged"

tree data/papers

python main.py \
    search_paper "deep reinforcement learning for multilingual reasoning" \
    --top_k 3

# ======== test image functions ========
tree data/test_input/image/

python main.py \
    add_image "data/test_input/image/tagged/cat/c23d86516f662895c5b93259049b66b6.png" \
    --topics "cat"

python main.py \
    add_image "data/test_input/image/tagged/cat/c857c245a8483779d74ab803cc7542b2.png" \
    --topics "cat"

python main.py \
    add_image "data/test_input/image/tagged/dog/2cd2bd4b097faffaa4c8a481fbcd4c1c.png" \
    --topics "dog"

python main.py \
    add_image "data/test_input/image/tagged/dog/68a1f4eefccbca873075f834e089e84c.png" \
    --topics "dog"

python main.py \
    add_image "data/test_input/image/tagged/parrot/608c21c07068412b962441755183e046.png" \
    --topics "parrot"

python main.py \
    add_image "data/test_input/image/tagged/parrot/a9247d3b3bcf44450b7149d0e764dfe5.png" \
    --topics "parrot"

tree data/images

python main.py \
    search_image "data/test_input/image/untagged/8fe1f9c9fc3faf2c953ca085cb593e86.png" \
    --top_k 3

python main.py \
    add_image "data/test_input/image/untagged"

tree data/images

python main.py \
    search_image "a photo of a cute dog" \
    --top_k 3