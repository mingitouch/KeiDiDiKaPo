#!/bin/bash

echo "cd /home/yanming.ym/download/mallet-2.0.7"
cd /home/yanming.ym/download/mallet-2.0.7

pwd
date
echo '开始训练模型'

./bin/mallet import-file --input ~/kdd-cup/all --output topic-input.mallet --keep-sequence --remove-stopwords

date

echo "读取数据完成"

echo "开始建立topic文件"

./bin/mallet train-topics --input topic-input.mallet --optimize-interval 10 --optimize-burn-in 30 --num-topics 100 --num-iterations 400 --output-state topic-state.gz --output-doc-topics ~/kdd-cup/lda.topic --output-model ~/kdd-cup/model.topic

date

echo "topic文件建立完毕"
