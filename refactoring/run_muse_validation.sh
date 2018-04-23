#!/bin/bash

export CUDA_VISIBLE_DEVICES="6"

cd ../../MUSE
python evaluate.py --src_lang en --tgt_lang es --src_emb ../gan_embeddings/refactoring/embeds_1_tmp.vec --tgt_emb ../gan_embeddings/refactoring/embeds_2_tmp.vec >> ../gan_embeddings/validation/tgt_src.txt 2>&1
python evaluate.py --src_lang es --tgt_lang en --src_emb ../gan_embeddings/refactoring/embeds_2_tmp.vec --tgt_emb ../gan_embeddings/refactoring/embeds_1_tmp.vec >> ../gan_embeddings/validation/src_tgt.txt 2>&1
