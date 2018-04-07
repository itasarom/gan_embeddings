#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

cd ../../MUSE
python evaluate.py --src_lang fr --tgt_lang en --src_emb ../gan_embeddings/GAN/embeds_1_tmp.vec --tgt_emb ../gan_embeddings/GAN/embeds_2_tmp.vec >> ../gan_embeddings/validation/tgt_src.txt 2>&1
python evaluate.py --src_lang en --tgt_lang fr --src_emb ../gan_embeddings/GAN/embeds_2_tmp.vec --tgt_emb ../gan_embeddings/GAN/embeds_1_tmp.vec >> ../gan_embeddings/validation/src_tgt.txt 2>&1