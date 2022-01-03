#coding:utf-8
import sys
import os
import json
import numpy as np

def txt_to_json(input_file, output_file, NEG=5):
    """\t分割的文本，转为json格式"""
    with open(output_file,"w",encoding="utf-8") as fo:
        # 读取文件内容
        contents = []
        for line in open(input_file,"r",encoding="utf-8"):
            line = line.strip().split("\t")
            contents.append(line)

        # 随机采样NEG个负样本
        for idx,data in enumerate(contents):
            neg_list = []
            for k in range(NEG):
                while True:
                    rng_idx = np.random.randint(len(contents))
                    if rng_idx not in neg_list and rng_idx != idx:
                        res = {
                            "query": data[0],
                            "doc_pos": data[1],
                            "doc_neg": contents[rng_idx][1]
                        }
                        fo.write(json.dumps(res, ensure_ascii=False)+"\n")
                        neg_list.append(rng_idx)
                        break
    print("Done.")


if __name__ == "__main__":
    txt_to_json(sys.argv[1], sys.argv[2])


