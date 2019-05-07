import os
import argparse
import struct
from tensorflow.core.example import example_pb2
import glob
import tqdm

def parse_bin(src_dir, dset, max_input_tokens, max_output_tokens):
    articles = []
    summaries = []
    
    filelist = glob.glob(os.path.join(src_dir, dset)+'*')
    filelist.sort()
    print(filelist)

    max_input_tokens = int(max_input_tokens)
    max_output_tokens = int(max_output_tokens)
    
    print("Retrieving all the articles/summaries...")
    for f in tqdm.tqdm(filelist):
        reader = open(f, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break  # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack(
                '%ds' % str_len, reader.read(str_len))[0]
            tf_example = example_pb2.Example.FromString(example_str)

            examples = []
            for key in tf_example.features.feature:
                examples.append(
                    '%s' % (tf_example.features.feature[key].bytes_list.value[0]))

            articles.append(examples[0][2:-1])
            summaries.append(examples[1][2:-1])

    print("Cleaning dataset...")
    articles = [clean(art,max_input_tokens) for art in articles]
    summaries = [clean(sum,max_output_tokens) for sum in summaries]

    articles_2=[]
    summaries_2 = []
    for i in range(len(articles)):
        if len(articles[i])>5 and len(summaries[i])>5:
            articles_2.append(articles[i])
            summaries_2.append(summaries[i])


    return articles_2, summaries_2


def clean(text,max_tokens):
    text = text.split(" ")
    text = [s for s in text if s not in ["<s>", "</s>", "\n"]]
    if len(text)==0:
        print("empty article")
    text = text[:max_tokens]
    text = " ".join(text) + "\n"
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='finished_files/chunked')
    parser.add_argument('--tgt_dir', default='cnn_dataset')
    parser.add_argument('--max_source_tokens', default=100000)
    parser.add_argument('--max_target_tokens', default=100000)

    args = parser.parse_args()

    cnn_debug_dir = os.path.join(args.tgt_dir, "cnn_debug")
    cnn_full_dir = os.path.join(args.tgt_dir, "cnn_full")


    if not os.path.isdir(cnn_debug_dir):
        os.makedirs(cnn_debug_dir)
    
    if not os.path.isdir(cnn_full_dir):
        os.makedirs(cnn_full_dir)

    for set in ['train', 'val', 'test']:
        articles, summaries = parse_bin(args.source_dir, set, args.max_source_tokens, args.max_target_tokens)
        print("Retrieved articles and summaries for {}".format(set))

        with open(os.path.join(cnn_debug_dir, "src_"+set+".txt"), 'w') as f:
            f.writelines(articles[:1000])
        with open(os.path.join(cnn_debug_dir, "tgt_"+set+".txt"), 'w') as f:
            f.writelines(summaries[:1000])

        with open(os.path.join(cnn_full_dir, "src_"+set+".txt"), 'w') as f:
            f.writelines(articles)
        with open(os.path.join(cnn_full_dir, "tgt_"+set+".txt"), 'w') as f:
            f.writelines(summaries)
       
        print("Wrote articles and summaries for {}".format(set))
