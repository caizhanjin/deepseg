import unittest
import os
import re
import multiprocessing
import collections
from tools.reload_model import ReloadModel
import subprocess


def count_file_lines_wc(file):
    return (subprocess.check_output(["wc", "-l", file]).strip().split()[0]).decode("utf8")


def count_file_lines(file):
    count = 0
    if not os.path.exists(file):
        return count
    with open(file, mode="rt", encoding="utf8") as f:
        for l in f:
            count += 1
    return count


def ask_model_decode_result(p):
    pass


def split_file_to_patch(input_file, num_parts):
    dir = input_file.split(os.sep)[0:-1]
    dir_path = os.path.join(os.sep.join(dir), "my")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num_lines = count_file_lines(input_file)
    lines_each = num_lines // num_parts + 1
    parts = []
    with open(input_file, mode="rt", encoding="utf8", buffering=8192) as r:
        for i in range(num_parts):
            name = "test_" + str(i) + ".txt"
            f = os.path.join(dir_path, name)
            with open(f, mode="wt", encoding="utf8", buffering=8192) as fw:
                for j in range(lines_each):
                    line = r.readline()
                    if not line:
                        break
                    fw.write(line)
            parts.append(f)
    return parts


def patch_file_to_total(out_files, model_file):
    for file in out_files:
        with open(file, 'rt', encoding='utf8', buffering=8142)as f_in, \
                open(model_file, 'wt', encoding='utf8', buffering=8142)as f_out:
            while True:
                if not f_in.readline():
                    break
                f_out.write(f_in.readline())


def collect_dict_datas(input, output):
    pattern = re.compile("[0-9]+")
    counter = collections.Counter()
    with open(input, mode="rt", encoding="utf8", buffering=8192) as f, \
            open(output, mode='wt', encoding='utf8', buffering=8142)as f_out:
        for line in f:
            vocabs = line.strip("\n").split(" ")
            for vs in vocabs:
                if len(vs) < 2:
                    continue
                if pattern.findall(vs):
                    continue
                counter[vs] += 1
        words = counter.most_common(250000)
        vocabs = set()
        for item in words:
            vocabs.add(item[0])
        for v in vocabs:
            f_out.write(v + "\n")


class TestCollectModelResult(unittest.TestCase):
    def setUp(self):
        self.input_file = r"D:\PyCharmProjects\deepseg\data\train.feature.txt"
        self.model_file = r"D:\PyCharmProjects\deepseg\data\train.models.txt"
        self.output_file = r"D:\PyCharmProjects\deepseg\data\train.dict.txt"

    def testCollectModelResult(self):
        model = ReloadModel(input_file=self.input_file, output_file=self.output_file)
        model.serving()

    # def testMultiprocessCollectModelResult(self):
    #     out_files = []
    #     pool = multiprocessing.Pool(processes=4)
    #     parts = split_file_to_patch(self.input_file, 5)
    #     print("\n分割文件完成")
    #     for p in parts:
    #         model = ReloadModel(input_file=p, output_file=self.output_file)
    #         result = pool.apply_async(func=model.serving(), args=())
    #         out_files.append(result.get())
    #         print(result.get())
    #     pool.close()
    #     pool.join()
    #     print("\n模型处理完成")
    #     patch_file_to_total(out_files,self.model_file)
    #     print("\n合并文件完成")
    #     collect_dict_datas(self.model_file,self.output_file)
    #     print("\n词典收集完成")


if __name__ == '__main__':
    unittest.main()
