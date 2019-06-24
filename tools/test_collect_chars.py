import unittest

import re


class TestCollectChars(unittest.TestCase):
    def setUp(self):
        self.input_file = r"E:\project\deepseg-master\testdata\fortest\test.std"
        self.output_file = r"E:\project\deepseg-master\testdata\fortest\test_1_.txt.vb"
        self.chinese_pattern = re.compile("[\u4e00-\u9fa5]")

    def testCollectChars(self):
        vs = set()
        with open(self.input_file, mode='rt', encoding='utf8', buffering=8142) as f_in:
            for line in f_in:
                vocabs = line.strip("\n").split(' ')
                for v in vocabs:
                    vs.add(v + "\n")
        vs = sorted(vs)
        with open(self.output_file, mode='wt', encoding="utf8", buffering=8142) as f_out:
            f_out.write("<UNK>" + "\n")
            f_out.write("<PAD>" + "\n")
            f_out.write("".join(vs) + "\n")


if __name__ == '__main__':
    unittest.main()
