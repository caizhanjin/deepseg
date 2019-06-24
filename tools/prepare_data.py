import re
import os

data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data"))


class PrepareData(object):
    def __init__(self):
        self.input_file = os.path.join(data_dir, 'train.std')
        self.output_file_feature = os.path.join(data_dir, "feature.txt")
        self.output_file_label = os.path.join(data_dir, "label.txt")
        self.chinese_pattern = re.compile("[\u4e00-\u9fa5]")

    def prepare_data(self):
        with open(self.input_file, mode='rt', encoding='utf8', buffering=8142) as f_in, \
                open(self.output_file_feature, mode='wt', encoding="utf8", buffering=8142) as f_feature, \
                open(self.output_file_label, mode='wt', encoding='utf8', buffering=8142) as f_label:
            for line in f_in:
                if not line:
                    continue
                features = self._gen_char_file(line)
                f_feature.write("".join(features).strip(',') + "\n")
                labels = self._gen_tag_file(line)
                f_label.write("".join(labels).strip(",") + "\n")

    def _gen_char_file(self, line):
        vocabs = []
        line = line.strip("\n").replace(" ", ",")
        lines = line.split(',')
        for vs in lines:
            if not self.chinese_pattern.findall(vs):
                word = vs
            else:
                word = ','.join(vs)
            vocabs.append(word + ",")
        return vocabs

    def _gen_tag_file(self, line):
        tags = []
        words = line.split(' ')
        for word in words:
            word = word.strip('\n')
            if len(word) == 1:
                tags.append("S" + ",")
            elif len(word) >= 2:
                if not self.chinese_pattern.findall(word):
                    tags.append("S" + ",")
                else:
                    tags.append("B" + ",")
                    for w in word[1:len(word) - 1]:
                        if w:
                            tags.append("M" + ",")
                    tags.append("E" + ",")
        return tags
