import os
import re


class Filter(object):

    def accept(self, line):
        raise NotImplementedError()


class SingleChineseFilter(Filter):

    def __init__(self):
        self.pattern = re.compile("[\u4e00-\u9fa5]+")
        self.key_words = ['号', '层', '幢', '与', '栋', '旁', '室', '楼']

    def accept(self, line):
        words = line.split(" ")
        acc = False
        for w in words:
            if len(w) > 1:
                continue
            if w in self.key_words:
                continue
            m = self.pattern.findall(w)
            if not m:
                continue
            acc = True
            break
        return acc


class DiffFilter(object):

    def __init__(self, diff_file):
        self.diff_file = diff_file

    def filter(self, filters):
        filepath = str(self.diff_file).split(os.sep)[0:-1]
        skipped_file = os.path.join(os.sep.join(filepath), "skipped.txt")
        selected_file = os.path.join(os.sep.join(filepath), "selected.txt")
        editable = os.path.join(os.sep.join(filepath), "editable.txt")

        with open(self.diff_file, mode="rt", encoding="utf8", buffering=8192) as f, \
                open(skipped_file, mode="wt", encoding="utf8", buffering=8192) as skip, \
                open(selected_file, mode="wt", encoding="utf8", buffering=8192) as select, \
                open(editable, mode="wt", encoding="utf8", buffering=8192) as ed:
            while True:
                line = f.readline()
                if not line:
                    break
                # empty = f.readline()
                jieba = f.readline().replace("STD_label:", "").strip("\n")
                model = f.readline().replace("SEG_label:", "").strip("\n")
                print(jieba)
                print(model)
                print()
                if not jieba or not model:
                    continue
                skip_line = True
                for flt in filters:
                    if flt.accept(model):
                        skip_line = False
                        break
                if not skip_line:
                    select.write("jieba: %s\n" % jieba)
                    select.write("model: %s\n" % model)
                    select.write("\n")

                    ed.write(model + "\n")
                    ed.write("\n\n")
                else:
                    skip.write("jieba: %s\n" % jieba)
                    skip.write("model: %s\n" % model)
                    skip.write("\n")


if __name__ == "__main__":
    filters = [SingleChineseFilter()]
    diff_file = "C:\\Users\\allen.luo\\Desktop\\diff_filter\\test.diff.txt"
    differ = DiffFilter(diff_file)
    differ.filter(filters)
