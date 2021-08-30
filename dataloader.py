from utils import read_file
import numpy


class DataLoader:
    def __init__(self, file_path, purpose='train', batch_size=32, bert_path=''):
        self.purpose = purpose
        self.batch_size = batch_size
        self.data = read_file(file_path, bert_path)
        self.ids_1 = numpy.array([triple[0][0] for triple in self.data], dtype=float)
        self.masks_1 = numpy.array([triple[0][1] for triple in self.data], dtype=float)
        self.ids_2 = numpy.array([triple[1][0] for triple in self.data], dtype=float)
        self.masks_2 = numpy.array([triple[1][1] for triple in self.data], dtype=float)
        self.label = numpy.array([int(triple[2])for triple in self.data], dtype=float)

    def get_batch_data(self):
        total_steps = len(self.data) // self.batch_size
        for cur_step in range(total_steps):
            begin = cur_step * self.batch_size
            end = (cur_step + 1) * self.batch_size
            batch_ids_1 = self.ids_1[begin: end]
            batch_masks_1 = self.masks_1[begin: end]
            batch_ids_2 = self.ids_2[begin: end]
            batch_masks_2 = self.masks_2[begin: end]
            batch_label = self.label[begin: end]
            yield batch_ids_1, batch_masks_1, batch_ids_2, batch_masks_2, batch_label


if __name__ == '__main__':
    a = DataLoader('data/test.txt', purpose='test')
    for id_1, mask_1, id_2, mask_2, label in a.get_batch_data():
        print(id_1.shape, mask_1.shape, id_2.shape, mask_2.shape, label.shape)
        break
