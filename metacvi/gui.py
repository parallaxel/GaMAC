import os

from metacvi.collector import DatasetInfoCollector
from metacvi.utils import traverse_data

NUM_SAMPLES = DatasetInfoCollector.PARTITIONS_TO_ESTIMATE
ACCESSOR_IDX = 0


class ComparisonContext:
    def __init__(self, data_path, sorted_indices):
        self.insertion_idx = len(sorted_indices)
        self.sorted_indices = sorted_indices
        self.data_path = data_path
        self.alternative = "ESTIMATE"

    def swap(self):
        self.alternative = "ESTIMATE" if self.alternative == "SORTED" else "SORTED"

    def shift(self):
        if self.insertion_idx > 0:
            self.insertion_idx -= 1

    def rendered_image_idx(self):
        return len(self.sorted_indices) if self.alternative == "ESTIMATE" else self.insertion_idx - 1

    def label(self):
        return f'{self.alternative} [{self.data_path}, {self.insertion_idx} / {len(self.sorted_indices)}]'

    def colour(self):
        return 'red' if self.alternative == "ESTIMATE" else 'blue'


class DataContext:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = [self._photo_image(idx) for idx in range(NUM_SAMPLES)]

        self._accessor_path = f'data/{data_path}/accessor-{ACCESSOR_IDX}.txt'
        if os.path.exists(self._accessor_path):
            raise FileExistsError(f"Data {data_path} has been already estimated by accessor {ACCESSOR_IDX}")

        self._sorted_indices = [0]
        self._comp_ctx = None

    def comp_ctx(self) -> ComparisonContext:
        return self._comp_ctx

    def next_comp_ctx(self):
        if len(self._sorted_indices) < NUM_SAMPLES:
            self._comp_ctx = ComparisonContext(self.data_path, self._sorted_indices)
        else:
            raise AttributeError(f'No more comparisons for {self.data_path}')

    def has_next(self):
        return self._current_idx < NUM_SAMPLES

    def save(self):
        insertion_idx = self.comp_ctx().insertion_idx
        self._sorted_indices.insert(insertion_idx, self._current_idx)

    def persist(self):
        with open(self._accessor_path, 'w') as fp:
            fp.write(self._sorted_indices.__str__())

    def _photo_image(self, idx):
        return PhotoImage(file=f'data/{self.data_path}/img-{idx}.png')

    @property
    def _current_idx(self):
        return len(self._sorted_indices)


class Application:
    def __init__(self, data_for_estimation):
        self._all_data = data_for_estimation
        self._current_idx, self._data_ctx = -1, None

    def data_ctx(self) -> DataContext:
        if self._data_ctx is None:
            raise AttributeError('No data context available')
        return self._data_ctx

    def next_data_ctx(self):
        if self._current_idx < len(self._all_data):
            self._current_idx += 1
            data_path = self._all_data[self._current_idx]
            self._data_ctx = DataContext(data_path)
            self._data_ctx.next_comp_ctx()
        else:
            raise AttributeError('Congratulations! You have estimated all available data')


def is_estimated(data_path):
    content = os.listdir(f'data/{data_path}')
    return f'accessor-{ACCESSOR_IDX}.txt' in content


if __name__ == '__main__':
    from tkinter import *

    root = Tk()
    root.title(f"GAMaC Assessment [Accessor #{ACCESSOR_IDX}]")
    root.geometry("3200x2400")

    data_for_estimation = [
        data_path for data_path, estimated in traverse_data(is_estimated).items() if not estimated
    ]
    application = Application(data_for_estimation)

    header = Label(font=("Arial", 20))
    header.pack()

    image = Label()
    image.pack(expand=True, anchor=CENTER)


    def update_ui():
        data_ctx = application.data_ctx()
        comp_ctx = data_ctx.comp_ctx()

        header.config(
            text=comp_ctx.label(),
            foreground=comp_ctx.colour()
        )

        image_idx = comp_ctx.rendered_image_idx()
        image.config(
            highlightbackground=comp_ctx.colour(),
            highlightthickness=4,
            image=data_ctx.images[image_idx]
        )


    def persist_ui(data_ctx):
        header.config(
            text=f'PERSISTING {data_ctx.data_path} ...',
            foreground='green'
        )
        image.config(
            highlightbackground='green',
            highlightthickness=4,
        )
        root.update_idletasks()


    def swap(_):
        application.data_ctx().comp_ctx().swap()
        update_ui()


    def shift(_):
        application.data_ctx().comp_ctx().shift()
        update_ui()


    def insert(_):
        data_ctx = application.data_ctx()
        data_ctx.save()
        if data_ctx.has_next():
            data_ctx.next_comp_ctx()
        else:
            persist_ui(data_ctx)
            data_ctx.persist()
            application.next_data_ctx()
        update_ui()


    root.bind('<KeyPress-Tab>', swap)
    root.bind('<KeyPress-Return>', insert)
    root.bind('<KeyPress-BackSpace>', shift)

    application.next_data_ctx()

    update_ui()
    root.mainloop()
