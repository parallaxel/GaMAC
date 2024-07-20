import numpy as np
import pandas as pd

from metacvi.collector import DatasetInfoCollector

NUM_SAMPLES = DatasetInfoCollector.PARTITIONS_TO_ESTIMATE
ACCESSOR_IDX = 0


class ComparisonContext:
    def __init__(self, data_name, left_idx, right_idx):
        self.left_idx, self.right_idx = left_idx, right_idx
        self.data_path, self.current = data_name, "LEFT"

        self.left_image = PhotoImage(file=f"data/{data_name}/img-{left_idx}.png")
        self.right_image = PhotoImage(file=f"data/{data_name}/img-{right_idx}.png")

    def swap(self):
        self.current = "RIGHT" if self.current == "LEFT" else "LEFT"

    @property
    def current_image(self):
        return self.left_image if self.current == "LEFT" else self.right_image

    @property
    def label(self):
        return f'[{self.data_path}] {self.left_idx} vs {self.right_idx}'


class DataContext:
    def __init__(self, data_name):
        self.data_name = data_name
        self.results = np.zeros(shape=(NUM_SAMPLES, NUM_SAMPLES), dtype=int)

    def next_pair(self):
        pairs = list()
        for left_idx in range(1, NUM_SAMPLES):
            for right_idx in range(left_idx):
                if self.results[left_idx, right_idx] == 0:
                    pairs.append((left_idx, right_idx))
        if len(pairs) == 0:
            return None
        random_idx = np.random.randint(0, len(pairs))
        return pairs[random_idx]

    def has_next(self):
        return self.next_pair() is not None

    def next_comparison(self) -> ComparisonContext:
        indices = self.next_pair()
        if indices is None:
            raise ValueError(f"No more comparisons available for {self.data_name}")
        return ComparisonContext(self.data_name, indices[0], indices[1])

    def save(self, comp: ComparisonContext):
        if comp.current == "LEFT":
            self.results[comp.left_idx, comp.right_idx] = 1
            self.results[comp.right_idx, comp.left_idx] = -1
        else:
            self.results[comp.left_idx, comp.right_idx] = -1
            self.results[comp.right_idx, comp.left_idx] = 1

    def persist(self):
        pd.DataFrame(self.results).to_csv(
            f'data/{self.data_name}/accessor-{ACCESSOR_IDX}.csv', header=False, index=False
        )


class ContextHolder:
    def __init__(self, context: ComparisonContext):
        self.context = context


if __name__ == '__main__':
    from tkinter import *
    from tkinter import ttk

    data_context = DataContext('wine-quality-red/tsne')

    root = Tk()
    root.title("GAMaC Accessor Application")
    root.geometry("3200x2400")

    holder = ContextHolder(data_context.next_comparison())

    header = Label(font=("Arial", 20))
    header.pack()

    image = ttk.Label()
    image.pack(expand=True, anchor=CENTER)

    def update_ui():
        header['text'] = holder.context.label
        image['image'] = holder.context.current_image


    def swap(_):
        holder.context.swap()
        update_ui()


    def choose(_):
        data_context.save(holder.context)
        if data_context.has_next():
            holder.context = data_context.next_comparison()
            update_ui()
        else:
            data_context.persist()
            exit(0)

    root.bind('<KeyPress-Tab>', swap)
    root.bind('<KeyPress-Return>', choose)

    update_ui()

    root.mainloop()
