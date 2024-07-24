import os

from metacvi.collector import DatasetInfoCollector

NUM_SAMPLES = DatasetInfoCollector.PARTITIONS_TO_ESTIMATE
DATA_NAME, ACCESSOR_IDX = 'wine-quality-red/pca', 0


class ComparisonContext:
    def __init__(self, sorted_indices):
        self.insertion_idx = len(sorted_indices)
        self.sorted_indices = sorted_indices

        self.alternative = "ESTIMATE"

    def swap(self):
        self.alternative = "ESTIMATE" if self.alternative == "SORTED" else "SORTED"

    def shift(self):
        if self.insertion_idx > 0:
            self.insertion_idx -= 1

    def rendered_image_idx(self):
        return len(self.sorted_indices) if self.alternative == "ESTIMATE" else self.insertion_idx - 1

    def label(self):
        return f'{self.alternative} [POSITION {self.insertion_idx} / {len(self.sorted_indices)}]'

    def colour(self):
        return 'red' if self.alternative == "ESTIMATE" else 'blue'


class Application:
    def __init__(self):
        self.accessor_path = f'data/{DATA_NAME}/accessor-{ACCESSOR_IDX}.txt'
        if os.path.exists(self.accessor_path):
            raise FileExistsError(f"Data {DATA_NAME} has been already estimated by accessor {ACCESSOR_IDX}")
        self.sorted_estimates = [0]

    def current_idx(self):
        return len(self.sorted_estimates)

    def next_context(self) -> ComparisonContext:
        return ComparisonContext(self.sorted_estimates)

    def has_next(self):
        return self.current_idx() < NUM_SAMPLES

    def save(self, context: ComparisonContext):
        self.sorted_estimates.insert(context.insertion_idx, self.current_idx())

    def persist(self):
        with open(self.accessor_path, 'w') as fp:
            fp.write(self.sorted_estimates.__str__())


class ContextHolder:
    def __init__(self, context: ComparisonContext):
        self.context = context
        self.images = [self._photo_image(idx) for idx in range(NUM_SAMPLES)]

    def _photo_image(self, idx):
        return PhotoImage(file=f'data/{DATA_NAME}/img-{idx}.png')


if __name__ == '__main__':
    from tkinter import *

    root = Tk()
    root.title(f"GAMaC Assessment [{DATA_NAME}]")
    root.geometry("3200x2400")

    application = Application()
    holder = ContextHolder(application.next_context())

    header = Label(font=("Arial", 20))
    header.pack()

    image = Label()
    image.pack(expand=True, anchor=CENTER)

    def update_ui():
        header.config(
            text=holder.context.label(),
            foreground=holder.context.colour()
        )
        image_idx = holder.context.rendered_image_idx()
        image.config(
            highlightbackground=holder.context.colour(),
            highlightthickness=4,
            image=holder.images[image_idx]
        )

    def swap(_):
        holder.context.swap()
        update_ui()

    def shift(_):
        holder.context.shift()
        update_ui()

    def insert(_):
        application.save(holder.context)
        if application.has_next():
            holder.context = application.next_context()
            update_ui()
        else:
            application.persist()
            exit(0)

    root.bind('<KeyPress-Tab>', swap)
    root.bind('<KeyPress-Return>', insert)
    root.bind('<KeyPress-BackSpace>', shift)

    update_ui()

    root.mainloop()
